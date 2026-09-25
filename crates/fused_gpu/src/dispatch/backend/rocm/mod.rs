#![allow(unsafe_code)]

use briny::raw::cast::cast_slice;
use rocm_rt::shared::MatrixKind;
pub use rocm_rt::{
    hip::{
        HipError,
        device::Device,
        graph::{Graph, GraphInstantiateFlags, KernelParams},
        memory::{Buffer, DevMapped, DevMappedAlloc},
        module::{Func, LaunchConfig},
        stream::Stream,
    },
    hiprtc::{
        HiprtcError,
        program::{CompileOptions, Hsaco},
    },
    shared::GfxVersion,
};

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend,
        TargetCompilationOptions, TargetFlags,
        backend::{
            MetaId, NodeId, Param,
            kernel::{Dependencies, RawKernel, Redirect, topo_sort},
            rocm::generate::generate_hip,
        },
    },
    errors::{Error, ErrorKind},
    tensor::{build_dims, calc_grid},
};

mod generate;

macro_rules! map_err {
    ($func:expr, $msg:expr$(,)?) => {
        ($func).map_err(|e| Error {
            msg: $msg,
            kind: e.into(),
            ctx: (),
        })
    };
}

pub fn is_rocm_present() -> bool {
    rocm_rt::is_amdhip64_present() && rocm_rt::is_hiprtc_present()
}

#[derive(Debug)]
pub struct GpuContext {
    device: Device,
    arch: GfxVersion,
    stream: Stream,
}

impl GpuContext {
    pub fn new() -> Result<Self, Error> {
        let device = map_err!(Device::current(), "failed to get current device")?;
        let arch = map_err!(
            device.gfx_version(),
            "failed to get GFX version from device"
        )?;
        let stream = map_err!(Stream::create(), "failed to get GFX version from device")?;

        Ok(Self {
            device,
            arch,
            stream,
        })
    }
}

impl GpuBackend for GpuContext {
    type Buffer = Buffer;
    type MetaBuf = DevMappedAlloc;
    type Kernel = Kernel;
    type Schedule = Graph;

    fn alloc(&self, len: u32) -> Result<Self::Buffer, Error> {
        Ok(map_err!(
            self.device.alloc(len),
            "failed to allocate GPU buffer"
        )?)
    }

    fn alloc_init(&self, data: &[u8]) -> Result<Self::Buffer, Error> {
        let buf = self.alloc(data.len() as u32)?;

        let host_buf = unsafe { DevMapped::from_slice(data) };

        map_err!(
            host_buf.copy_to_dev(&buf, 0, 0, data.len()),
            "failed copying host to device"
        )?;

        Ok(buf)
    }

    fn alloc_meta(&self, data: &[u32]) -> Result<Self::MetaBuf, Error> {
        map_err!(
            DevMappedAlloc::new(cast_slice(data)),
            "failed to allocate host buffer"
        )
    }

    fn compile(
        &self,
        src: &RawKernel,
        params: &[Param],
        options: &CompilationOptions,
    ) -> Result<Self::Kernel, Error> {
        let hip = generate_hip(src, params, options)?;

        let opts = CompileOptions {
            opt_level: Some(3),
            fast_math: Some(true),
            name: Some(c"kernel"),
            options: &[],
            defines: &[],
            include_paths: &[],
            arch: self.arch,
        };

        let hsaco = map_err!(Hsaco::compile(hip, &opts), "failed to compile HSACO binary")?;
        let module = map_err!(hsaco.load(), "failed to load module from binary")?;
        let func = map_err!(
            module.get_func_c(c"kernel"),
            "failed to get `kernel` function from module"
        )?;

        Ok(Kernel {
            block: src.block,
            iter_space: src.iter_space.clone(),
            func,
        })
    }

    fn target_spec(&self) -> TargetCompilationOptions {
        let mut flags = TargetFlags::empty();

        let capabilities = self.arch.matrix_capabilities();

        if capabilities.kind != MatrixKind::None {
            flags |= TargetFlags::LIN_ACC;
        }

        TargetCompilationOptions {
            flags,
        }
    }

    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error> {
        let len = out.len();

        let host_buf = map_err!(DevMapped::new(out), "failed to allocate host buffer")?;

        map_err!(buffer.copy_to_host(&host_buf, 0, 0, len), "failed to copy")
    }

    fn dispatch_kernel(
        &self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        bindings: &[&Self::Buffer],
        meta: &Self::MetaBuf,
    ) -> Result<(), Error> {
        let mut kernel_args = Vec::with_capacity(1 + bindings.len());

        kernel_args.push((&raw const *meta) as *mut u8);

        for binding in bindings {
            kernel_args.push((&raw const **binding) as *mut u8);
        }

        unsafe {
            map_err!(
                self.stream.launch(
                    &kernel.func,
                    &mut kernel_args,
                    LaunchConfig {
                        grid,
                        block: kernel.block,
                    }
                ),
                "failed to launch kernel"
            )
        }
    }

    fn dispatch_schedule(&self, graph: &Self::Schedule) -> Result<(), Error> {
        unsafe { map_err!(self.stream.launch_graph(graph), "failed to launch graph") }
    }

    fn schedule(
        &self,
        kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        bindings: &[&Self::Buffer],
        meta: &[u32],
        meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule, Error> {
        let kernels = topo_sort(&kernels)?;

        let mut graph = map_err!(
            Graph::new(GraphInstantiateFlags::DeviceLaunch),
            "failed to instantiate graph"
        )?;

        let mut nodes = vec![None; kernels.len()];

        for (idx, kernel) in kernels.iter().enumerate() {
            let dep = kernel
                .dep
                .iter()
                .filter_map(|dep| nodes[*dep])
                .collect::<Vec<_>>();
            let kernel = match &kernel.val {
                Redirect::Redirected(kernel_id) => match &kernels[*kernel_id].val {
                    Redirect::Redirected(_) => {
                        return Err(Error {
                            msg: "double redirection or loop encountered in kernel resolution",
                            kind: ErrorKind::UnresolvedRedirection,
                            ctx: (),
                        });
                    }
                    Redirect::Unmasked(func) => func.0.clone(),
                },
                Redirect::Unmasked(func) => func.0.clone(),
            };

            let mut kernel_args = Vec::with_capacity(1 + bindings.len());

            kernel_args.push((&raw const *meta_buf) as *mut u8);

            for binding in bindings {
                kernel_args.push((&raw const **binding) as *mut u8);
            }

            let iter_space = build_dims(kernel.iteration_space(), meta);
            let grid = calc_grid(&iter_space, *kernel.block());

            let params = KernelParams::new(
                &kernel.func,
                &mut kernel_args,
                LaunchConfig {
                    grid,
                    block: kernel.block,
                },
            );

            nodes[idx].replace(map_err!(
                graph.add_kernel_node(&dep, &params),
                "failed to add kernel node to graph"
            )?);
        }

        Ok(graph)
    }

    fn is_ready(&self) -> Result<bool, Error> {
        map_err!(
            self.stream.is_ready(),
            "failed to query stream for completion"
        )
    }

    fn sync(&self) -> Result<(), Error> {
        map_err!(self.stream.sync(), "failed to synchronize stream")
    }

    fn upload(
        &self,
        buffer: &Self::Buffer,
        data: &[u8],
        src_off: u32,
        dst_off: u32,
    ) -> Result<(), Error> {
        let host_buf = unsafe { DevMapped::from_slice(data) };

        unsafe {
            map_err!(host_buf.map(), "failed to register data in upload")?;
        }

        map_err!(
            host_buf.copy_to_dev(buffer, src_off as usize, dst_off as usize, data.len()),
            "failed to copy host to GPU device"
        )
    }

    fn copy(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Error> {
        map_err!(
            src.copy_to(dst, 0, 0, src.size() as usize),
            "failed to copy GPU buffer"
        )
    }
}

impl GpuBufferBackend for Buffer {
    fn size(&self) -> u32 {
        self.size().div_ceil(4)
    }

    fn size_bytes(&self) -> u32 {
        self.size()
    }
}

#[derive(Clone)]
pub struct Kernel {
    block: [u32; 3],
    iter_space: Vec<MetaId>,
    func: Func,
}

impl GpuKernelBackend for Kernel {
    fn block(&self) -> &[u32; 3] {
        &self.block
    }

    fn iteration_space(&self) -> &[MetaId] {
        &self.iter_space
    }
}

impl From<HipError> for ErrorKind {
    fn from(value: HipError) -> Self {
        match value {
            HipError::AlreadyAcquired | HipError::NotReady => Self::SyncError,
            HipError::AlreadyMapped
            | HipError::ArrayIsMapped
            | HipError::IllegalAddress
            | HipError::NotMapped
            | HipError::NotMappedAsArray
            | HipError::NotMappedAsPointer
            | HipError::RuntimeMemory
            | HipError::UnmapFailed => Self::InvalidMemoryOp,
            HipError::NoDevice
            | HipError::InvalidDevice
            | HipError::InvalidContext
            | HipError::ContextIsDestroyed => Self::InvalidDevice,
            HipError::InvalidConfiguration
            | HipError::InvalidValue
            | HipError::InvalidChannelDescriptor
            | HipError::InvalidDeviceFunction
            | HipError::InvalidDevicePointer
            | HipError::InvalidGraphicsContext
            | HipError::InvalidHandle
            | HipError::InvalidImage
            | HipError::InvalidKernelFile
            | HipError::InvalidMemcpyDirection
            | HipError::InvalidPitchValue
            | HipError::InvalidSource
            | HipError::InvalidSymbol
            | HipError::InvalidTexture
            | HipError::MissingConfiguration => Self::InvalidArgument,
            HipError::Assert
            | HipError::ECCNotCorrectable
            | HipError::GraphExecUpdateFailure
            | HipError::IllegalState
            | HipError::MapFailed
            | HipError::OperatingSystem
            | HipError::RuntimeOther
            | HipError::SetOnActiveProcess
            | HipError::Tbd
            | HipError::Unknown => Self::InternalError,
            HipError::CapturedEvent
            | HipError::StreamCaptureImplicit
            | HipError::StreamCaptureInvalidated
            | HipError::StreamCaptureIsolation
            | HipError::StreamCaptureMerge
            | HipError::StreamCaptureUnjoined
            | HipError::StreamCaptureUnmatched
            | HipError::StreamCaptureUnsupported
            | HipError::StreamCaptureWrongThread => Self::InconsistentCapture,
            HipError::ContextAlreadyCurrent
            | HipError::ContextAlreadyInUse
            | HipError::HostMemoryAlreadyRegistered => Self::AlreadySet,
            HipError::OutOfMemory | HipError::CooperativeLaunchTooLarge => Self::OutOfMemory,
            HipError::NotInitialized
            | HipError::Deinitialized
            | HipError::HostMemoryNotRegistered => Self::NotInitialized,
            HipError::InsufficientDriver | HipError::NotSupported => Self::UnsupportedFeature,
            HipError::FileNotFound => Self::FileNotFound,
            HipError::LaunchFailure
            | HipError::LaunchOutOfResources
            | HipError::LaunchTimeOut
            | HipError::NoBinaryForGpu
            | HipError::PriorLaunchFailure => Self::LaunchFailure,
            HipError::UnsupportedLimit => Self::UnsupportedLimit,
            HipError::PeerAccessAlreadyEnabled
            | HipError::PeerAccessNotEnabled
            | HipError::PeerAccessUnsupported => Self::PeerAccessError,
            HipError::ProfilerAlreadyStarted
            | HipError::ProfilerAlreadyStopped
            | HipError::ProfilerDisabled
            | HipError::ProfilerNotInitialized => Self::ProfilerError,
            HipError::SharedObjectInitFailed
            | HipError::SharedObjectSymbolNotFound
            | HipError::NotFound => Self::UnresolvedSymbol,
        }
    }
}

impl From<HiprtcError> for ErrorKind {
    fn from(value: HiprtcError) -> Self {
        match value {
            HiprtcError::BuiltinOperationFailure
            | HiprtcError::Compilation
            | HiprtcError::InternalError
            | HiprtcError::NameExpressionNotValid
            | HiprtcError::NoLoweredNamesBeforeCompilation
            | HiprtcError::NoNameExpressionsAfterCompilation
            | HiprtcError::ProgramCreationFailure => Self::InternalError,
            HiprtcError::OutOfMemory => Self::OutOfMemory,
            HiprtcError::Linking => Self::LinkingError,
            HiprtcError::InvalidInput
            | HiprtcError::InvalidOption
            | HiprtcError::InvalidProgram => Self::InvalidArgument,
        }
    }
}
