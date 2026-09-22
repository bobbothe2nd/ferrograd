#![allow(unsafe_code)]

use briny::raw::cast::cast_slice;
use rocm_rt::{
    hip::{
        HipError, device::Device, memory::{Buffer, DevMapped, DevMappedAlloc}, module::{Func, LaunchConfig}, stream::Stream,
    }, hiprtc::{
        HiprtcError,
        program::{CompileOptions, Hsaco},
    }, shared::GfxVersion,
};

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend,
        TargetCompilationOptions, TargetFlags,
        backend::{
            MetaId, NodeId, Param,
            kernel::{Dependencies, RawKernel, Redirect},
            rocm::generate::generate_hip,
        },
    },
    errors::{Error, ErrorKind},
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
    type Schedule = Schedule;

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
        let hip = generate_hip(src, params, options);

        let opts = CompileOptions {
            opt_level: Some(3),
            fast_math: Some(true),
            name: None,
            options: &[],
            defines: &[],
            include_paths: &[],
            arch: self.arch,
        };

        let hsaco = map_err!(Hsaco::compile(hip, &opts), "failed to compile HSACO binary")?;
        let module = map_err!(hsaco.load(), "failed to load module from binary")?;
        let func = map_err!(
            module.get_func_c(c"main"),
            "failed to get function from module"
        )?;

        Ok(Kernel {
            block: src.block,
            iter_space: src.iter_space.clone(),
            func,
        })
    }

    fn target_spec(&self) -> TargetCompilationOptions {
        TargetCompilationOptions {
            flags: TargetFlags::empty(),
        }
    }

    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error> {
        let len = out.len();

        let host_buf = map_err!(DevMapped::new(out), "failed to allocate host buffer")?;

        map_err!(buffer.copy_to_host(&host_buf, 0, 0, len), "failed to copy")?;

        Ok(())
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
            )?;
        }

        Ok(())
    }

    fn dispatch_schedule(&self, _schedule: &Self::Schedule) -> Result<(), Error> {
        Ok(())
    }

    fn schedule(
        &self,
        _kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        _bindings: &[&Self::Buffer],
        _meta: &[u32],
        _meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn is_ready(&self) -> Result<bool, Error> {
        map_err!(self.stream.query(), "failed to query stream for completion")
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
        let host_buf = unsafe {
            DevMapped::from_slice(data)
        };

        unsafe {
            map_err!(host_buf.map(), "failed to register data in upload")?;
        }

        map_err!(host_buf.copy_to_dev(buffer, src_off as usize, dst_off as usize, data.len()), "failed to copy host to GPU device")?;

        Ok(())
    }

    fn copy(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Error> {
        map_err!(src.copy_to(dst, 0, 0, src.size() as usize), "failed to copy GPU buffer")?;

        Ok(())
    }
}

impl GpuBufferBackend for Buffer {
    fn size(&self) -> u32 {
        self.size() / 4
    }

    fn size_bytes(&self) -> u32 {
        self.size()
    }
}

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

pub struct Schedule {}

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
    fn from(_value: HiprtcError) -> Self {
        Self::InternalError
    }
}
