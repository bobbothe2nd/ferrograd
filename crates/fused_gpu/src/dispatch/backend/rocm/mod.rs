#![allow(unsafe_code)]

use briny::raw::cast::cast_slice;
use rocm_rt::{hip::{HipError, device::Device, memory::{Buffer, DevMapped}, module::{Func, LaunchConfig}, stream::Stream}, hiprtc::{HiprtcError, program::{CompileOptions, Hsaco}}, shared::GfxVersion};

use crate::{dispatch::{CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend, PollStatus, TargetCompilationOptions, TargetFlags, backend::{MetaId, NodeId, Param, kernel::{Dependencies, RawKernel, Redirect}, rocm::generate::generate_hip}}, errors::{Error, ErrorKind}};

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
}

impl GpuContext {
    pub async fn new() -> Result<Self, Error> {
        let device = map_err!(Device::current(), "failed to get current device")?;
        let arch = map_err!(device.gfx_version(), "failed to get GFX version from device")?;

        Ok(Self {
            device,
            arch,
        })
    }
}

impl GpuBackend for GpuContext {
    type BatchState = Stream;
    type Batcher<'a> = &'a Stream;
    type Buffer = Buffer;
    type MetaBuf = DevMapped;
    type Kernel = Kernel;
    type Schedule = Schedule;

    fn alloc(&self, len: u32) -> Result<Self::Buffer, Error> {
        Ok(map_err!(self.device.alloc(len), "failed to allocate GPU buffer")?)
    }

    fn alloc_init(&self, data: &[u8]) -> Result<Self::Buffer, Error> {
        let buf = self.alloc(data.len() as u32)?;

        let host_buf = map_err!(DevMapped::new(data), "failed to allocate host buffer")?;

        map_err!(unsafe {
            host_buf.copy_to_dev_unchecked(&buf, 0, 0, data.len())
        }, "failed copying host to device")?;

        Ok(buf)
    }

    fn alloc_meta(&self, data: &[u32]) -> Result<Self::MetaBuf, Error> {
        map_err!(DevMapped::new(cast_slice(data)), "failed to allocate host buffer")
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
        let func = map_err!(module.get_func_c(c"main"), "failed to get function from module")?;

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

        let host_buf = map_err!(DevMapped::alloc(len), "failed to allocate host buffer")?;

        map_err!(buffer.copy_to_host(&host_buf, 0, 0, len), "failed to copy")?;

        out.copy_from_slice(host_buf.as_slice());

        Ok(())
    }

    fn dispatch_kernel(
        &self,
        batcher: &mut Self::Batcher<'_>,
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
            map_err!(batcher.launch(&kernel.func, &mut kernel_args, LaunchConfig {
                grid,
                block: kernel.block,
            }), "failed to launch kernel")?;
        }

        Ok(())
    }

    fn dispatch_schedule(&self, _batcher: &mut Self::Batcher<'_>, _schedule: &Self::Schedule) -> Result<(), Error> {
        Ok(())
    }

    fn encode(&self, _state: Self::BatchState) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn prepare_batch(&self) -> Result<Self::BatchState, Error> {
        map_err!(self.device.set_default(), "failed to set default device")?;

        map_err!(Stream::create(), "failed to create stream")
    }

    fn start_batch<'a>(&self, state: &'a mut Self::BatchState) -> Self::Batcher<'a> {
        state
    }

    fn schedule(
        &self,
        _kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        _bindings: &[&Self::Buffer],
        _meta: &[u32],
    ) -> Result<Self::Schedule, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn poll(&self) -> PollStatus {
        PollStatus::Failed
    }

    fn sync(&self) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn upload(
        &self,
        _buffer: &Self::Buffer,
        _data: &[u8],
        _dst_off: u32,
    ) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn pipe(
        &self,
        _src: &Self::Buffer,
        _dst: &Self::Buffer,
    ) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }
}

impl GpuBufferBackend for Buffer {
    fn size(&self) -> u32 {
        (self.size() as u32) / 4
    }

    fn size_bytes(&self) -> u32 {
        self.size() as u32
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
            HipError::AlreadyAcquired
            | HipError::NotReady => ErrorKind::SyncError,
            HipError::AlreadyMapped
            | HipError::ArrayIsMapped
            | HipError::IllegalAddress
            | HipError::NotMapped
            | HipError::NotMappedAsArray
            | HipError::NotMappedAsPointer
            | HipError::RuntimeMemory
            | HipError::UnmapFailed => ErrorKind::InvalidMemoryOp,
            HipError::NoDevice
            | HipError::InvalidDevice
            | HipError::InvalidContext
            | HipError::ContextIsDestroyed => ErrorKind::InvalidDevice,
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
            | HipError::MissingConfiguration => ErrorKind::InvalidArgument,
            HipError::Assert
            | HipError::ECCNotCorrectable
            | HipError::GraphExecUpdateFailure
            | HipError::IllegalState
            | HipError::MapFailed
            | HipError::OperatingSystem
            | HipError::RuntimeOther
            | HipError::SetOnActiveProcess
            | HipError::Tbd
            | HipError::Unknown => ErrorKind::InternalError,
            HipError::CapturedEvent
            | HipError::StreamCaptureImplicit
            | HipError::StreamCaptureInvalidated
            | HipError::StreamCaptureIsolation
            | HipError::StreamCaptureMerge
            | HipError::StreamCaptureUnjoined
            | HipError::StreamCaptureUnmatched
            | HipError::StreamCaptureUnsupported
            | HipError::StreamCaptureWrongThread => ErrorKind::InconsistentCapture,
            HipError::ContextAlreadyCurrent
            | HipError::ContextAlreadyInUse
            | HipError::HostMemoryAlreadyRegistered => ErrorKind::AlreadySet,
            HipError::OutOfMemory
            | HipError::CooperativeLaunchTooLarge => ErrorKind::OutOfMemory,
            HipError::NotInitialized
            | HipError::Deinitialized
            | HipError::HostMemoryNotRegistered => ErrorKind::NotInitialized,
            HipError::InsufficientDriver
            | HipError::NotSupported => ErrorKind::UnsupportedFeature,
            HipError::FileNotFound => ErrorKind::FileNotFound,
            HipError::LaunchFailure
            | HipError::LaunchOutOfResources
            | HipError::LaunchTimeOut
            | HipError::NoBinaryForGpu
            | HipError::PriorLaunchFailure => ErrorKind::LaunchFailure,
            HipError::UnsupportedLimit => ErrorKind::UnsupportedLimit,
            HipError::PeerAccessAlreadyEnabled
            | HipError::PeerAccessNotEnabled
            | HipError::PeerAccessUnsupported => ErrorKind::PeerAccessError,
            HipError::ProfilerAlreadyStarted
            | HipError::ProfilerAlreadyStopped
            | HipError::ProfilerDisabled
            | HipError::ProfilerNotInitialized => ErrorKind::ProfilerError,
            HipError::SharedObjectInitFailed
            | HipError::SharedObjectSymbolNotFound
            | HipError::NotFound => ErrorKind::UnresolvedSymbol,
        }
    }
}

impl From<HiprtcError> for ErrorKind {
    fn from(_value: HiprtcError) -> Self {
        ErrorKind::InternalError
    }
}
