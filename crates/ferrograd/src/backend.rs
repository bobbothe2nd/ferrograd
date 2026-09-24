#![cfg_attr(not(feature = "wgsl"), allow(unreachable_patterns, unused_variables))]

use briny::traits::Pod;
use fused_gpu::{
    dispatch::{
        self, GpuContext as InnerCtx,
        backend::{
            NodeId, NopGpuContext,
            kernel::{Dependencies, RawKernel, Redirect, SaveIndicator},
        },
    },
    errors::Error,
    io::{BpatHeader, SerialTensorError},
    tensor::{Tensor, ToBuffer, bf16, f16},
};

use core::ops::{Deref, DerefMut};
use std::path::Path;

#[cfg(feature = "wgsl")]
use fused_gpu::dispatch::backend::wgsl;

#[cfg(feature = "rocm")]
use fused_gpu::dispatch::backend::rocm;

use crate::{
    dispatch::{
        AllocTensors, CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend,
        GpuKernelGroup, KernelGroup, OptimState, Param, TargetCompilationOptions,
    },
    graph::Graph,
};

pub struct SavedNodes(pub(crate) Vec<SaveIndicator>);

pub struct MetaBinding(pub(crate) [u32]);

pub struct Schedule<'a, B: GpuBackend = Dynamic>(dispatch::Schedule<'a, B>);

impl<'a, B: GpuBackend> Deref for Schedule<'a, B> {
    type Target = dispatch::Schedule<'a, B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: GpuBackend> DerefMut for Schedule<'_, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContext<B: GpuBackend = Dynamic>(InnerCtx<B>);

impl GpuContext<Dynamic> {
    /// Tries to create a context prioritizing ROCm -> WGSL
    #[inline]
    pub fn new() -> Result<Self, Error> {
        #[cfg(feature = "rocm")]
        {
            #[cfg(feature = "wgsl")]
            {
                if rocm::is_rocm_present() {
                    Self::new_rocm()
                } else {
                    Self::new_wgsl()
                }
            }
            #[cfg(not(feature = "wgsl"))]
            {
                Self::new_rocm()
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            #[cfg(feature = "wgsl")]
            {
                Self::new_wgsl()
            }
            #[cfg(not(feature = "wgsl"))]
            {
                Ok(Self::new_with_context(Dynamic::None(NopGpuContext::new()?)))
            }
        }
    }

    #[inline]
    #[cfg(feature = "rocm")]
    pub fn new_rocm() -> Result<Self, Error> {
        Ok(Self::new_with_context(Dynamic::Rocm(
            rocm::GpuContext::new()?,
        )))
    }

    #[inline]
    #[cfg(feature = "wgsl")]
    pub fn new_wgsl() -> Result<Self, Error> {
        Ok(Self::new_with_context(Dynamic::Wgsl(
            wgsl::GpuContext::new()?,
        )))
    }
}

impl<B: GpuBackend> GpuContext<B> {
    #[inline]
    #[must_use]
    pub const fn new_with_context(ctx: B) -> Self {
        Self(InnerCtx::new_with_context(ctx))
    }

    #[inline]
    #[must_use]
    pub const fn from_inner(inner: InnerCtx<B>) -> Self {
        Self(inner)
    }

    #[inline]
    #[must_use]
    pub fn into_inner(self) -> InnerCtx<B> {
        self.0
    }

    #[inline]
    pub fn as_inner(&self) -> &InnerCtx<B> {
        &self.0
    }

    #[inline]
    pub fn as_inner_mut(&mut self) -> &mut InnerCtx<B> {
        &mut self.0
    }

    #[inline]
    pub fn load_tensors<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<Tensor<B>>, Error<SerialTensorError>> {
        self.0.load_tensors(path)
    }

    #[inline]
    pub fn save_tensors<P: AsRef<Path>>(
        &self,
        path: P,
        tensors: &[Tensor<B>],
        header: BpatHeader,
    ) -> Result<(), Error<SerialTensorError>> {
        self.0.save_tensors(path, tensors, header)
    }

    #[inline]
    pub fn alloc_tensors(
        &self,
        graph: &Graph<'_>,
        saved: &SavedNodes,
        meta: &MetaBinding,
        state: &OptimState,
    ) -> Result<AllocTensors<B>, Error> {
        self.0.alloc_tensors(&graph.0, &saved.0, &meta.0, state)
    }

    #[inline]
    #[must_use]
    pub fn alloc_meta(&self, meta: &[u32]) -> &MetaBinding {
        unsafe { core::mem::transmute::<&[u32], &MetaBinding>(meta) }
    }

    #[inline]
    pub fn schedule<'a>(
        &self,
        kernels: GpuKernelGroup<B>,
        meta: &MetaBinding,
        in_tensors: &'a [Tensor<B>],
        alloc_tensors: &'a AllocTensors<B>,
        state: &'a [B::Buffer],
        _optim: &OptimState,
    ) -> Result<Schedule<'a, B>, Error> {
        Ok(Schedule(self.0.schedule(
            kernels,
            &meta.0,
            in_tensors,
            alloc_tensors,
            state,
        )?))
    }

    #[inline]
    #[must_use]
    pub fn detect_target(&self) -> TargetCompilationOptions {
        self.0.detect_target()
    }

    pub fn sync(&self) -> Result<(), Error> {
        self.0.sync()
    }

    pub fn dispatch_forward(&self, schedule: &Schedule<'_, B>) -> Result<(), Error> {
        self.0.dispatch_forward(schedule)
    }

    pub fn dispatch_backward(&self, schedule: &Schedule<'_, B>) -> Result<(), Error> {
        self.0.dispatch_backward(schedule)
    }

    pub fn dispatch_loss(
        &self,
        schedule: &mut Schedule<'_, B>,
        target: &Tensor<B>,
    ) -> Result<(), Error> {
        self.0.dispatch_loss(schedule, target)
    }

    pub fn dispatch_optim(
        &self,
        schedule: &mut Schedule<'_, B>,
        weight: &Tensor<B>,
        grad: usize,
        tensors: &AllocTensors<B>,
    ) -> Result<(), Error> {
        self.0.dispatch_optim(schedule, weight, grad, tensors)
    }

    #[inline]
    pub fn download<T: Pod, S: ToBuffer<B>>(&self, tensor: &S, dst: &mut [T]) -> Result<(), Error> {
        self.0.download(tensor, dst)
    }

    #[inline]
    pub fn upload<T: Pod, S: ToBuffer<B>>(
        &self,
        tensor: &S,
        src: &[T],
        src_off: u32,
        dst_off: u32,
    ) -> Result<(), Error> {
        self.0.upload(tensor, src, src_off, dst_off)
    }

    #[inline]
    pub fn copy<S1: ToBuffer<B>, S2: ToBuffer<B>>(&self, src: &S1, dst: &S2) -> Result<(), Error> {
        self.0.copy(src, dst)
    }

    #[inline]
    pub fn compile<'a>(
        &self,
        ir: &'a KernelGroup<'a>,
        options: &CompilationOptions,
    ) -> Result<GpuKernelGroup<'a, B>, Error> {
        self.0.compile(ir, options)
    }

    #[inline]
    pub fn init_tensor_bf16(&self, shape: Vec<u32>, data: &[bf16]) -> Result<Tensor<B>, Error> {
        self.0.init_tensor_bf16(shape, data)
    }

    #[inline]
    pub fn init_tensor_f16(&self, shape: Vec<u32>, data: &[f16]) -> Result<Tensor<B>, Error> {
        self.0.init_tensor_f16(shape, data)
    }

    #[inline]
    pub fn init_tensor_f32(&self, shape: Vec<u32>, data: &[f32]) -> Result<Tensor<B>, Error> {
        self.0.init_tensor_f32(shape, data)
    }

    #[inline]
    pub fn init_tensor_f64(&self, shape: Vec<u32>, data: &[f64]) -> Result<Tensor<B>, Error> {
        self.0.init_tensor_f64(shape, data)
    }

    #[inline]
    pub fn new_onehot(&self, classes: u32) -> Result<Tensor<B>, Error> {
        self.0.new_onehot(classes)
    }

    #[inline]
    pub fn init_onehot(&self, indices: &[u32]) -> Result<Tensor<B>, Error> {
        self.0.new_onehot_init(indices)
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Dynamic {
    None(NopGpuContext),
    #[cfg(feature = "wgsl")]
    Wgsl(wgsl::GpuContext),
    #[cfg(feature = "rocm")]
    Rocm(rocm::GpuContext)
}

macro_rules! impl_op {
    ($($op:ident(& $([$qualifier:ty])? self, $($arg:ident: $ty:ty),*) -> Result<$out:ty, $err:ty>)*) => {
        $(
            #[inline]
            fn $op(&$($qualifier)?self, $($arg: $ty),*) -> Result<$out, $err> {
                Ok(match self {
                    Self::None(ctx) => ctx.$op($($arg.into()),*)?.into(),
                    #[cfg(feature = "wgsl")]
                    Self::Wgsl(ctx) => ctx.$op($($arg.into()),*)?.into(),
                    #[cfg(feature = "rocm")]
                    Self::Rocm(ctx) => ctx.$op($($arg.into()),*)?.into(),
                })
            }
        )*
    };

    ($($op:ident(& $([$qualifier:ty])? self, $($arg:ident: $ty:ty),*) -> $out:ty)*) => {
        $(
            #[inline]
            fn $op(&$($qualifier)?self, $($arg: $ty),*) -> $out {
                match self {
                    Self::None(ctx) => ctx.$op($($arg.into()),*).into(),
                    #[cfg(feature = "wgsl")]
                    Self::Wgsl(ctx) => ctx.$op($($arg.into()),*).into(),
                    #[cfg(feature = "rocm")]
                    Self::Rocm(ctx) => ctx.$op($($arg.into()),*).into(),
                }
            }
        )*
    };
}

impl GpuBackend for Dynamic {
    type Buffer = DynBuffer;
    type Kernel = DynKernel;
    type Schedule = DynSchedule;
    type MetaBuf = DynMetaBuf;

    impl_op! {
        target_spec(&self,) -> TargetCompilationOptions
    }

    impl_op! {
        sync(&self,) -> Result<(), Error>
        is_ready(&self,) -> Result<bool, Error>
        upload(
            &self,
            buffer: &Self::Buffer,
            data: &[u8],
            src_off: u32,
            dst_off: u32
        ) -> Result<(), Error>
        copy(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Error>
        compile(
            &self,
            src: &RawKernel,
            params: &[Param],
            options: &CompilationOptions
        ) -> Result<Self::Kernel, Error>
        download(&self, buffer: &Self::Buffer, data: &mut [u8]) -> Result<(), Error>
        dispatch_schedule(&self, schedule: &Self::Schedule) -> Result<(), Error>
        alloc(&self, len: u32) -> Result<Self::Buffer, Error>
        alloc_init(&self, init: &[u8]) -> Result<Self::Buffer, Error>
        alloc_meta(&self, data: &[u32]) -> Result<Self::MetaBuf, Error>
    }

    fn dispatch_kernel(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
        meta: &Self::MetaBuf,
    ) -> Result<(), Error> {
        match self {
            Self::None(_) => Ok(()),
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => {
                let bindings = bindings.iter().copied().map(Into::into).collect::<Vec<_>>();

                ctx.dispatch_kernel(kernel.into(), wg, &bindings, meta.into())
            }
            #[cfg(feature = "rocm")]
            Self::Rocm(ctx) => {
                let bindings = bindings.iter().copied().map(Into::into).collect::<Vec<_>>();

                ctx.dispatch_kernel(kernel.into(), wg, &bindings, meta.into())
            }
        }
    }

    #[inline]
    fn schedule(
        &self,
        kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        bindings: &[&Self::Buffer],
        meta: &[u32],
        meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule, Error> {
        Ok(match self {
            Self::None(_) => ().into(),
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => {
                let kernels = kernels
                    .into_iter()
                    .map(|x| Dependencies {
                        val: match x.val {
                            Redirect::Redirected(node) => Redirect::Redirected(node),
                            Redirect::Unmasked(val) => {
                                Redirect::Unmasked((val.0.into(), val.1, val.2))
                            }
                        },
                        dep: x.dep,
                    })
                    .collect::<Vec<_>>();
                let bindings = bindings.iter().cloned().map(Into::into).collect::<Vec<_>>();

                ctx.schedule(kernels, &bindings, meta, meta_buf.into())?
                    .into()
            }
            #[cfg(feature = "rocm")]
            Self::Rocm(ctx) => {
                let kernels = kernels
                    .into_iter()
                    .map(|x| Dependencies {
                        val: match x.val {
                            Redirect::Redirected(node) => Redirect::Redirected(node),
                            Redirect::Unmasked(val) => {
                                Redirect::Unmasked((val.0.into(), val.1, val.2))
                            }
                        },
                        dep: x.dep,
                    })
                    .collect::<Vec<_>>();
                let bindings = bindings.iter().cloned().map(Into::into).collect::<Vec<_>>();

                ctx.schedule(kernels, &bindings, meta, meta_buf.into())?
                    .into()
            }
        })
    }
}

macro_rules! impl_backend {
    ($(#[$meta:meta])? $name:ident$(<$($lifetime:lifetime),*>)?, $nop:ident, $wgsl:ident, $rocm:ident$(,)?) => {
        #[non_exhaustive]
        #[allow(clippy::large_enum_variant)]
        $(#[$meta])?
        pub enum $name$(<$($lifetime),*>)? {
            None($nop, $(PhantomData<$(&$lifetime ()),*>)?),
            #[cfg(feature = "wgsl")]
            Wgsl(wgsl::$wgsl$(<$($lifetime),*>)?),
            #[cfg(feature = "rocm")]
            Rocm(rocm::$rocm$(<$($lifetime),*>)?),
        }

        impl_backend!(@backend $name, $nop$(<$($lifetime),*>)? => None, self, "nop");
        impl_backend!(@backend $name, $wgsl$(<$($lifetime),*>)? => Wgsl, wgsl, "WGSL", "wgsl");
        impl_backend!(@backend $name, $rocm$(<$($lifetime),*>)? => Rocm, rocm, "ROCm", "rocm");
    };

    (@backend $on_name:ident$(<$($lifetime:lifetime),*>)?, $name:ident => $backend:ident, $module:ident, $backend_fmt:literal$(, $feature:literal)?$(,)?) => {
        $(#[cfg(feature = $feature)])?
        impl$(<$($lifetime),*>)? From<$module::$name$(<$($lifetime),*>)?> for $on_name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: $module::$name$(<$($lifetime),*>)?) -> Self {
                Self::$backend(value)
            }
        }

        $(#[cfg(feature = $feature)])?
        impl$(<$($lifetime),*>)? From<$on_name$(<$($lifetime),*>)?> for $module::$name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: $on_name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $on_name::$backend(ctx) => ctx,
                    _ => panic!(concat!("unsupported operation for ", $backend_fmt, " backend")),
                }
            }
        }

        $(#[cfg(feature = $feature)])?
        impl<'__a, $($($lifetime),*)?> From<&'__a $on_name$(<$($lifetime),*>)?> for &'__a $module::$name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: &'__a $on_name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $on_name::$backend(ctx) => ctx,
                    _ => panic!(concat!("unsupported operation for ", $backend_fmt, " backend")),
                }
            }
        }

        $(#[cfg(feature = $feature)])?
        impl<'__a, $($($lifetime),*)?> From<&'__a mut $on_name$(<$($lifetime),*>)?> for &'__a mut $module::$name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: &'__a mut $on_name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $on_name::$backend(ctx) => ctx,
                    _ => panic!(concat!("unsupported operation for ", $backend_fmt, " backend")),
                }
            }
        }
    }
}

type Unit = ();

impl_backend!(
    #[derive(Debug)]
    DynBuffer,
    Unit,
    Buffer,
    Buffer,
);

impl GpuBufferBackend for DynBuffer {
    fn size(&self) -> u32 {
        match self {
            Self::None(_) => 0,
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => ctx.size() as u32,
            #[cfg(feature = "rocm")]
            Self::Rocm(ctx) => ctx.size() as u32,
        }
    }

    fn size_bytes(&self) -> u32 {
        match self {
            Self::None(_) => 0,
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => ctx.size_bytes(),
            #[cfg(feature = "rocm")]
            Self::Rocm(ctx) => ctx.size_bytes(),
        }
    }
}

impl ToBuffer<Dynamic> for DynBuffer {
    fn as_buffer(&self) -> &<Dynamic as GpuBackend>::Buffer {
        self
    }

    fn to_buffer(self) -> <Dynamic as GpuBackend>::Buffer {
        self
    }
}

impl_backend!(
    DynKernel,
    Unit,
    GpuKernel,
    Kernel,
);

impl GpuKernelBackend for DynKernel {
    impl_op! {
        iteration_space(&self,) -> &[dispatch::backend::MetaId]
        block(&self,) -> &[u32; 3]
    }
}

impl_backend!(
    DynSchedule,
    Unit,
    Schedule,
    Graph,
);
impl_backend!(
    DynMetaBuf,
    Unit,
    Buffer,
    DevMappedAlloc,
);
