#![cfg_attr(not(feature = "wgsl"), allow(unreachable_patterns, unused_variables))]

use briny::traits::Pod;
use fused_gpu::{
    dispatch::{
        AllocTensors, CompilationOptions, GpuBackend, GpuBufferBackend, GpuContext as InnerCtx,
        GpuKernelBackend, PollStatus, TargetCompilationOptions,
        backend::{
            NodeId, NopGpuBuffer, NopGpuContext, NopGpuKernel, OptimState, Param,
            kernel::{Dependencies, RawKernel, Redirect, SaveIndicator},
        },
    },
    errors::Error,
    io::{BpatHeader, SerialTensorError},
    tensor::{Tensor, ToBuffer, bf16, f16},
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::Path,
};

#[cfg(feature = "wgsl")]
use fused_gpu::dispatch::backend::wgsl;

use crate::{
    dispatch::{self, GpuKernelGroup, KernelGroup},
    graph::Graph,
};

pub struct SavedNodes(pub(crate) Vec<SaveIndicator>);

pub struct MetaBinding(pub(crate) [u32]);

pub struct Schedule<'a, B: GpuBackend = Dynamic>(fused_gpu::dispatch::Schedule<'a, B>);

impl<'a, B: GpuBackend> Deref for Schedule<'a, B> {
    type Target = fused_gpu::dispatch::Schedule<'a, B>;

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
    #[inline]
    pub fn new() -> Result<Self, Error> {
        pollster::block_on(Self::new_nonblocking())
    }

    #[inline]
    pub async fn new_nonblocking() -> Result<Self, Error> {
        #[cfg(feature = "wgsl")]
        {
            Ok(Self::new_with_context(Dynamic::Wgsl(
                wgsl::GpuContext::new().await?,
            )))
        }
        #[cfg(not(feature = "wgsl"))]
        {
            Ok(Self::new_with_context(Dynamic::None(
                NopGpuContext::new().await?,
            )))
        }
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
    #[must_use]
    pub fn alloc_tensors<const N: usize>(
        &self,
        graph: &Graph<'_>,
        saved: &SavedNodes,
        meta: &MetaBinding,
        state: &OptimState<N>,
    ) -> AllocTensors<B> {
        self.0.alloc_tensors(&graph.0, &saved.0, &meta.0, state)
    }

    #[inline]
    #[must_use]
    pub fn alloc_meta(&self, meta: &[u32]) -> &MetaBinding {
        unsafe { core::mem::transmute::<&[u32], &MetaBinding>(meta) }
    }

    #[inline]
    pub fn prepare_batch(&self) -> dispatch::BatchState<'_, B> {
        self.0.prepare_batch()
    }

    #[inline]
    pub fn start_batch<'a>(
        &'a self,
        state: &'a mut dispatch::BatchState<'_, B>,
    ) -> dispatch::Batcher<'a, B> {
        self.0.start_batch(state)
    }

    #[inline]
    pub fn schedule<'a, const N: usize>(
        &self,
        kernels: GpuKernelGroup<B>,
        meta: &MetaBinding,
        in_tensors: &'a [Tensor<B>],
        alloc_tensors: &'a AllocTensors<B>,
        state: &'a [B::Buffer],
        _optim: &OptimState<N>,
    ) -> Result<Schedule<'a, B>, Error> {
        Ok(Schedule(self.0.schedule::<N>(
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

    #[inline]
    pub fn download<T: Pod, S: ToBuffer<B>>(&self, tensor: &S, dst: &mut [T]) -> Result<(), Error> {
        self.0.download(tensor, dst)
    }

    #[inline]
    pub fn upload<T: Pod, S: ToBuffer<B>>(
        &self,
        tensor: &S,
        src: &[T],
        dst_off: u32,
    ) -> Result<dispatch::SubmissionIndex<'_, B>, Error> {
        self.0.upload(tensor, src, dst_off)
    }

    #[inline]
    pub fn pipe<S1: ToBuffer<B>, S2: ToBuffer<B>>(
        &self,
        src: &S1,
        dst: &S2,
    ) -> Result<dispatch::SubmissionIndex<'_, B>, Error> {
        self.0.pipe(src, dst)
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
    #[must_use]
    pub fn init_tensor_bf16(&self, shape: Vec<u32>, data: &[bf16]) -> Tensor<B> {
        self.0.init_tensor_bf16(shape, data)
    }

    #[inline]
    #[must_use]
    pub fn init_tensor_f16(&self, shape: Vec<u32>, data: &[f16]) -> Tensor<B> {
        self.0.init_tensor_f16(shape, data)
    }

    #[inline]
    #[must_use]
    pub fn init_tensor_f32(&self, shape: Vec<u32>, data: &[f32]) -> Tensor<B> {
        self.0.init_tensor_f32(shape, data)
    }

    #[inline]
    #[must_use]
    pub fn init_tensor_f64(&self, shape: Vec<u32>, data: &[f64]) -> Tensor<B> {
        self.0.init_tensor_f64(shape, data)
    }

    #[inline]
    #[must_use]
    pub fn new_onehot(&self, classes: u32) -> Tensor<B> {
        self.0.new_onehot(classes)
    }

    #[inline]
    #[must_use]
    pub fn init_onehot(&self, indices: &[u32]) -> Tensor<B> {
        self.0.new_onehot_init(indices)
    }
}

#[derive(Debug)]
pub enum Dynamic {
    None(NopGpuContext),
    #[cfg(feature = "wgsl")]
    Wgsl(wgsl::GpuContext),
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
                }
            }
        )*
    };
}

impl GpuBackend for Dynamic {
    type Buffer = DynBuffer;
    type BatchState = DynBatchState;
    type Batcher<'a> = DynBatcher<'a>;
    type Kernel = DynKernel;
    type Schedule = DynSchedule;
    type SyncSubmissions = DynSyncSubmissions;
    type SubmissionIndex = DynSubmissionIndex;
    type ParamLayout = DynParamLayout;

    impl_op! {
        target_spec(&self,) -> TargetCompilationOptions
        alloc(&self, len: usize) -> Self::Buffer
        alloc_init(&self, init: &[u8]) -> Self::Buffer
        alloc_meta(&self, data: &[u32]) -> Self::Buffer
        prepare_batch(&self,) -> Self::BatchState
        dispatch_schedule(&self, pass: &mut Self::Batcher<'_>, schedule: &Self::Schedule) -> ()
        encode(&self, state: Self::BatchState) -> Self::SyncSubmissions
        sync(&self, submission_index: Self::SubmissionIndex) -> ()
        submit(&self, submission: Self::SyncSubmissions) -> Self::SubmissionIndex
        poll(&self,) -> PollStatus
    }

    impl_op! {
        upload(
            &self,
            buffer: &Self::Buffer,
            data: &[u8],
            dst_off: u32
        ) -> Result<Self::SubmissionIndex, Error>
        pipe(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<Self::SubmissionIndex, Error>
        compile(
            &self,
            src: &RawKernel,
            params: &[Param],
            options: &CompilationOptions
        ) -> Result<Self::Kernel, Error>
        download(&self, buffer: &Self::Buffer, data: &mut [u8]) -> Result<(), Error>
    }

    fn start_batch<'a>(&self, state: &'a mut Self::BatchState) -> Self::Batcher<'a> {
        match self {
            Self::None(_) => ().into(),
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => ctx.start_batch(state.into()).into(),
        }
    }

    fn dispatch_kernel(
        &self,
        batcher: &mut Self::Batcher<'_>,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
    ) {
        match self {
            Self::None(ctx) => {
                let bindings = bindings.iter().copied().map(Into::into).collect::<Vec<_>>();

                ctx.dispatch_kernel(batcher.into(), kernel.into(), wg, &bindings)
            }
            #[cfg(feature = "wgsl")]
            Self::Wgsl(ctx) => {
                let bindings = bindings.iter().copied().map(Into::into).collect::<Vec<_>>();

                ctx.dispatch_kernel(batcher.into(), kernel.into(), wg, &bindings)
            }
        }
    }

    #[inline]
    fn schedule(
        &self,
        kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        bindings: &[&Self::Buffer],
        meta: &[u32],
    ) -> Result<Self::Schedule, Error> {
        Ok(match self {
            Self::None(ctx) => {
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
                let bindings = bindings.iter().copied().map(Into::into).collect::<Vec<_>>();

                ctx.schedule(kernels, &bindings, meta)?.into()
            }
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

                ctx.schedule(kernels, &bindings, meta)?.into()
            }
        })
    }
}

macro_rules! impl_backend {
    ($(#[$meta:meta])? $name:ident$(<$($lifetime:lifetime),*>)?, $nop:ty, $other:ident) => {
        #[allow(clippy::large_enum_variant)]
        $(#[$meta])?
        pub enum $name$(<$($lifetime),*>)? {
            None($nop, $(PhantomData<$(&$lifetime ()),*>)?),
            #[cfg(feature = "wgsl")]
            Wgsl(wgsl::$other$(<$($lifetime),*>)?),
        }

        impl$(<$($lifetime),*>)? From<$nop> for $name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: $nop) -> Self {
                Self::None(value, $({ $(let _: &$lifetime ();)* PhantomData })?)
            }
        }

        #[cfg(feature = "wgsl")]
        impl$(<$($lifetime),*>)? From<wgsl::$other$(<$($lifetime),*>)?> for $name$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: wgsl::$other$(<$($lifetime),*>)?) -> Self {
                Self::Wgsl(value)
            }
        }

        impl$(<$($lifetime),*>)? From<$name$(<$($lifetime),*>)?> for $nop {
            #[inline]
            fn from(value: $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::None(ctx, ..) => ctx,
                    _ => panic!("unsupported operation for nop backend"),
                }
            }
        }

        impl<'__a, $($($lifetime),*)?> From<&'__a $name$(<$($lifetime),*>)?> for &'__a $nop {
            #[inline]
            fn from(value: &'__a $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::None(ctx, ..) => ctx,
                    _ => panic!("unsupported operation for WGSL backend"),
                }
            }
        }

        impl<'__a, $($($lifetime),*)?> From<&'__a mut $name$(<$($lifetime),*>)?> for &'__a mut $nop {
            #[inline]
            fn from(value: &'__a mut $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::None(ctx, ..) => ctx,
                    _ => panic!("unsupported operation for WGSL backend"),
                }
            }
        }

        #[cfg(feature = "wgsl")]
        impl$(<$($lifetime),*>)? From<$name$(<$($lifetime),*>)?> for wgsl::$other$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::Wgsl(ctx) => ctx,
                    _ => panic!("unsupported operation for WGSL backend"),
                }
            }
        }

        #[cfg(feature = "wgsl")]
        impl<'__a, $($($lifetime),*)?> From<&'__a $name$(<$($lifetime),*>)?> for &'__a wgsl::$other$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: &'__a $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::Wgsl(ctx) => ctx,
                    _ => panic!("unsupported operation for WGSL backend"),
                }
            }
        }

        #[cfg(feature = "wgsl")]
        impl<'__a, $($($lifetime),*)?> From<&'__a mut $name$(<$($lifetime),*>)?> for &'__a mut wgsl::$other$(<$($lifetime),*>)? {
            #[inline]
            fn from(value: &'__a mut $name$(<$($lifetime),*>)?) -> Self {
                match value {
                    $name::Wgsl(ctx) => ctx,
                    _ => panic!("unsupported operation for WGSL backend"),
                }
            }
        }
    };
}

impl_backend!(
    #[derive(Debug)]
    DynBuffer,
    NopGpuBuffer,
    GpuBuffer
);

impl GpuBufferBackend for DynBuffer {
    impl_op! {
        size(&self,) -> u32
        size_bytes(&self,) -> u32
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
    #[derive(Clone)]
    DynKernel,
    NopGpuKernel,
    GpuKernel
);

impl GpuKernelBackend for DynKernel {
    impl_op! {
        iteration_space(&self,) -> &[fused_gpu::dispatch::backend::MetaId]
        block(&self,) -> &[u32; 3]
    }
}

impl_backend!(DynBatchState, (), CommandEncoder);
impl_backend!(DynSyncSubmissions, (), CommandBuffer);
impl_backend!(DynParamLayout, (), PipelineLayout);
impl_backend!(DynSubmissionIndex, (), SubmissionIndex);
impl_backend!(DynSchedule, (), Schedule);
impl_backend!(DynBatcher<'a>, (), ComputePass);
