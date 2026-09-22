//! Full dispatch of math operations and GPU contVec;

use crate::{
    dispatch::backend::{
        Graph, MetaId, NodeId, OptimState, Param, StateDim,
        kernel::{Dependencies, RawKernel, Redirect, SaveIndicator},
    },
    errors::Error,
    tensor::{Tensor, ToBuffer, build_dims},
};
use briny::{
    raw::cast::{slice_to_bytes, slice_to_bytes_mut},
    traits::Pod,
};
use core::fmt::Debug;
use half::slice::HalfFloatSliceExt;
use std::{vec, vec::Vec};

pub mod backend;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PollStatus {
    QueueEmpty,
    Failed,
    Pending,
    Ready,
}

#[derive(Hash, PartialEq, Eq)]
pub struct CompilationOptions {
    pub target: TargetCompilationOptions,
    pub debug: DebugCompilationOptions,
    pub opt: OptCompilationOptions,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct TargetCompilationOptions {
    pub flags: TargetFlags,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct TargetFlags: u8 {
        /// Support for linear algebra accelerator (e.g. tensor core via `wmma`)
        const LIN_ACC = 1 << 0;

        /// Links a BLAS library (e.g. `cuBLAS`)
        const BLAS_LIB = 1 << 1;

        /// Support for asynchronous memory load
        const ASYNC_MEM_LOAD = 1 << 2;

        /// Support for asynchronous memory store
        const ASYNC_MEM_STORE = 1 << 3;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct DebugCompilationOptions: u8 {
        /// Formats intermediate representation
        const PRETTY_PRINT_IR = 1 << 0;
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OptCompilationOptions {
    /// Sets tile size or block size.
    pub tile_size: u32,

    /// Sets the amount of passes to optimize each kernel.
    ///
    /// The compiler will quickly run out of optimizations if this is set too high.
    pub passes: u8,

    /// Defines the set of optimizations the compiler will run each pass.
    pub flags: OptFlags,
}

impl Default for OptCompilationOptions {
    fn default() -> Self {
        Self {
            tile_size: 16,
            passes: 3,
            flags: OptFlags::all(),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct OptFlags: u16 {
        /// Enables or disables dead code elimination.
        ///
        /// `let useless = 1234;` -> nothing
        const DEAD_CODE = 1 << 0;

        /// Enables or disables constant folding.
        ///
        /// `2*2` -> `4`
        const CONST_FOLD = 1 << 1;

        /// Converts assignments to op-assignments.
        ///
        /// `a = b + a;` -> `a += b;`
        const OP_ASSIGN = 1 << 2;

        /// Applies the identity prperty to arithmetic.
        ///
        /// `x*1` or `x/1` or `x+0` or `x-0` -> `x`
        const IDENTITY = 1 << 3;

        /// Converts constant division statements to multiplication.
        ///
        /// `x/2` -> `x*0.5`
        const DIV_CONST = 1 << 4;

        /// Fuses multiply-add operations.
        ///
        /// `a*b+c` -> `fma(a,b,c)`
        const MUL_ADD = 1 << 5;

        /// Moves operations that dont depend on the state of a loop to before the loop started.
        ///
        /// `loop { a = b; }` -> `a = b; loop {  }`
        const LOOP_STATELESS = 1 << 6;

        /// Removes empty scopes (loop, if) to simplify code.
        ///
        /// `loop {  }` -> nothing
        const EMPTY_SCOPE = 1 << 7;

        /// Changes mutable variables to immutable variables if they are never mutated.
        ///
        /// `let mut a = 123;` -> `let a = 123;`
        const UNUSED_MUT = 1 << 8;

        /// Removes copies of immutable variables, replacing their usages directly with the original variable.
        ///
        /// `let a = 123; let b = a; let c = b - 1;` -> `let a = 123; let c = a - 1;`
        const COPY_IMMUT = 1 << 9;

        /// Removes needless late intialization of mutable variables.
        ///
        /// `let mut a; a = 2;` -> `let mut a = 2;`
        const LATE_INIT = 1 << 10;
    }
}

pub trait GpuBufferBackend {
    fn size(&self) -> u32;

    fn size_bytes(&self) -> u32;
}

pub trait GpuKernelBackend {
    fn iteration_space(&self) -> &[MetaId];
    fn block(&self) -> &[u32; 3];
}

/// Core GPU backend trait, defining a context which uses the GPU
pub trait GpuBackend: Sized {
    type Buffer: GpuBufferBackend;
    type MetaBuf;
    type Kernel: GpuKernelBackend;
    type Schedule;

    /// The target-specific configuration for the compiler.
    fn target_spec(&self) -> TargetCompilationOptions;

    /// Allocates an uninitialized GPU buffer with the size `len` in bytes.
    fn alloc(&self, len: u32) -> Result<Self::Buffer, Error>;

    /// Allocates a slice of `u8` (to be reinterpreted as larger datatypes).
    fn alloc_init(&self, data: &[u8]) -> Result<Self::Buffer, Error>;

    /// Allocates a slice of `u32` on the GPU.
    ///
    /// This allocation is often small and uniform.
    fn alloc_meta(&self, data: &[u32]) -> Result<Self::MetaBuf, Error>;

    /// Copies the content of a CPU buffer to a GPU buffer at offset `dst_off`.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn upload(
        &self,
        buffer: &Self::Buffer,
        data: &[u8],
        src_off: u32,
        dst_off: u32,
    ) -> Result<(), Error>;

    /// Copies the content of one buffer to another.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn copy(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Error>;

    /// Copies the content of a GPU buffer to a CPU buffer.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error>;

    /// Compiles the `RawKernel` (containing ops+vars) to a GPU kernel ([`Self::Kernel`]).
    ///
    /// # Errors
    ///
    /// For an invalid kernel, param layout, or compilation options and all related errors.
    fn compile(
        &self,
        src: &RawKernel,
        params: &[Param],
        options: &CompilationOptions,
    ) -> Result<Self::Kernel, Error>;

    fn schedule(
        &self,
        kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        bindings: &[&Self::Buffer],
        meta: &[u32],
        meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule, Error>;

    fn dispatch_kernel(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
        meta: &Self::MetaBuf,
    ) -> Result<(), Error>;

    fn dispatch_schedule(&self, schedule: &Self::Schedule) -> Result<(), Error>;

    fn sync(&self) -> Result<(), Error>;

    fn is_ready(&self) -> Result<bool, Error>;
}

/// Allocate a buffer on the GPU.
pub fn gpu_alloc<B: GpuBackend>(context: &B, len: u32) -> Result<GpuBuffer<B>, Error> {
    Ok(GpuBuffer {
        inner: context.alloc(len)?,
    })
}

/// Allocate a buffer on the GPU and copy CPU memory into it.
pub fn gpu_alloc_init<T: Pod, B: GpuBackend>(
    context: &B,
    data: &[T],
) -> Result<GpuBuffer<B>, Error> {
    Ok(GpuBuffer {
        inner: context.alloc_init(slice_to_bytes(data))?,
    })
}

/// Generic GPU buffer for any backend.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct GpuBuffer<B: GpuBackend = backend::GpuContext> {
    pub(crate) inner: B::Buffer,
}

impl<B: GpuBackend> GpuBuffer<B> {
    /// Requests the length of the buffer in 32-bit chunks (`f32`, `u32`, etc.).
    pub fn size(&self) -> u32 {
        self.inner.size()
    }

    /// Requests the length of the buffer in bytes.
    pub fn size_bytes(&self) -> u32 {
        self.inner.size_bytes()
    }
}

/// A group of compiled kernels including all required metadata.
///
/// It includes the kernels of the forward, backward, and loss passes, the order
/// in which to execute them, and the parameters they use.
#[derive(Debug)]
pub struct KernelGroup<'a, B: GpuBackend = backend::GpuContext> {
    pub(crate) forward: Vec<Dependencies<Redirect<(B::Kernel, usize, &'a [bool])>>>,
    pub(crate) backward: Vec<Dependencies<Redirect<(B::Kernel, usize, &'a [bool])>>>,
    pub(crate) loss: B::Kernel,
    pub(crate) optim: B::Kernel,
}

/// Schedule used to improve performance by caching critical launch information.
pub struct Schedule<'a, B: GpuBackend = backend::GpuContext> {
    forward: B::Schedule,
    backward: B::Schedule,
    loss: LossSchedule<'a, B>,
    optim: OptimSchedule<'a, B>,
}

struct LossSchedule<'a, B: GpuBackend> {
    grid: [u32; 3],
    meta: &'a B::MetaBuf,
    bindings: [&'a B::Buffer; 4],
    kernel: B::Kernel,
}

struct OptimSchedule<'a, B: GpuBackend> {
    meta: &'a B::MetaBuf,
    bindings: Vec<&'a B::Buffer>,
    kernel: B::Kernel,
    state: &'a [B::Buffer],
}

/// Generic GPU context storing a handle to the device and shaders.
///
/// Perhaps most important structure in all of Fused GPU this is. Without it,
/// you couldn't compile kernels, execute kernels, create tensors, copy data.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContext<B: GpuBackend = backend::GpuContext> {
    pub(crate) inner: B,
}

impl GpuContext<backend::GpuContext> {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            inner: backend::GpuContext::new()?,
        })
    }
}

impl<B: GpuBackend> GpuContext<B> {
    /// Wrapping the inner context.
    pub const fn new_with_context(ctx: B) -> Self {
        Self { inner: ctx }
    }

    /// Detects the available target configuration for this device/context.
    pub fn detect_target(&self) -> TargetCompilationOptions {
        self.inner.target_spec()
    }

    /// Compiles a graph into a kernels.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent but often a result of invalid input.
    /// Regardless, errors must be handled properly in critical code.
    pub fn compile<'a>(
        &self,
        ir: &'a backend::kernel::KernelGroup<'a>,
        options: &CompilationOptions,
    ) -> Result<KernelGroup<'a, B>, Error> {
        let forward = ir
            .forward
            .kernels
            .iter()
            .map(|kernel| {
                let dep = kernel.dep.clone();

                let compiled = match &kernel.val {
                    Redirect::Unmasked(kernel) => {
                        let params = ir
                            .forward
                            .params
                            .iter()
                            .enumerate()
                            .filter_map(
                                |(i, param)| if kernel.params[i] { Some(*param) } else { None },
                            )
                            .collect::<Vec<_>>();

                        Redirect::Unmasked((
                            self.inner.compile(&kernel.raw, &params, options)?,
                            kernel.raw.root,
                            kernel.params.as_slice(),
                        ))
                    }
                    Redirect::Redirected(idx) => Redirect::Redirected(*idx),
                };

                Ok(Dependencies { val: compiled, dep })
            })
            .collect::<Result<
                Vec<Dependencies<Redirect<(<B as GpuBackend>::Kernel, usize, &'a [bool])>>>,
                Error,
            >>()?;

        let backward = ir
            .backward
            .kernels
            .iter()
            .map(|kernel| {
                let dep = kernel.dep.clone();

                let compiled = match &kernel.val {
                    Redirect::Unmasked(kernel) => {
                        let params = ir
                            .backward
                            .params
                            .iter()
                            .enumerate()
                            .filter_map(
                                |(i, param)| if kernel.params[i] { Some(*param) } else { None },
                            )
                            .collect::<Vec<_>>();

                        Redirect::Unmasked((
                            self.inner.compile(&kernel.raw, &params, options)?,
                            kernel.raw.root,
                            kernel.params.as_slice(),
                        ))
                    }
                    Redirect::Redirected(idx) => Redirect::Redirected(*idx),
                };

                Ok(Dependencies { val: compiled, dep })
            })
            .collect::<Result<
                Vec<Dependencies<Redirect<(<B as GpuBackend>::Kernel, usize, &'a [bool])>>>,
                Error,
            >>()?;

        let loss = self.inner.compile(&ir.loss.raw, &ir.loss.params, options)?;
        let optim = self
            .inner
            .compile(&ir.optim.raw, &ir.optim.params, options)?;

        Ok(KernelGroup {
            forward,
            backward,
            loss,
            optim,
        })
    }

    /// Copies the content of a GPU buffer into a CPU buffer.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn download<T: Pod, S: ToBuffer<B>>(&self, tensor: &S, dst: &mut [T]) -> Result<(), Error> {
        self.inner
            .download(tensor.as_buffer(), slice_to_bytes_mut(dst))
    }

    /// Copies the content of a CPU buffer into GPU buffer.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn upload<T: Pod, S: ToBuffer<B>>(
        &self,
        tensor: &S,
        dst: &[T],
        src_off: u32,
        dst_off: u32,
    ) -> Result<(), Error> {
        self.inner
            .upload(tensor.as_buffer(), slice_to_bytes(dst), src_off, dst_off)
    }

    /// Copies the content of one buffer to another without mutating the source.
    ///
    /// This is faster than chaining [`Self::download`] into [`Self::upload`] because it
    /// bypasses CPU memory.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn copy<S1: ToBuffer<B>, S2: ToBuffer<B>>(&self, src: &S1, dst: &S2) -> Result<(), Error> {
        self.inner.copy(src.as_buffer(), dst.as_buffer())
    }

    pub fn sync(&self) -> Result<(), Error> {
        self.inner.sync()
    }

    pub fn alloc_tensors(
        &self,
        graph: &Graph<'_>,
        saved: &[SaveIndicator],
        meta: &[u32],
        state: &OptimState,
    ) -> Result<AllocTensors<B>, Error> {
        let mut forward_saved = Vec::new();
        let mut grad_tensors = Vec::new();
        let mut state_tensors = Vec::new();

        let total_grads = state.shapes.len();

        for (idx, save) in saved.iter().enumerate() {
            let node = &graph.nodes[idx];
            let node_shape = &node.shape;
            let num_shape = build_dims(node_shape, meta);

            let len = num_shape.iter().product::<u32>() * node.dtype.size() as u32;

            if save.is_defined_in_forward() {
                let buf = self.inner.alloc(len)?;
                forward_saved.push(buf);
            }

            if save.is_defined_in_backward() {
                let buf = self.inner.alloc(len)?;
                grad_tensors.push(buf);

                state_tensors.reserve(total_grads);

                for state_t in 0..total_grads {
                    let len = match state.shapes[state_t] {
                        StateDim::Const(value) => value,
                        StateDim::GradRelative(value) => len * value,
                    };
                    let buf = self.inner.alloc(len)?;

                    state_tensors.push(buf);
                }
            }
        }

        let meta_buffer = self.inner.alloc_meta(meta)?;

        let node_shape = &graph.nodes[graph.nodes.len() - 1].shape;
        let node_dtype = &graph.nodes[graph.nodes.len() - 1].dtype;
        let shape = build_dims(node_shape, meta);
        let len = shape.iter().product::<u32>() * node_dtype.size() as u32;

        let forward_out = self.inner.alloc(len)?;
        let loss_t = self.inner.alloc(len)?;
        let seed = self.inner.alloc(len)?;

        Ok(AllocTensors {
            meta: meta_buffer,
            forward_saved,
            grad_tensors,
            forward_out,
            seed,
            state: state_tensors,
            loss_t: Tensor {
                shape,
                data: GpuBuffer { inner: loss_t },
            },
        })
    }

    pub fn new_tensor<const N: u32>(&self, shape: Vec<u32>) -> Result<Tensor<B>, Error> {
        let len = shape.iter().product::<u32>() * const { N / 8 };
        let data = gpu_alloc(&self.inner, len)?;
        Ok(Tensor { shape, data })
    }

    pub fn init_tensor_f64(&self, shape: Vec<u32>, data: &[f64]) -> Result<Tensor<B>, Error> {
        debug_assert_eq!(
            shape.iter().product::<u32>(),
            data.len() as u32,
            "shape product (left) and data length (right) mismatch"
        );

        let data = gpu_alloc_init(&self.inner, data)?;
        Ok(Tensor { shape, data })
    }

    pub fn init_tensor_f32(&self, shape: Vec<u32>, data: &[f32]) -> Result<Tensor<B>, Error> {
        debug_assert_eq!(
            shape.iter().product::<u32>(),
            data.len() as u32,
            "shape product (left) and data length (right) mismatch"
        );

        let data = gpu_alloc_init(&self.inner, data)?;
        Ok(Tensor { shape, data })
    }

    pub fn init_tensor_f16(&self, shape: Vec<u32>, data: &[half::f16]) -> Result<Tensor<B>, Error> {
        debug_assert_eq!(
            shape.iter().product::<u32>(),
            data.len() as u32,
            "shape product (left) and data length (right) mismatch"
        );

        let data_u16 = data.reinterpret_cast();
        let data = gpu_alloc_init(&self.inner, data_u16)?;

        Ok(Tensor { shape, data })
    }

    pub fn init_tensor_bf16(
        &self,
        shape: Vec<u32>,
        data: &[half::bf16],
    ) -> Result<Tensor<B>, Error> {
        debug_assert_eq!(
            shape.iter().product::<u32>(),
            data.len() as u32,
            "shape product (left) and data length (right) mismatch"
        );

        let data_u16 = data.reinterpret_cast();
        let data = gpu_alloc_init(&self.inner, data_u16)?;

        Ok(Tensor { shape, data })
    }

    /// Allocates an empty one-hot vector.
    pub fn new_onehot(&self, classes: u32) -> Result<Tensor<B>, Error> {
        let data = gpu_alloc(&self.inner, classes)?;
        Ok(Tensor {
            data,
            shape: vec![classes],
        })
    }

    /// Allocates a new one-hot vector with the defined classes.
    pub fn new_onehot_init(&self, indices: &[u32]) -> Result<Tensor<B>, Error> {
        let data = gpu_alloc_init(&self.inner, indices)?;
        Ok(Tensor {
            data,
            shape: vec![indices.len() as u32],
        })
    }

    pub fn schedule<'a>(
        &self,
        kernels: KernelGroup<B>,
        meta: &[u32],
        in_tensors: &'a [Tensor<B>],
        alloc_tensors: &'a AllocTensors<B>,
        state: &'a [B::Buffer],
    ) -> Result<Schedule<'a, B>, Error>
    where
        B::Buffer: Sized,
    {
        let mut bindings = Vec::new();

        alloc_tensors
            .forward_saved
            .iter()
            .for_each(|t| bindings.push(t));
        for t in in_tensors {
            bindings.push(&t.data.inner);
        }

        bindings.push(&alloc_tensors.forward_out);

        let forward = self
            .inner
            .schedule(kernels.forward, &bindings, meta, &alloc_tensors.meta)?;

        bindings.clear();

        bindings.push(&alloc_tensors.seed);

        alloc_tensors
            .grad_tensors
            .iter()
            .for_each(|t| bindings.push(t));
        for t in in_tensors {
            bindings.push(&t.data.inner);
        }
        alloc_tensors
            .forward_saved
            .iter()
            .for_each(|t| bindings.push(t));

        let backward =
            self.inner
                .schedule(kernels.backward, &bindings, meta, &alloc_tensors.meta)?;

        let grid = alloc_tensors.loss_t.calc_grid(*kernels.loss.block());

        let bindings = [
            &alloc_tensors.loss_t.data.inner,
            &alloc_tensors.seed,
            &alloc_tensors.forward_out,
            &alloc_tensors.seed,
        ];

        let meta = &alloc_tensors.meta;

        let loss = LossSchedule {
            grid,
            meta,
            bindings,
            kernel: kernels.loss,
        };

        let mut bindings = vec![&alloc_tensors.seed, &alloc_tensors.seed];

        for _ in 0..state.len() {
            bindings.push(&alloc_tensors.seed);
        }

        let optim = OptimSchedule {
            meta,
            bindings,
            kernel: kernels.optim,
            state,
        };

        Ok(Schedule {
            forward,
            backward,
            loss,
            optim,
        })
    }

    pub fn dispatch_forward(&self, schedule: &Schedule<'_, B>) -> Result<(), Error> {
        self.inner.dispatch_schedule(&schedule.forward)
    }

    pub fn dispatch_backward(&self, schedule: &Schedule<'_, B>) -> Result<(), Error> {
        self.inner.dispatch_schedule(&schedule.backward)
    }

    pub fn dispatch_loss<T: ToBuffer<B>>(
        &self,
        schedule: &mut Schedule<'_, B>,
        target: &T,
    ) -> Result<(), Error> {
        let mut bindings = schedule.loss.bindings;
        bindings[3] = target.as_buffer();

        self.inner.dispatch_kernel(
            &schedule.loss.kernel,
            schedule.loss.grid,
            &bindings,
            schedule.loss.meta,
        )
    }

    pub fn dispatch_optim(
        &self,
        schedule: &mut Schedule<'_, B>,
        weight: &Tensor<B>,
        grad: usize,
        tensors: &AllocTensors<B>,
    ) -> Result<(), Error> {
        let grid = weight.calc_grid(*schedule.optim.kernel.block());

        let mut bindings = schedule.optim.bindings.clone();

        bindings[0] = weight.as_buffer();
        bindings[1] = &tensors.grad_tensors[grad];

        for state_t in 0..(bindings.len() - 2) {
            bindings[2 + state_t] = &schedule.optim.state[state_t];
        }

        self.inner
            .dispatch_kernel(&schedule.optim.kernel, grid, &bindings, schedule.optim.meta)
    }

    #[cfg(feature = "io")]
    pub fn save_tensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        tensors: &[Tensor<B>],
        header: crate::io::BpatHeader,
    ) -> Result<(), Error<crate::io::SerialTensorError>> {
        crate::io::save_tensors(path, self, tensors, header)
    }

    #[cfg(feature = "io")]
    pub fn load_tensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<Vec<Tensor<B>>, Error<crate::io::SerialTensorError>> {
        crate::io::load_tensors(path, self)
    }
}

pub struct AllocTensors<B: GpuBackend = backend::GpuContext> {
    pub meta: B::MetaBuf,
    pub forward_saved: Vec<B::Buffer>,
    pub grad_tensors: Vec<B::Buffer>,
    pub forward_out: B::Buffer,
    pub seed: B::Buffer,

    pub state: Vec<B::Buffer>,

    pub loss_t: Tensor<B>,
}
