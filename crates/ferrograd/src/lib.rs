mod backend;
mod graph;
mod id;
mod optim;

pub use fused_gpu::tensor;

pub mod dispatch {
    pub use fused_gpu::dispatch::{
        AllocTensors, BatchState, Batcher, CompilationOptions, DebugCompilationOptions, GpuBackend,
        GpuBuffer, GpuBufferBackend, GpuKernelBackend, OptCompilationOptions, OptFlags, PollStatus,
        SubmissionIndex, SyncSubmission, TargetCompilationOptions,
    };

    pub use fused_gpu::dispatch::backend::{
        Axis, DType, DispatchOptions, GraphOp, LossType, Metadata, Node, Op, OptimState, OptimType,
        Param, ParamTy, SharedAlloc, StateDim, Value, ValueState,
    };

    pub use fused_gpu::dispatch::backend::kernel::{
        Dependencies, Kernel, KernelGroup, KernelsChained, KernelsRedirected, LinkedKernel,
        NodeInput, RawKernel, Redirect, SaveIndicator,
    };

    pub use fused_gpu::dispatch::KernelGroup as GpuKernelGroup;

    pub use fused_gpu::errors::GraphErrorContext;

    pub use crate::{
        backend::{
            DynBatchState, DynBatcher, DynBuffer, DynKernel, DynParamLayout, DynSchedule,
            DynSubmissionIndex, DynSyncSubmissions, Dynamic, GpuContext, MetaBinding, SavedNodes,
            Schedule,
        },
        graph::{DefineOps, Graph},
        id::{EdgeId, MetaId, NodeId, ParamId, SharedId, ValueId},
    };
}

pub mod errors {
    pub use fused_gpu::errors::{Error, ErrorKind};
}

pub mod io {
    pub use fused_gpu::io::versions::v0_v2::{
        load_into_tensors_v0, load_into_tensors_v2_bf16, load_into_tensors_v2_f16,
        load_into_tensors_v2_f32, load_tensors_v0, load_tensors_v2_bf16, load_tensors_v2_f16,
        load_tensors_v2_f32, save_tensors_v0, save_tensors_v2_bf16, save_tensors_v2_f16,
        save_tensors_v2_f32,
    };

    pub use fused_gpu::io::versions::v1::{
        load_into_tensors_v1, load_into_tensors_v1m, load_tensors_v1, load_tensors_v1m,
        save_tensors_v1, save_tensors_v1m,
    };

    pub use fused_gpu::io::headers::{
        BPAT_MAGIC_V0, BPAT_MAGIC_V1, BPAT_MAGIC_V1_MICRO, BPAT_MAGIC_V2_BF16, BPAT_MAGIC_V2_F16,
        BPAT_MAGIC_V2_F32,
    };

    pub use fused_gpu::io::{BpatHeader, SerialTensorError, load_tensors, save_tensors};
}

pub mod nn {
    pub use crate::optim::Optim;
    pub use ferrograd_nn::loss::*;
    pub use ferrograd_nn::op::*;
}

// #[allow(dead_code)]
// #[doc = include_str!("../README.md")]
// fn test() {}
