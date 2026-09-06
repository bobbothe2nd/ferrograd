# `fused_gpu`

Advanced graph-based GPU compiler for linear algebra and AI/ML/DL.

## Usage

Very basic usage of this crate is shown below:

```rust
use fused_gpu::dispatch::{
    CompilationOptions, GpuContext,
    backend::{DType, Graph, LossType, Metadata, OptimState, OptimType},
};

// used for shapes in graph, must be multiples of 16 right now
let mut meta = Metadata::new();
let m = meta.new_field();
let n = meta.new_field();

// define the optimizer state, should match OptimType
let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

// define the graph for automatic kernel fusion
let mut graph = Graph::new(
    LossType::MEAN_SQUARED_ERROR,
    OptimType::STOCHASTIC_GRADIENT_DESCENT,
);

// define inputs
let a = graph.input(&[m, n], DType::F32);
let b = graph.input(&[m, n], DType::F32);
let c = graph.input(&[m, n], DType::F32);

// define operations here (e.g. ferrograd-nn)

// compute saved nodes ahead of time for compilation
let saved = graph.compute_saved_nodes();

// validate and sort graph
graph.validate(meta).unwrap();
graph.topo_sort().unwrap();

// must rebuild outputs after sorting
graph.rebuild_outputs();

// default compilation options are usually fine
let options = CompilationOptions::default();

// create GpuContext, the handle to a GPU device that compiles and launches kernels
let ctx = GpuContext::new().unwrap();
let ir = graph.lower(meta, &options, &saved).unwrap();
let kernels = ctx.compile(&ir, &options).unwrap();

// allocate inputs with the same shapes defined by graph and metadata
let in_tensors = [
    ctx.init_tensor_f32(&[32, 32], &[3.0; 1024]),
    ctx.init_tensor_f32(&[32, 32], &[2.0; 1024]),
    ctx.init_tensor_f32(&[32, 32], &[1.0; 1024]),
];

// includes learning rate and shape metadata defined by graph
let meta_binding = [1e-3_f32.to_bits(), 32, 32];
assert!(meta.validate_meta(&meta_binding));

// allocate a lot more tensors
let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

// because this is easier than defining a target and dispatch_loss
let upload = ctx.upload(&saved_tensors.seed, &[1_f32; 1024]).unwrap();

// scheduling is precomputing everything that doesn't need to be recomputed for every dispatch
let mut schedule = ctx
    .schedule(
        &kernels,
        &meta_binding,
        &in_tensors,
        &saved_tensors,
        &[],
        &state,
    )
    .unwrap();

// dispatch everything
let mut state = ctx.prepare_batch();

{
    let mut pass = ctx.start_batch(&mut state);

    pass.dispatch_forward(&schedule);
    pass.dispatch_backward(&schedule);

    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[0], 0, &saved_tensors);
    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[1], 1, &saved_tensors);
    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[2], 2, &saved_tensors);
}

// synchronize previous seed upload before launching batch
upload.sync();

// launch and synchronize batch output
state.encode().submit().sync();

let mut dst = [0_f32; 1024];

let grad_tensors = &saved_tensors.grad_tensors;

ctx.download(&saved_tensors.forward_out, &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 7.0));

ctx.download(&grad_tensors[0], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 0.0));

ctx.download(&grad_tensors[1], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 0.0));

ctx.download(&grad_tensors[2], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 0.0));

ctx.download(&in_tensors[0], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 3.0 - 2e-3));

ctx.download(&in_tensors[1], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 2.0 - 3e-3));

ctx.download(&in_tensors[2], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 1.0 - 1e-3));

// intentionally incorrect target (4 != 7) to produce meaningful gradients
let target = ctx.init_tensor_f32(&[32, 32], &[4.0; 1024]);

// or dispatch with loss
let mut state = ctx.prepare_batch();

{
    let mut pass = ctx.start_batch(&mut state);

    pass.dispatch_forward(&schedule);
    pass.dispatch_loss(&schedule, &target);
    pass.dispatch_backward(&schedule);

    // all grads are zero including seed
    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[0], 0, &saved_tensors);
    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[1], 1, &saved_tensors);
    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[2], 2, &saved_tensors);
}

// launch and synchronize batch output
state.encode().submit().sync();
```

Also supports `f16` and `bf16` on most backends. These are re-exported from `half` in the `tensors` module.

## Operations

Basic tested operations include:

- `sub`
- `matmul`
- `add`
- `mul`
- `softmax`

Other unary/binary operations are likely trivially correct but not extensively tested.

SGD is supported and tested.

## Backends

Only supports WGPU/WGSL backends. CUDA and ROCm backends planned before `v1.0.0`. Other backends (e.g. CPU) are unlikely. Custom backends are fully supported.
