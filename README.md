# `fused_gpu`

Advanced graph-based GPU compiler for linear algebra and AI/ML/DL.

## Usage

Very basic usage of this crate is shown below:

```rust
let mut meta = Metadata::new();
let m = meta.new_field();
let n = meta.new_field();
let k = meta.new_field();

let mut graph = Graph::new(LossType::CROSS_ENTROPY);
let a = graph.input(&[m, k]);
let b = graph.input(&[k, n]);
let c = graph.input(&[m, n]);

let x = graph.matmul(a, b);
let s = graph.sub(c, x);
graph.softmax(s);

let saved = graph.compute_saved_nodes();
let options = CompilationOptions::default();

let ctx = GpuContext::new().unwrap();
graph.validate(meta).unwrap();
graph.topo_sort().unwrap();
graph.rebuild_outputs();
let ir = graph.lower(meta, &options, &saved).unwrap();
let kernels = ctx.compile(&ir, &options).unwrap();

let in_tensors = [
    ctx.new_tensor_init(&[16, 32], &[3.0; 512]),
    ctx.new_tensor_init(&[32, 64], &[2.0; 2048]),
    ctx.new_tensor_init(&[16, 64], &[1.0; 1024]),
];

let meta_binding = [16, 64, 32];
assert!(meta.validate_meta(&meta_binding));

let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding);

let upload = ctx.upload(&saved_tensors.seed, &[1.0_f32; 1024]).unwrap();

let schedule = ctx
    .schedule(&kernels, &meta_binding, &in_tensors, &saved_tensors)
    .unwrap();

let mut state = ctx.prepare_batch();

{
    let mut pass = ctx.start_batch(&mut state);

    pass.dispatch_forward(&schedule);
    pass.dispatch_backward(&schedule);
}

upload.sync();

state.encode().submit().sync();

let mut dst = [0_f32; 2048];

let out_tensor = &saved_tensors.forward_out;
let grad_tensors = &saved_tensors.grad_tensors;

ctx.download(&out_tensor, &mut dst).unwrap();
let download = &dst[..1024];
std::eprintln!("{:?}", &download[..64]);
assert!(download.iter().all(|x| *x == 1.0 / 64.0));

ctx.download(&grad_tensors[0], &mut dst).unwrap();
std::eprintln!("{:?}", &dst[..64]);
let download = &dst[..512];
assert!(download.iter().all(|x| *x == 0.0));

ctx.download(&grad_tensors[1], &mut dst).unwrap();
assert!(dst.iter().all(|x| *x == 0.0));

ctx.download(&grad_tensors[2], &mut dst).unwrap();
let download = &dst[..1024];
assert!(download.iter().all(|x| *x == 0.0));
```

Also supports `f16` and `bf16` on most backends. These are re-exported from `half` in the `tensors` module.

## Project Status

Basic tested operations include:

- `sub`
- `matmul`
- `add`
- `mul`
- `softmax`

Other unary/binary operations are likely trivially correct but not extensively tested.

Stateless optimizers (like SGD) are coming soon. Other optimizers (like Adam) may be added later.

## Backends

Only supports WGPU/WGSL backends. CUDA and ROCm backends planned before `v1.0.0`. Other backends (e.g. CPU) are unlikely. Custom backends are fully supported.
