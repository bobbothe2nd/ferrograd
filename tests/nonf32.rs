use fused_gpu::{
    dispatch::{
        CompilationOptions, GpuContext,
        backend::{DType, Graph, LossType, Metadata, OptimState, OptimType},
    },
    tensor::f16,
};

#[test]
fn mul_add_f16() {
    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::MEAN_SQUARED_ERROR,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );
    let a = graph.input(&[m, n], DType::F16);
    let b = graph.input(&[m, n], DType::F16);
    let c = graph.input(&[m, n], DType::F16);

    let x = graph.mul(a, b);
    graph.add(c, x);

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let ctx = GpuContext::new().unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f16([32, 32].to_vec(), &[f16::from_f32(3.0); 1024]),
        ctx.init_tensor_f16([32, 32].to_vec(), &[f16::from_f32(2.0); 1024]),
        ctx.init_tensor_f16([32, 32].to_vec(), &[f16::from_f32(1.0); 1024]),
    ];

    let meta_binding = [briny::raw::cast::reinterpret(1e-3_f32), 32, 32];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx
        .upload(&saved_tensors.seed, &[f16::from_f32(1.0); 1024])
        .unwrap();

    let schedule = ctx
        .schedule(
            &kernels,
            &meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &state,
        )
        .unwrap();

    let mut state = ctx.prepare_batch();

    {
        let mut pass = ctx.start_batch(&mut state);

        pass.dispatch_forward(&schedule);
        pass.dispatch_backward(&schedule);
    }

    upload.sync();

    state.encode().submit().sync();

    let mut dst = [f16::from_f32(0_f32); 1024];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(7.0)));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(2.0)));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(3.0)));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(1.0)));
}
