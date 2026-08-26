use fused_gpu::dispatch::{
    CompilationOptions, GpuContext,
    backend::{DType, Graph, LossType, Metadata, OptimState, OptimType},
};

#[test]
fn mul_add_forward_backward() {
    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::MEAN_SQUARED_ERROR,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );
    let a = graph.input(&[m, n], DType::F32);
    let b = graph.input(&[m, n], DType::F32);
    let c = graph.input(&[m, n], DType::F32);

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
        ctx.init_tensor_f32(&[32, 32], &[3.0; 1024]),
        ctx.init_tensor_f32(&[32, 32], &[2.0; 1024]),
        ctx.init_tensor_f32(&[32, 32], &[1.0; 1024]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), 32, 32];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx.upload(&saved_tensors.seed, &[1_f32; 1024]).unwrap();

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

    let mut state = ctx.prepare_batch();

    {
        let mut pass = ctx.start_batch(&mut state);

        pass.dispatch_forward(&schedule);
        pass.dispatch_backward(&schedule);

        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[0], 0, &saved_tensors);
        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[1], 1, &saved_tensors);
        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[2], 2, &saved_tensors);
    }

    upload.sync();

    state.encode().submit().sync();

    let mut dst = [0_f32; 1024];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
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
}
