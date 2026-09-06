use ferrograd::{
    dispatch::{
        CompilationOptions, DType, DebugCompilationOptions, GpuContext, Graph, Metadata,
        OptCompilationOptions,
    },
    nn::{MEAN_SQUARED_ERROR, Optim},
};

#[test]
fn mul_add_forward_backward() {
    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = Optim::STOCHASTIC_GRADIENT_DESCENT.state;

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, Optim::STOCHASTIC_GRADIENT_DESCENT.lower);

    {
        let mut graph = graph.define_ops();

        let a = graph.input(&[m, n], DType::F32);
        let b = graph.input(&[m, n], DType::F32);
        let c = graph.input(&[m, n], DType::F32);

        let x = graph.mul(a, b);
        graph.add(c, x);
    }

    let saved = graph.compute_saved_nodes();
    graph.validate(meta).unwrap();

    let ctx = GpuContext::new().unwrap();
    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    let meta_binding = [briny::raw::cast::reinterpret(1e-3_f32), 32, 32];
    assert!(meta.validate_meta(&meta_binding));
    let meta_binding = ctx.alloc_meta(&meta_binding);

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, meta_binding, &state);

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f32([32, 32].to_vec(), &[3.0; 1024]),
        ctx.init_tensor_f32([32, 32].to_vec(), &[2.0; 1024]),
        ctx.init_tensor_f32([32, 32].to_vec(), &[1.0; 1024]),
    ];

    let upload = ctx.upload(&saved_tensors.seed, &[1_f32; 1024], 0).unwrap();

    let mut schedule = ctx
        .schedule(
            kernels,
            meta_binding,
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
