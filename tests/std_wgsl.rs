extern crate alloc;

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
        ctx.init_tensor_f32([32, 32].to_vec(), &[3.0; 1024]),
        ctx.init_tensor_f32([32, 32].to_vec(), &[2.0; 1024]),
        ctx.init_tensor_f32([32, 32].to_vec(), &[1.0; 1024]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), 32, 32];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx.upload(&saved_tensors.seed, &[1_f32; 1024], 0).unwrap();

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

    let mut dst = [0_f32; 1024];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 7.0));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 2.0));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 3.0));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 1.0));
}

#[test]
fn matmul_sub_softmax_forward_backward() {
    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::CROSS_ENTROPY,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );
    let a = graph.input(&[m, k], DType::F32);
    let b = graph.input(&[k, n], DType::F32);
    let c = graph.input(&[m, n], DType::F32);

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
        ctx.init_tensor_f32([16, 32].to_vec(), &[3.0; 512]),
        ctx.init_tensor_f32([32, 64].to_vec(), &[2.0; 2048]),
        ctx.init_tensor_f32([16, 64].to_vec(), &[1.0; 1024]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), 16, 64, 32];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx
        .upload(&saved_tensors.seed, &[1.0_f32; 1024], 0)
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

    let mut dst = [0_f32; 2048];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
    std::eprintln!("{:?}", &dst[..64]);
    let download = &dst[..1024];
    assert!(download.iter().all(|x| *x == 1.0 / 64.0));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    let download = &dst[..512];
    assert!(download.iter().all(|x| *x == 0.0));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 0.0));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    let download = &dst[..1024];
    assert!(download.iter().all(|x| *x == 0.0));
}

#[test]
fn div_const_softmax_forward_backward() {
    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::CROSS_ENTROPY,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );
    let x = graph.input(&[m, n], DType::F32);

    let two = graph.constant_f32(2.0);
    let logits = graph.div(two, x);
    let soft = graph.softmax(logits);
    graph.add(soft, two);

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let ctx = GpuContext::new().unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [ctx.init_tensor_f32([16, 32].to_vec(), &[3.0; 512])];

    let meta_binding = [1e-3_f32.to_bits(), 16, 32];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx.upload(&saved_tensors.seed, &[1_f32; 512], 0).unwrap();

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

    let mut dst = [0_f32; 512];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 2.0 + (1.0 / 32.0)));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    assert!(dst.iter().all(|x| *x == 0.0));
}

#[test]
fn matmul_add_forward_backward() {
    const M: u32 = 32;
    const N: u32 = 64;
    const K: u32 = 16;

    const A_VAL: f32 = 3.0;
    const B_VAL: f32 = 2.0;
    const C_VAL: f32 = 1.0;

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::MEAN_SQUARED_ERROR,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );

    {
        let a = graph.input(&[m, k], DType::F32);
        let b = graph.input(&[k, n], DType::F32);
        let c = graph.input(&[m, n], DType::F32);

        let x = graph.matmul(a, b);
        graph.add(x, c);
    }

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let ctx = GpuContext::new().unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f32([M, K].to_vec(), &[A_VAL; (M * K) as usize]),
        ctx.init_tensor_f32([K, N].to_vec(), &[B_VAL; (K * N) as usize]),
        ctx.init_tensor_f32([M, N].to_vec(), &[C_VAL; (M * N) as usize]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), M, N, K];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx
        .upload(&saved_tensors.seed, &[1_f32; (M * N) as usize], 0)
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

    let mut dst = alloc::vec![0_f32; (M * N).max(M * K).max(K * N) as usize];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(
        download
            .iter()
            .all(|x| *x == (A_VAL * B_VAL * K as f32) + C_VAL)
    );

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    let download = &dst[..(M * K) as usize];
    assert!(download.iter().all(|x| *x == B_VAL * N as f32));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    let download = &dst[..(K * N) as usize];
    assert!(download.iter().all(|x| *x == A_VAL * M as f32));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(download.iter().all(|x| *x == 1.0));
}

#[test]
fn matmul_chain3_forward_backward() {
    const M: u32 = 32;
    const N: u32 = 64;
    const K: u32 = 128;
    const H: u32 = 256;

    const A_VAL: f32 = 3.0;
    const B_VAL: f32 = 2.0;
    const C_VAL: f32 = 1.0;
    const D_VAL: f32 = 0.5;
    const E_VAL: f32 = 1.0;
    const X_VAL: f32 = A_VAL * B_VAL * K as f32;
    const Y_VAL: f32 = C_VAL * X_VAL * M as f32;
    const Z_VAL: f32 = Y_VAL * D_VAL * N as f32;
    const Y_GRAD: f32 = D_VAL * H as f32;
    const X_GRAD: f32 = Y_GRAD * C_VAL * H as f32;
    const D_GRAD: f32 = Y_VAL * H as f32;
    const C_GRAD: f32 = Y_GRAD * X_VAL * N as f32;
    const B_GRAD: f32 = A_VAL * X_GRAD * M as f32;
    const A_GRAD: f32 = X_GRAD * B_VAL * N as f32;

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();
    let h = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::MEAN_SQUARED_ERROR,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );

    let a = graph.input(&[m, k], DType::F32);
    let b = graph.input(&[k, n], DType::F32);
    let c = graph.input(&[h, m], DType::F32);
    let d = graph.input(&[n, h], DType::F32);
    let e = graph.input(&[h, h], DType::F32);

    let x = graph.matmul(a, b);
    let y = graph.matmul(c, x);
    let z = graph.matmul(y, d);
    graph.add(z, e);

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let ctx = GpuContext::new().unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f32([M, K].to_vec(), &[A_VAL; (M * K) as usize]),
        ctx.init_tensor_f32([K, N].to_vec(), &[B_VAL; (K * N) as usize]),
        ctx.init_tensor_f32([H, M].to_vec(), &[C_VAL; (H * M) as usize]),
        ctx.init_tensor_f32([N, H].to_vec(), &[D_VAL; (N * H) as usize]),
        ctx.init_tensor_f32([H, H].to_vec(), &[E_VAL; (H * H) as usize]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx
        .upload(&saved_tensors.seed, &[1_f32; (H * H) as usize], 0)
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

    let max_len = in_tensors.iter().map(|x| x.len()).max().unwrap_or(0) as usize;
    let mut dst = alloc::vec![0.0; max_len];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;
    let saved_tensors = &saved_tensors.forward_saved;

    ctx.download(&saved_tensors[0], &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(download.iter().all(|x| *x == X_VAL));

    ctx.download(&saved_tensors[1], &mut dst).unwrap();
    let download = &dst[..(H * N) as usize];
    assert!(download.iter().all(|x| *x == Y_VAL));

    ctx.download(&saved_tensors[2], &mut dst).unwrap();
    let download = &dst[..(H * H) as usize];
    assert!(download.iter().all(|x| *x == Z_VAL));

    ctx.download(out_tensor, &mut dst).unwrap();
    let download = &dst[..(H * H) as usize];
    assert!(download.iter().all(|x| *x == Z_VAL + E_VAL));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    let download = &dst[..(M * K) as usize];
    assert!(download.iter().all(|x| *x == A_GRAD));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    let download = &dst[..(K * N) as usize];
    assert!(download.iter().all(|x| *x == B_GRAD));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    let download = &dst[..(H * M) as usize];
    assert!(download.iter().all(|x| *x == C_GRAD));

    ctx.download(&grad_tensors[3], &mut dst).unwrap();
    let download = &dst[..(N * H) as usize];
    assert!(download.iter().all(|x| *x == D_GRAD));

    ctx.download(&grad_tensors[4], &mut dst).unwrap();
    let download = &dst[..(H * H) as usize];
    assert!(download.iter().all(|x| *x == 1.0));
}

#[test]
fn matmul_sub_forward_backward() {
    const M: u32 = 32;
    const N: u32 = 64;
    const K: u32 = 128;

    const A_VAL: f32 = 3.0;
    const B_VAL: f32 = 2.0;
    const C_VAL: f32 = 1.0;
    const D_VAL: f32 = 0.5;

    const X_VAL: f32 = A_VAL * B_VAL * K as f32;
    const Y_VAL: f32 = C_VAL * D_VAL * K as f32;
    const Z_VAL: f32 = X_VAL - Y_VAL;

    const A_GRAD: f32 = B_VAL * N as f32;
    const B_GRAD: f32 = A_VAL * M as f32;
    const C_GRAD: f32 = -D_VAL * N as f32;
    const D_GRAD: f32 = -C_VAL * M as f32;

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();

    let state = OptimState::STOCHASTIC_GRADIENT_DESCENT;

    let mut graph = Graph::new(
        LossType::MEAN_SQUARED_ERROR,
        OptimType::STOCHASTIC_GRADIENT_DESCENT,
    );

    let a = graph.input(&[m, k], DType::F32);
    let b = graph.input(&[k, n], DType::F32);
    let c = graph.input(&[m, k], DType::F32);
    let d = graph.input(&[k, n], DType::F32);

    let x = graph.matmul(a, b);
    let y = graph.matmul(c, d);

    graph.sub(x, y);

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let ctx = GpuContext::new().unwrap();

    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f32([M, K].to_vec(), &[A_VAL; (M * K) as usize]),
        ctx.init_tensor_f32([K, N].to_vec(), &[B_VAL; (K * N) as usize]),
        ctx.init_tensor_f32([M, K].to_vec(), &[C_VAL; (M * K) as usize]),
        ctx.init_tensor_f32([K, N].to_vec(), &[D_VAL; (K * N) as usize]),
    ];

    let meta_binding = [1e-3_f32.to_bits(), M, N, K];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let upload = ctx
        .upload(&saved_tensors.seed, &[1_f32; (M * N) as usize], 0)
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

    let max_len = in_tensors.iter().map(|x| x.len()).max().unwrap_or(0) as usize;
    let mut dst = alloc::vec![0.0; max_len];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;
    let saved_tensors = &saved_tensors.forward_saved;

    ctx.download(&saved_tensors[0], &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(download.iter().all(|x| *x == X_VAL));

    ctx.download(&saved_tensors[1], &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(download.iter().all(|x| *x == Y_VAL));

    ctx.download(out_tensor, &mut dst).unwrap();
    let download = &dst[..(M * N) as usize];
    assert!(download.iter().all(|x| *x == Z_VAL));

    ctx.download(&grad_tensors[0], &mut dst).unwrap();
    let download = &dst[..(M * K) as usize];
    assert!(download.iter().all(|x| *x == A_GRAD));

    ctx.download(&grad_tensors[1], &mut dst).unwrap();
    let download = &dst[..(K * N) as usize];
    assert!(download.iter().all(|x| *x == B_GRAD));

    ctx.download(&grad_tensors[2], &mut dst).unwrap();
    let download = &dst[..(M * K) as usize];
    assert!(download.iter().all(|x| *x == C_GRAD));

    ctx.download(&grad_tensors[3], &mut dst).unwrap();
    let download = &dst[..(K * N) as usize];
    assert!(download.iter().all(|x| *x == D_GRAD));
}
