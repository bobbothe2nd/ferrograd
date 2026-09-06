use ferrograd::{
    dispatch::{
        CompilationOptions, DType, DebugCompilationOptions, Dynamic, GpuContext, Graph, Metadata,
        OptCompilationOptions, Schedule, SyncSubmission,
    },
    nn::{MEAN_SQUARED_ERROR, Optim},
    tensor::f16,
};
use gpu_telemetry::monitor::{GpuMonitor, telemetry::Telemetry};
use std::time::{Duration, Instant};

fn encode<'a>(
    ctx: &'a GpuContext,
    schedule: &'a Schedule,
    iters: usize,
) -> SyncSubmission<'a, Dynamic> {
    let mut state = ctx.prepare_batch();

    {
        let mut pass = ctx.start_batch(&mut state);

        for _ in 0..iters {
            pass.dispatch_forward(schedule);

            pass.dispatch_backward(schedule);
        }
    }

    state.encode()
}

fn model_runtime(ctx: &GpuContext, schedule: &Schedule) {
    const ITERS: [usize; 6] = [50, 100, 250, 100, 10, 1];

    let mut prev_iters = ITERS[0];
    let mut previous_encoded = encode(ctx, schedule, prev_iters);

    for iters in ITERS.into_iter().skip(1) {
        let sync_start = Instant::now();
        let runtime_start = Instant::now();

        let submission = previous_encoded.submit();

        let runtime_elapsed = runtime_start.elapsed();

        println!("MODEL SUBMISSION LATENCY: {runtime_elapsed:?} elapsed");

        previous_encoded = encode(ctx, schedule, iters);

        submission.sync();

        let sync_elapsed = sync_start.elapsed();

        println!("  SYNCHRONIZATION {prev_iters}: {sync_elapsed:?} elapsed");

        prev_iters = iters;
    }

    let sync_start = Instant::now();
    let runtime_start = Instant::now();

    let submission = previous_encoded.submit();

    let runtime_elapsed = runtime_start.elapsed();

    submission.sync();

    let sync_elapsed = sync_start.elapsed();

    println!("MODEL SUBMISSION LATENCY: {runtime_elapsed:?} elapsed");
    println!("  SYNCHRONIZATION 1: {sync_elapsed:?} elapsed");
}

fn main() {
    const M: u32 = 768;
    const N: u32 = 1024;
    const K: u32 = 512;
    const H: u32 = 256;

    const A_VAL: f16 = f16::from_f32_const(3.0);
    const B_VAL: f16 = f16::from_f32_const(2.0);
    const C_VAL: f16 = f16::from_f32_const(1.0);
    const D_VAL: f16 = f16::from_f32_const(0.5);
    const E_VAL: f16 = f16::from_f32_const(1.0);

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();
    let h = meta.new_field();

    let state = Optim::STOCHASTIC_GRADIENT_DESCENT.state;

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, Optim::STOCHASTIC_GRADIENT_DESCENT.lower);

    {
        let mut graph = graph.define_ops();

        let a = graph.input(&[m, k], DType::F16);
        let b = graph.input(&[k, n], DType::F16);
        let c = graph.input(&[h, m], DType::F16);
        let d = graph.input(&[n, h], DType::F16);
        let e = graph.input(&[h, h], DType::F16);

        let x = graph.matmul(a, b);
        let y = graph.matmul(c, x);
        let z = graph.matmul(y, d);
        graph.add(z, e);
    }

    let saved = graph.compute_saved_nodes();
    graph.validate(meta).unwrap();

    let compile_start = Instant::now();

    let ctx = GpuContext::new().unwrap();
    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    let meta_binding = [1e-3_f32.to_bits(), M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));
    let meta_binding = ctx.alloc_meta(&meta_binding);

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, meta_binding, &state);

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = [
        ctx.init_tensor_f16([M, K].to_vec(), &[A_VAL; (M * K) as usize]),
        ctx.init_tensor_f16([K, N].to_vec(), &[B_VAL; (K * N) as usize]),
        ctx.init_tensor_f16([H, M].to_vec(), &[C_VAL; (H * M) as usize]),
        ctx.init_tensor_f16([N, H].to_vec(), &[D_VAL; (N * H) as usize]),
        ctx.init_tensor_f16([H, H].to_vec(), &[E_VAL; (H * H) as usize]),
    ];

    ctx.upload(&saved_tensors.seed, &[1_f32; (H * H) as usize], 0)
        .unwrap()
        .sync();

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed");

    let schedule = ctx
        .schedule(
            kernels,
            meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &state,
        )
        .unwrap();

    let monitor: GpuMonitor<Telemetry> = GpuMonitor::start(Duration::from_millis(10)).unwrap();

    model_runtime(&ctx, &schedule);

    let telemetry = monitor.stop().unwrap();

    let mut max_budget = 0;
    let mut accum_usage = 0;

    for sample in &telemetry.samples {
        for heap in &sample.heaps {
            if let Some(usage) = heap.usage {
                accum_usage += usage;
            }

            if let Some(budget) = heap.budget
                && budget > max_budget
            {
                max_budget = budget;
            }
        }
    }

    let avg_usage = accum_usage / (telemetry.samples.len() as u64);

    println!("\n  AVERAGE MEMORY USAGE: {}", avg_usage);
    println!(" MAXIMUM MEMORY BUDGET: {}", max_budget);
}
