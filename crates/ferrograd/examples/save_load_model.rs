use ferrograd::{
    dispatch::{
        CompilationOptions, SimpleDType, DebugCompilationOptions, GpuContext, Graph, Metadata,
        OptCompilationOptions,
    },
    io::BpatHeader,
    nn::{MEAN_SQUARED_ERROR, Optim},
};
use gpu_telemetry::monitor::{GpuMonitor, telemetry::Telemetry};
use rand_core::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::time::{Duration, Instant};

const PATH: &str = "data/v2f32_test.bpat";

const ITERS: usize = 32;
const EPOCHS: usize = 64;

const LR: f32 = 1e-10;

const INTERVAL: Duration = Duration::from_millis(50);

fn main() {
    const M: u32 = 768;
    const N: u32 = 384;
    const K: u32 = 512;
    const H: u32 = 256;

    const A_VAL: f32 = 0.03;
    const B_VAL: f32 = 0.02;
    const C_VAL: f32 = 0.01;
    const D_VAL: f32 = 0.05;
    const E_VAL: f32 = 1.0;

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();
    let h = meta.new_field();

    let optim = Optim::sgd();

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, optim.lower);

    {
        let mut graph = graph.define_ops();

        let a = graph.input(&[m, k], SimpleDType::F32);
        let b = graph.input(&[k, n], SimpleDType::F32);
        let c = graph.input(&[h, m], SimpleDType::F32);
        let d = graph.input(&[n, h], SimpleDType::F32);
        let e = graph.input(&[h, h], SimpleDType::F32);

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

    let meta_binding = [LR.to_bits(), M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));
    let meta_binding = ctx.alloc_meta(&meta_binding);

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, meta_binding, &optim.state).unwrap();

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = ctx
        .load_tensors(PATH)
        .unwrap_or_else(|_| vec![
            ctx.init_tensor_f32([M, K].to_vec(), &[A_VAL; (M * K) as usize]).unwrap(),
            ctx.init_tensor_f32([K, N].to_vec(), &[B_VAL; (K * N) as usize]).unwrap(),
            ctx.init_tensor_f32([H, M].to_vec(), &[C_VAL; (H * M) as usize]).unwrap(),
            ctx.init_tensor_f32([N, H].to_vec(), &[D_VAL; (N * H) as usize]).unwrap(),
            ctx.init_tensor_f32([H, H].to_vec(), &[E_VAL; (H * H) as usize]).unwrap(),
        ]);

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed\n");

    let mut schedule = ctx
        .schedule(
            kernels,
            meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &optim.state,
        )
        .unwrap();

    let mut epoch = 0;

    let mut rng = XorShiftRng::seed_from_u64(0);

    loop {
        println!("EPOCH {epoch}:");

        let target = {
            let target = 2.0 * (rng.next_u32() as f32 / u32::MAX as f32) - 1.0;

            let arr = [target; (H * H) as usize];

            ctx.init_tensor_f32(vec![H, H], &arr)
        }.unwrap();

        let monitor: GpuMonitor<Telemetry> = GpuMonitor::start(INTERVAL).unwrap();

        for _ in 0..ITERS {
            ctx.dispatch_forward(&schedule).unwrap();
            ctx.dispatch_loss(&mut schedule, &target).unwrap();
            ctx.dispatch_backward(&schedule).unwrap();

            ctx.dispatch_optim(&mut schedule, &in_tensors[0], 0, &saved_tensors).unwrap();
            ctx.dispatch_optim(&mut schedule, &in_tensors[1], 1, &saved_tensors).unwrap();
            ctx.dispatch_optim(&mut schedule, &in_tensors[2], 2, &saved_tensors).unwrap();
            ctx.dispatch_optim(&mut schedule, &in_tensors[3], 3, &saved_tensors).unwrap();
            ctx.dispatch_optim(&mut schedule, &in_tensors[4], 4, &saved_tensors).unwrap();
        }

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
        println!(" MAXIMUM MEMORY BUDGET: {}\n", max_budget);

        if epoch % EPOCHS == EPOCHS - 1 {
            ctx.sync().unwrap();

            ctx.save_tensors(PATH, &in_tensors, BpatHeader::BpatV2f32)
                .unwrap();
        }

        epoch += 1;
    }
}
