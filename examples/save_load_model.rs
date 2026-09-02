use fused_gpu::{
    dispatch::{
        CompilationOptions, GpuContext,
        backend::{DType, Graph, LossType, Metadata, OptimState, OptimType},
    },
    io::BpatHeader,
    tensor::bf16,
};
use gpu_telemetry::monitor::{GpuMonitor, telemetry::Telemetry};
use rand_core::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{
    array::from_fn,
    time::{Duration, Instant},
};

const ITERS: usize = 16;

fn main() {
    const M: u32 = 768;
    const N: u32 = 1024;
    const K: u32 = 512;
    const H: u32 = 256;

    const A_VAL: bf16 = bf16::from_f32_const(3.0);
    const B_VAL: bf16 = bf16::from_f32_const(2.0);
    const C_VAL: bf16 = bf16::from_f32_const(1.0);
    const D_VAL: bf16 = bf16::from_f32_const(0.5);
    const E_VAL: bf16 = bf16::from_f32_const(1.0);

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

    {
        let a = graph.input(&[m, k], DType::F32);
        let b = graph.input(&[k, n], DType::F32);
        let c = graph.input(&[h, m], DType::F32);
        let d = graph.input(&[n, h], DType::F32);
        let e = graph.input(&[h, h], DType::F32);

        let x = graph.matmul(a, b);
        let y = graph.matmul(c, x);
        let z = graph.matmul(y, d);
        graph.add(z, e);
    }

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let compile_start = Instant::now();

    let ctx = GpuContext::new().unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = ctx
        .load_tensors("data/v2bf16_test.bpat")
        .unwrap_or_else(|_| {
            vec![
                ctx.init_tensor_bf16([M, K].to_vec(), &[A_VAL; (M * K) as usize]),
                ctx.init_tensor_bf16([K, N].to_vec(), &[B_VAL; (K * N) as usize]),
                ctx.init_tensor_bf16([H, M].to_vec(), &[C_VAL; (H * M) as usize]),
                ctx.init_tensor_bf16([N, H].to_vec(), &[D_VAL; (N * H) as usize]),
                ctx.init_tensor_bf16([H, H].to_vec(), &[E_VAL; (H * H) as usize]),
            ]
        });

    let meta_binding = [1e-3_f32.to_bits(), M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding, &state);

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed\n");

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

    let mut epoch = 0;

    loop {
        println!("EPOCH {epoch}:");

        let target = {
            let mut rng = XorShiftRng::seed_from_u64(0);

            let arr: [bf16; (H * H) as usize] = from_fn(|_| bf16::from_bits(rng.next_u32() as u16));

            ctx.init_tensor_bf16(vec![H, H], &arr)
        };

        let monitor: GpuMonitor<Telemetry> = GpuMonitor::start(Duration::from_millis(10)).unwrap();

        let mut previous_encoded = {
            let mut state = ctx.prepare_batch();

            {
                let mut pass = ctx.start_batch(&mut state);

                for _ in 0..ITERS {
                    pass.dispatch_forward(&schedule);
                    pass.dispatch_loss(&mut schedule, &target);
                    pass.dispatch_backward(&schedule);

                    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[0], 0, &saved_tensors);
                    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[1], 1, &saved_tensors);
                    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[2], 2, &saved_tensors);
                    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[3], 3, &saved_tensors);
                    pass.dispatch_optim::<0>(&mut schedule, &in_tensors[4], 4, &saved_tensors);
                }
            }

            state.encode()
        };

        for _ in 0..3 {
            let sync_start = Instant::now();

            let submission = previous_encoded.submit();

            previous_encoded = {
                let mut state = ctx.prepare_batch();

                {
                    let mut pass = ctx.start_batch(&mut state);

                    for _ in 0..ITERS {
                        pass.dispatch_forward(&schedule);
                        pass.dispatch_loss(&mut schedule, &target);
                        pass.dispatch_backward(&schedule);

                        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[0], 0, &saved_tensors);
                        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[1], 1, &saved_tensors);
                        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[2], 2, &saved_tensors);
                        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[3], 3, &saved_tensors);
                        pass.dispatch_optim::<0>(&mut schedule, &in_tensors[4], 4, &saved_tensors);
                    }
                }

                state.encode()
            };

            submission.sync();

            let sync_elapsed = sync_start.elapsed();

            println!("  SYNCHRONIZATION: {sync_elapsed:?} elapsed");
        }

        let sync_start = Instant::now();

        let submission = previous_encoded.submit();

        submission.sync();

        let sync_elapsed = sync_start.elapsed();

        println!("  SYNCHRONIZATION: {sync_elapsed:?} elapsed");

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

        if epoch % 10 == 9 {
            ctx.save_tensors("data/v2bf16_test.bpat", &in_tensors, BpatHeader::BpatV2bf16)
                .unwrap();
        }

        epoch += 1;
    }
}
