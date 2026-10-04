use ferrograd::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, GpuContext, Graph, Metadata,
        OptCompilationOptions, SimpleDType,
    },
    io::BpatHeader,
    nn::{MEAN_SQUARED_ERROR, Optim},
};
use rand_core::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{io::Write, time::Instant};

const PATH: &str = "data/v2f32_test.bpat";

const ITERS: usize = 256;
const EPOCHS_TO_SAVE: usize = 32;

const LR: f32 = 1e-10;

fn main() {
    const M: u32 = 768;
    const N: u32 = 384;
    const K: u32 = 512;
    const H: u32 = 256;

    const A_VAL: f32 = 0.002;
    const B_VAL: f32 = 0.003;
    const C_VAL: f32 = 0.01;
    const D_VAL: f32 = 0.007;
    const E_VAL: f32 = 0.008;

    stderrlog::new().verbosity(log::Level::Debug).init().unwrap();

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

        let z_res = graph.add(z, e);

        let p = graph.matmul(z_res, e);

        let out = graph.add(p, z_res);

        graph.matmul(out, e);
    }

    let saved = graph.compute_saved_nodes();
    graph.validate(meta).unwrap();

    let compile_start = Instant::now();

    let ctx = GpuContext::new().unwrap();
    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::PRETTY_PRINT_IR,
    };

    let meta_binding = [LR.to_bits(), M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));
    let meta_binding = ctx.alloc_meta(&meta_binding);

    let saved_tensors = ctx
        .alloc_tensors(&graph, &saved, meta_binding, &optim.state)
        .unwrap();

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = ctx.load_tensors(PATH).unwrap_or_else(|_| {
        vec![
            ctx.init_tensor_f32(&[M, K], &[A_VAL; (M * K) as usize])
                .unwrap(),
            ctx.init_tensor_f32(&[K, N], &[B_VAL; (K * N) as usize])
                .unwrap(),
            ctx.init_tensor_f32(&[H, M], &[C_VAL; (H * M) as usize])
                .unwrap(),
            ctx.init_tensor_f32(&[N, H], &[D_VAL; (N * H) as usize])
                .unwrap(),
            ctx.init_tensor_f32(&[H, H], &[E_VAL; (H * H) as usize])
                .unwrap(),
        ]
    });

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed/n");

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

    let target = {
        let val = 2.0 * (rng.next_u32() as f32 / u32::MAX as f32) - 1.0;
        ctx.init_tensor_f32(&[H, H], &[val; (H * H) as usize]).unwrap()
    };

    loop {
        {
            let val = 2.0 * (rng.next_u32() as f32 / u32::MAX as f32) - 1.0;
            ctx.upload(&target, &[val; H as usize], H * (epoch as u32 % H)).unwrap()
        }

        {
            let launch_start = Instant::now();

            for _ in 0..4 {
                for _ in 0..(ITERS / 4) {
                    ctx.dispatch_forward(&schedule).unwrap();
                    ctx.dispatch_loss(&mut schedule, &target).unwrap();
                    ctx.dispatch_backward(&schedule).unwrap();
                }

                ctx.dispatch_optim(&mut schedule).unwrap();
            }

            let launch_elapsed = launch_start.elapsed();

            println!("EPOCH {epoch} LAUNCH TIME: {launch_elapsed:?} elapsed");
        }

        if epoch % EPOCHS_TO_SAVE == EPOCHS_TO_SAVE - 1 {
            print!(" saving...");
            std::io::stdout().flush().unwrap();
            ctx.save_tensors(PATH, &in_tensors, BpatHeader::BpatV2f32).unwrap();

            print!("\r calculating loss...");
            std::io::stdout().flush().unwrap();
            let mut loss_t = Box::<[f32]>::new_uninit_slice((H * H) as usize);
            ctx.download(&saved_tensors.loss_t, &mut loss_t, 0).unwrap();
            let loss_t = unsafe { loss_t.assume_init() };
            let loss = loss_t.iter().sum::<f32>() / (H * H) as f32;

            print!("\r syncing...         ");
            std::io::stdout().flush().unwrap();
            ctx.sync().unwrap();

            println!("\r saved model @ loss {loss:?}");
        }

        epoch += 1;
    }
}
