use fused_gpu::dispatch::backend::{Op, OptimState, OptimType, ValueState};

pub const STOCHASTIC_GRADIENT_DESCENT: OptimType = OptimType {
    lower: |kernel, dtype, lr, weight_param, grad_param, gid, _, _| {
        let grad = kernel.raw.def_var(
            dtype,
            ValueState::Immut,
            Some(Op::ParamLoad {
                param: grad_param,
                index: gid,
            }),
        );

        let scaled_grad =
            kernel
                .raw
                .def_var(dtype, ValueState::Immut, Some(Op::Mul { a: grad, b: lr }));

        let zero = kernel
            .raw
            .def_var(dtype, ValueState::Inline, Some(dtype.constant_float(0.0)?));

        kernel.raw.param_sub(weight_param, gid, scaled_grad);
        kernel.raw.param_store(grad_param, gid, zero);

        Ok(())
    },
};

pub const STOCHASTIC_GRADIENT_DESCENT_STATE: OptimState<0> = OptimState { shapes: [] };
