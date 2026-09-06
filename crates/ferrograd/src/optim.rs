use ferrograd_nn::optim::{STOCHASTIC_GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT_STATE};
use fused_gpu::dispatch::backend::{OptimState, OptimType};

pub struct Optim<const N: usize> {
    pub state: OptimState<N>,
    pub lower: OptimType,
}

impl Optim<0> {
    pub const STOCHASTIC_GRADIENT_DESCENT: Self = Self {
        state: STOCHASTIC_GRADIENT_DESCENT_STATE,
        lower: STOCHASTIC_GRADIENT_DESCENT,
    };
}
