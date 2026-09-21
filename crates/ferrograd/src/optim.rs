use ferrograd_nn::optim::{STOCHASTIC_GRADIENT_DESCENT, stochastic_gradient_descent_state};
use fused_gpu::dispatch::backend::{OptimState, OptimType};

pub struct Optim {
    pub state: OptimState,
    pub lower: OptimType,
}

impl Optim {
    pub fn sgd() -> Self {
        Self {
            state: stochastic_gradient_descent_state(),
            lower: STOCHASTIC_GRADIENT_DESCENT,
        }
    }
}
