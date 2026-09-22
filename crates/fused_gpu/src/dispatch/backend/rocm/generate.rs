use crate::dispatch::{
    CompilationOptions,
    backend::{Param, kernel::RawKernel},
};

pub fn generate_hip(_src: &RawKernel, _params: &[Param], _options: &CompilationOptions) -> String {
    let mut hip = String::new();

    hip.push_str("");

    hip
}
