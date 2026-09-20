use crate::dispatch::{CompilationOptions, backend::{Param, kernel::RawKernel}};

pub fn generate_hip(
    src: &RawKernel,
    params: &[Param],
    options: &CompilationOptions,
) -> String {
    "".into()
}
