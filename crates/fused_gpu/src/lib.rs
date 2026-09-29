//! `fused_gpu` implements optimized GPU kernels for almost any operation.
//!
//! It supports CUDA, ROCM, and WGSL. It will compile anything the feature set describes:
//!
//! - `wgsl`: Enable WGSL support
//! - `io`: Enable support for I/O (`bpat`)
//! - `telemetry`: Cleans `gpu_telemetry` error handling
//!
//! You can also use custom backends.

#![forbid(
    clippy::unimplemented,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::approx_constant,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unconditional_recursion,
    clippy::std_instead_of_core
)]
#![deny(clippy::pedantic, clippy::nursery, clippy::all, unsafe_code)]
#![allow(
    clippy::too_many_lines,
    clippy::cast_ptr_alignment,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

pub mod dispatch;

#[cfg(feature = "io")]
pub mod io;

pub mod errors;

pub mod tensor;
