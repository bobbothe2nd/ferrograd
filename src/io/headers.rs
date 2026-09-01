//! Clearly documented headers for all BPAT versions.

/// The original BPAT header. Used `f64` storage.
///
/// Used on `briny_ai` `v0.1.0`-`v0.2.2`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┐
/// │ Header       │ Tensors                    │
/// ├──────────────┼────────────────────────────┤
/// │ `bpat`       │ u64: ndim                  │
/// │ u8: count    │ [u64; ndim] shape          │
/// │              │ [f64; prod(shape)] data    │
/// └──────────────┴────────────────────────────┘
/// ```
pub const BPAT_MAGIC_V0: [u8; 4] = *b"bpat";

/// The first BPAT header with checksums.
///
/// Created on `briny_ai` `v0.3.0`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┬────────────────────┐
/// │ Header       │ Tensors                    │ Checksum           │
/// ├──────────────┼────────────────────────────┼────────────────────┤
/// │ `BPATv1\0\0` │ u64: ndim                  │ u32: file checksum │
/// │ u8: count    │ [u64; ndim] shape          │                    │
/// │              │ [f64; prod(shape)] data    │                    │
/// │              │ u32: checksum              │                    │
/// └──────────────┴────────────────────────────┴────────────────────┘
/// ```
pub const BPAT_MAGIC_V1: [u8; 8] = *b"BPATv1\0\0";

/// A more compact BPAT format with checksums. Started using `f32` storage.
///
/// Created on `briny_ai` `v0.6.0`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┬────────────────────┐
/// │ Header       │ Tensors                    │ Checksum           │
/// ├──────────────┼────────────────────────────┼────────────────────┤
/// │ `BPATv1m\0`  │ u32: ndim                  │ u32: file checksum │
/// │ u8: count    │ [u32; ndim] shape          │                    │
/// │              │ [f32; prod(shape)] data    │                    │
/// │              │ u32: checksum              │                    │
/// └──────────────┴────────────────────────────┴────────────────────┘
/// ```
pub const BPAT_MAGIC_V1_MICRO: [u8; 8] = *b"BPATv1m\0";

/// The most compact BPAT header. Started using `f16` storage.
///
/// Created on `fused_gpu` `v0.1.0-alpha.4`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┐
/// │ Header       │ Tensors                    │
/// ├──────────────┼────────────────────────────┤
/// │ `BPAT\0f16`  │ u32: ndim                  │
/// │ u8: count    │ [u32; ndim] shape          │
/// │              │ [f16; prod(shape)] data    │
/// └──────────────┴────────────────────────────┘
/// ```
pub const BPAT_MAGIC_V2_F16: [u8; 8] = *b"BPAT\0f16";

/// Same as [`BPAT_MAGIC_V2_F16`], but uses `bf16` storage instead.
///
/// Created on `fused_gpu` `v0.1.0-alpha.4`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┐
/// │ Header       │ Tensors                    │
/// ├──────────────┼────────────────────────────┤
/// │ `BPATbf16`   │ u32: ndim                  │
/// │ u8: count    │ [u32; ndim] shape          │
/// │              │ [bf16; prod(shape)] data   │
/// └──────────────┴────────────────────────────┘
/// ```
pub const BPAT_MAGIC_V2_BF16: [u8; 8] = *b"BPATbf16";

/// Same as [`BPAT_MAGIC_V2_F16`], but uses `bf16` storage instead.
///
/// Created on `fused_gpu` `v0.1.0-alpha.4`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┐
/// │ Header       │ Tensors                    │
/// ├──────────────┼────────────────────────────┤
/// │ `BPAT\0f32`  │ u32: ndim                  │
/// │ u8: count    │ [u32; ndim] shape          │
/// │              │ [f32; prod(shape)] data    │
/// └──────────────┴────────────────────────────┘
/// ```
pub const BPAT_MAGIC_V2_F32: [u8; 8] = *b"BPAT\0f32";
