//! Revival of `bpat` storage format used in `briny_ai`.

use core::fmt;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use crate::{
    dispatch::{GpuBackend, GpuContext},
    errors::{Error, ErrorKind},
    tensor::Tensor,
};

pub mod headers;

pub mod versions;

/// An enumerated header dispatching BPAT formats.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BpatHeader {
    /// A heavy but precise format using `f64`
    ///
    /// [`headers::BPAT_MAGIC_V0`]
    BpatV0,

    /// A format that guarantees data integrity
    ///
    /// [`headers::BPAT_MAGIC_V1`]
    BpatV1,

    /// A shorter format like `v1`, but using `f32`.
    ///
    /// [`headers::BPAT_MAGIC_V1_MICRO`]
    BpatV1M,

    /// A very compact format
    ///
    /// [`headers::BPAT_MAGIC_V2_F16`]
    BpatV2f16,

    /// A very compact format
    ///
    /// [`headers::BPAT_MAGIC_V2_BF16`]
    BpatV2bf16,

    /// Similar to `v0`, using `f32`
    ///
    /// [`headers::BPAT_MAGIC_V2_F32`]
    BpatV2f32,
}

/// The type of error in tensor serialization.
#[derive(Debug)]
pub enum SerialTensorError {
    /// An error occurred within the GPU, not I/O
    Unrelated,

    /// Bad integrity: e.g. mismatched checksums.
    IntegrityUnverified,

    /// Invalid data: e.g. shape product != data len
    InvalidData,

    /// Invalid header: e.g. not valid `bpat` signature
    InvalidHeader,

    /// Invalid path: e.g. no file exists
    InvalidPath,

    /// Failed to read/write to/from file.
    FailedFileIo,
}

impl fmt::Display for SerialTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegrityUnverified => write!(f, "integrity unverified"),
            Self::InvalidData => write!(f, "invalid data"),
            Self::InvalidHeader => write!(f, "invalid header"),
            Self::InvalidPath => write!(f, "invalid path"),
            Self::Unrelated => write!(f, "[other]"),
            Self::FailedFileIo => write!(f, "failed file I/O"),
        }
    }
}

/// Saves the given tensors to a file.
///
/// # Errors
///
/// Failure conditions are as follows:
///
/// - The tensors are invalid
/// - The file path is invalid
pub fn save_tensors<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
    header: BpatHeader,
) -> Result<(), Error<SerialTensorError>> {
    match header {
        BpatHeader::BpatV0 => versions::v0_v2::save_tensors_v0(path, ctx, tensors),
        BpatHeader::BpatV1 => versions::v1::save_tensors_v1(path, ctx, tensors),
        BpatHeader::BpatV1M => versions::v1::save_tensors_v1m(path, ctx, tensors),
        BpatHeader::BpatV2bf16 => versions::v0_v2::save_tensors_v2_bf16(path, ctx, tensors),
        BpatHeader::BpatV2f16 => versions::v0_v2::save_tensors_v2_f16(path, ctx, tensors),
        BpatHeader::BpatV2f32 => versions::v0_v2::save_tensors_v2_f32(path, ctx, tensors),
    }
}

/// Loads tensors from a file.
///
/// # Errors
///
/// Returns a [`SerialTensorError`] on failure to load tensors.
///
/// - The file exists
/// - The file is in `bpat` format
pub fn load_tensors<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
) -> Result<Vec<Tensor<B>>, Error<SerialTensorError>> {
    let mut file = BufReader::new(File::open(path).map_err(|_| Error {
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidPath,
        msg: "no such file exists",
    })?);

    let mut file_start = [0; 4];
    file.read_exact(&mut file_start).map_err(|_| Error {
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidHeader,
        msg: "header not found",
    })?;

    if file_start == headers::BPAT_MAGIC_V0 {
        return versions::v0_v2::load_tensors_v0(&mut file, ctx);
    }

    let mut magic_end = [0; 4];
    file.read_exact(&mut magic_end).map_err(|_| Error {
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidHeader,
        msg: "header not found",
    })?;

    match_magic(&file_start, &magic_end)?(&mut file, ctx)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn match_magic<B: GpuBackend>(
    magic_start: &[u8; 4],
    magic_end: &[u8; 4],
) -> Result<
    fn(&mut BufReader<File>, &GpuContext<B>) -> Result<Vec<Tensor<B>>, Error<SerialTensorError>>,
    Error<SerialTensorError>,
> {
    use headers::{
        BPAT_MAGIC_V1, BPAT_MAGIC_V1_MICRO, BPAT_MAGIC_V2_BF16, BPAT_MAGIC_V2_F16,
        BPAT_MAGIC_V2_F32,
    };

    let func = if BPAT_MAGIC_V1.starts_with(magic_start) && BPAT_MAGIC_V1.ends_with(magic_end) {
        versions::v1::load_tensors_v1
    } else if BPAT_MAGIC_V1_MICRO.starts_with(magic_start)
        && BPAT_MAGIC_V1_MICRO.ends_with(magic_end)
    {
        versions::v1::load_tensors_v1m
    } else if BPAT_MAGIC_V2_BF16.starts_with(magic_start) && BPAT_MAGIC_V2_BF16.ends_with(magic_end)
    {
        versions::v0_v2::load_tensors_v2_bf16
    } else if BPAT_MAGIC_V2_F16.starts_with(magic_start) && BPAT_MAGIC_V2_F16.ends_with(magic_end) {
        versions::v0_v2::load_tensors_v2_f16
    } else if BPAT_MAGIC_V2_F32.starts_with(magic_start) && BPAT_MAGIC_V2_F32.ends_with(magic_end) {
        versions::v0_v2::load_tensors_v2_f32
    } else {
        return Err(Error {
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::InvalidHeader,
            msg: "invalid magic header",
        });
    };

    Ok(func)
}
