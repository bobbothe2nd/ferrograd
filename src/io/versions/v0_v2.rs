use std::{fs::File, io::{BufWriter, Write}};

use briny::{raw::cast::cast_slice, traits::Pod};

use crate::{dispatch::{GpuBackend, GpuContext}, errors::{Error, ErrorKind}, io::{SerialTensorError, headers::BPAT_MAGIC_V0}, tensor::Tensor};

/// Currently requires tensors to be `f64`.
pub fn save_tensors_v0<B: GpuBackend>(
    path: &str,
    ctx: GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<8, f64, B>(path, ctx, tensors)
}

/// Currently requires tensors to be `f32`.
pub fn save_tensors_v2_f32<B: GpuBackend>(
    path: &str,
    ctx: GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, f32, B>(path, ctx, tensors)
}

/// Currently requires tensors to be `f16`.
pub fn save_tensors_v2_f16<B: GpuBackend>(
    path: &str,
    ctx: GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, half::f16, B>(path, ctx, tensors)
}

/// Currently requires tensors to be `bf16`.
pub fn save_tensors_v2_bf16<B: GpuBackend>(
    path: &str,
    ctx: GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, half::bf16, B>(path, ctx, tensors)
}

#[inline(always)]
fn save_tensors<const U: usize, F: Default + Pod + Clone, B: GpuBackend>(
    path: &str,
    ctx: GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    let mut file = BufWriter::new(File::create(path).map_err(|_| Error {
        msg: "file not found",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidPath,
    })?);

    file.write(&BPAT_MAGIC_V0);
    file.write(&[tensors.len().try_into().map_err(|_| Error {
        msg: "exceeds max tensors of 255",
        kind: ErrorKind::InvalidArgument,
        ctx: SerialTensorError::Unrelated,
    })?]);

    for t in tensors {
        file.write(&(t.rank()).to_le_bytes()[..U]);

        let shape_u64 = t.shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        file.write(cast_slice(&shape_u64));

        let len = t.data.size_bytes() as usize / size_of::<F>();

        if len != t.shape.iter().product::<u32>() as usize {
            return Err(Error {
                msg: "tensor prod(shape)/data.size mismatch (is it F?)",
                kind: ErrorKind::InvalidArgument,
                ctx: SerialTensorError::InvalidData,
            })
        }

        let mut buf = vec![F::default(); len];

        ctx.download::<F, _>(t, &mut buf).map_err(|err| Error {
            msg: err.msg,
            kind: err.kind,
            ctx: SerialTensorError::Unrelated,
        })?;

        file.write(cast_slice(&buf));
    }

    Ok(())
}
