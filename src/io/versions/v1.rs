use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

use briny::{
    raw::cast::{slice_to_bytes, slice_to_bytes_mut},
    traits::Pod,
};
use crc32fast::Hasher;

use crate::{
    dispatch::{GpuBackend, GpuContext},
    errors::{Error, ErrorKind},
    io::{SerialTensorError, headers::BPAT_MAGIC_V0},
    tensor::Tensor,
};

/// Currently requires tensors to be `f64`.
pub fn save_tensors_v1<B: GpuBackend>(
    path: &str,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<8, f64, B>(path, ctx, tensors)
}

/// Currently requires tensors to be `f32`.
pub fn save_tensors_v1m<B: GpuBackend>(
    path: &str,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, f32, B>(path, ctx, tensors)
}

#[inline]
fn save_tensors<const U: usize, F: Default + Pod + Clone, B: GpuBackend>(
    path: &str,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    let mut file = BufWriter::new(File::create(path).map_err(|_| Error {
        msg: "file not found",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidPath,
    })?);

    let mut file_hasher = Hasher::new();
    file_hasher.update(&BPAT_MAGIC_V0);

    let len = tensors.len().try_into().map_err(|_| Error {
        msg: "exceeds max tensors of 255",
        kind: ErrorKind::InvalidArgument,
        ctx: SerialTensorError::Unrelated,
    })?;
    file_hasher.update(&[len]);

    file.write(&BPAT_MAGIC_V0).map_err(|_| Error {
        msg: "filed to write to file",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::FailedFileIo,
    })?;
    file.write(&[len]).map_err(|_| Error {
        msg: "filed to write to file",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::FailedFileIo,
    })?;

    for t in tensors {
        let mut tensor_hasher = Hasher::new();

        let ndim = &(t.rank()).to_le_bytes()[..U];
        tensor_hasher.update(ndim);

        file.write(ndim).map_err(|_| Error {
            msg: "failed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        })?;

        let shape_u64 = t.shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let shape_u64 = slice_to_bytes(&shape_u64);
        tensor_hasher.update(shape_u64);

        file.write(shape_u64).map_err(|_| Error {
            msg: "filed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        })?;

        let size = t.data.size_bytes() as usize;
        let len = size / size_of::<F>();

        if len != t.shape.iter().product::<u32>() as usize {
            return Err(Error {
                msg: "tensor prod(shape)/data.size mismatch",
                kind: ErrorKind::InvalidArgument,
                ctx: SerialTensorError::InvalidData,
            });
        }

        let mut buf = vec![0u8; size];

        ctx.download(t, &mut buf).map_err(|err| Error {
            msg: err.msg,
            kind: err.kind,
            ctx: SerialTensorError::Unrelated,
        })?;

        tensor_hasher.update(&buf);

        file.write(&buf).map_err(|_| Error {
            msg: "filed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        })?;

        file_hasher.combine(&tensor_hasher);

        let tensor_crc = tensor_hasher.finalize().to_le_bytes();
        file_hasher.update(&tensor_crc);

        file.write(&tensor_crc).map_err(|_| Error {
            msg: "filed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        })?;
    }

    let file_crc = file_hasher.finalize().to_le_bytes();
    file.write(&file_crc).map_err(|_| Error {
        msg: "filed to write to file",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::FailedFileIo,
    })?;

    Ok(())
}

macro_rules! impl_load {
    ($name:ident, $name2:ident, $init:ident, $float:ident, $unsigned:ident) => {
        #[inline(always)]
        pub fn $name<B: GpuBackend>(
            file: &mut BufReader<File>,
            ctx: &GpuContext<B>,
        ) -> Result<Vec<Tensor<B>>, Error<SerialTensorError>> {
            let mut len = [0];
            file.read_exact(&mut len).map_err(|_| Error {
                msg: "unexpected EOF",
                kind: ErrorKind::SerializationError,
                ctx: SerialTensorError::FailedFileIo,
            })?;
            let len = u8::from_le_bytes(len) as usize;

            let mut tensors = Vec::with_capacity(len);

            for _ in 0..len {
                let mut rank = [0; size_of::<$unsigned>()];
                file.read_exact(&mut rank).map_err(|_| Error {
                    msg: "unexpected EOF",
                    kind: ErrorKind::SerializationError,
                    ctx: SerialTensorError::FailedFileIo,
                })?;
                let rank = <$unsigned>::from_le_bytes(rank);

                let mut shape = Vec::with_capacity(rank as usize);

                for _ in 0..rank {
                    let mut shape_u64 = [0; size_of::<$unsigned>()];
                    file.read_exact(&mut shape_u64).map_err(|_| Error {
                        msg: "unexpected EOF",
                        kind: ErrorKind::SerializationError,
                        ctx: SerialTensorError::FailedFileIo,
                    })?;
                    shape.push(<$unsigned>::from_le_bytes(shape_u64) as u32);
                }

                let len = shape.iter().product::<u32>();

                let mut data = vec![$float::default(); len as usize];

                file.read_exact(slice_to_bytes_mut(&mut data))
                    .map_err(|_| Error {
                        msg: "unexpected EOF",
                        kind: ErrorKind::SerializationError,
                        ctx: SerialTensorError::FailedFileIo,
                    })?;

                tensors.push(ctx.$init(shape, &data));
            }

            Ok(tensors)
        }

        #[inline(always)]
        pub fn $name2<B: GpuBackend>(
            file: &mut BufReader<File>,
            ctx: &GpuContext<B>,
            tensors: &mut [Tensor<B>],
        ) -> Result<(), Error<SerialTensorError>> {
            let mut len = [0];
            file.read_exact(&mut len).map_err(|_| Error {
                msg: "unexpected EOF",
                kind: ErrorKind::SerializationError,
                ctx: SerialTensorError::FailedFileIo,
            })?;
            let len = u8::from_le_bytes(len) as usize;

            let mut prev_sync = None::<$crate::dispatch::SubmissionIndex<B>>;

            for tensor in tensors.iter_mut().take(len) {
                let mut rank = [0; size_of::<$unsigned>()];
                file.read_exact(&mut rank).map_err(|_| Error {
                    msg: "unexpected EOF",
                    kind: ErrorKind::SerializationError,
                    ctx: SerialTensorError::FailedFileIo,
                })?;
                let rank = <$unsigned>::from_le_bytes(rank);

                let mut shape = Vec::with_capacity(rank as usize);

                for _ in 0..rank {
                    let mut shape_u64 = [0; size_of::<$unsigned>()];
                    file.read_exact(&mut shape_u64).map_err(|_| Error {
                        msg: "unexpected EOF",
                        kind: ErrorKind::SerializationError,
                        ctx: SerialTensorError::FailedFileIo,
                    })?;
                    shape.push(<$unsigned>::from_le_bytes(shape_u64) as u32);
                }

                let len = shape.iter().product::<u32>();

                let mut data = vec![$float::default(); len as usize];

                file.read_exact(slice_to_bytes_mut(&mut data))
                    .map_err(|_| Error {
                        msg: "unexpected EOF",
                        kind: ErrorKind::SerializationError,
                        ctx: SerialTensorError::FailedFileIo,
                    })?;

                tensor.shape = shape;

                if let Some(prev_sync) = prev_sync {
                    prev_sync.sync();
                }

                prev_sync = Some(ctx.upload(tensor, &data).map_err(|err| Error {
                    msg: err.msg,
                    kind: err.kind,
                    ctx: SerialTensorError::Unrelated,
                })?);

                file.seek_relative(4).map_err(|_| Error {
                    msg: "unexpected EOF",
                    kind: ErrorKind::SerializationError,
                    ctx: SerialTensorError::FailedFileIo,
                })?;
            }

            Ok(())
        }
    };
}

impl_load!(
    load_tensors_v1,
    load_into_tensors_v1,
    init_tensor_f64,
    f64,
    u64
);
impl_load!(
    load_tensors_v1m,
    load_into_tensors_v1m,
    init_tensor_f32,
    f32,
    u32
);
