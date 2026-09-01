use std::{fs::File, io::{BufReader, BufWriter, Read, Write}};

use briny::{raw::cast::{slice_to_bytes, slice_to_bytes_mut}, traits::Pod};

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
        file.write(&(t.rank()).to_le_bytes()[..U]).map_err(|_| Error {
            msg: "failed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        });

        let shape_u64 = t.shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        file.write(slice_to_bytes(&shape_u64));

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

        file.write(slice_to_bytes(&buf));
    }

    Ok(())
}

macro_rules! impl_load {
    ($name:ident, $name2:ident, $init:ident, $float:ident, $unsigned:ident) => {
        #[inline(always)]
        pub fn $name<B: GpuBackend>(
            file: &mut BufReader<File>,
            ctx: GpuContext<B>,
        ) -> Result<Vec<Tensor<B>>, Error<SerialTensorError>> {
            let mut len = [0];
            file.read_exact(&mut len);
            let len = u8::from_le_bytes(len) as usize;

            let mut tensors = Vec::with_capacity(len);

            for _ in 0..len {
                let mut rank = [0; size_of::<$unsigned>()];
                file.read_exact(&mut rank);
                let rank = <$unsigned>::from_le_bytes(rank);

                let mut shape = Vec::with_capacity(rank as usize);

                for _ in 0..rank {
                    let mut shape_u64 = [0; size_of::<$unsigned>()];
                    file.read_exact(&mut shape_u64);
                    shape.push(<$unsigned>::from_le_bytes(shape_u64) as u32);
                }

                let len = shape.iter().product::<u32>();

                let mut data = vec![$float::default(); len as usize];

                file.read_exact(slice_to_bytes_mut(&mut data)).map_err(|_| Error {
                    msg: "unexpected EOF",
                    kind: ErrorKind::SerializationError,
                    ctx: SerialTensorError::FailedFileIo,
                });

                tensors.push(ctx.$init(shape, &data));
            }

            Ok(tensors)
        }

        #[inline(always)]
        pub fn $name2<B: GpuBackend>(
            file: &mut BufReader<File>,
            ctx: GpuContext<B>,
            tensors: &mut [Tensor<B>],
        ) -> Result<(), Error<SerialTensorError>> {
            let mut len = [0];
            file.read_exact(&mut len);
            let len = u8::from_le_bytes(len) as usize;

            for tensor in tensors.iter_mut().take(len) {
                let mut rank = [0; size_of::<$unsigned>()];
                file.read_exact(&mut rank);
                let rank = <$unsigned>::from_le_bytes(rank);

                let mut shape = Vec::with_capacity(rank as usize);

                for _ in 0..rank {
                    let mut shape_u64 = [0; size_of::<$unsigned>()];
                    file.read_exact(&mut shape_u64);
                    shape.push(<$unsigned>::from_le_bytes(shape_u64) as u32);
                }

                let len = shape.iter().product::<u32>();

                let mut data = vec![$float::default(); len as usize];

                file.read_exact(slice_to_bytes_mut(&mut data)).map_err(|_| Error {
                    msg: "unexpected EOF",
                    kind: ErrorKind::SerializationError,
                    ctx: SerialTensorError::FailedFileIo,
                });

                tensor.shape = shape;

                ctx.upload(tensor, &data);
            }

            Ok(())
        }
    };
}

impl_load!(load_tensors_v0, load_into_tensors_v0, init_tensor_f64, f64, u64);

use half::{bf16, f16};

impl_load!(load_tensors_v2_bf16, load_into_tensors_v2_bf16, init_tensor_bf16, bf16, u32);
impl_load!(load_tensors_v2_f16, load_into_tensors_v2_f16, init_tensor_f16, f16, u32);
impl_load!(load_tensors_v2_f32, load_into_tensors_v2_f32, init_tensor_f32, f32, u32);
