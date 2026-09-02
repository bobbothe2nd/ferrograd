use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use briny::{
    raw::cast::{slice_to_bytes, slice_to_bytes_mut},
    traits::Pod,
};

use crate::{
    dispatch::{GpuBackend, GpuContext},
    errors::{Error, ErrorKind},
    io::{
        SerialTensorError,
        headers::{BPAT_MAGIC_V0, BPAT_MAGIC_V2_BF16, BPAT_MAGIC_V2_F16, BPAT_MAGIC_V2_F32},
    },
    tensor::Tensor,
};

/// Currently requires tensors to be `f64`.
pub fn save_tensors_v0<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<8, P, f64, B>(path, ctx, tensors, &BPAT_MAGIC_V0)
}

/// Currently requires tensors to be `f32`.
pub fn save_tensors_v2_f32<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, P, f32, B>(path, ctx, tensors, &BPAT_MAGIC_V2_F32)
}

/// Currently requires tensors to be `f16`.
pub fn save_tensors_v2_f16<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, P, half::f16, B>(path, ctx, tensors, &BPAT_MAGIC_V2_F16)
}

/// Currently requires tensors to be `bf16`.
pub fn save_tensors_v2_bf16<P: AsRef<Path>, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
) -> Result<(), Error<SerialTensorError>> {
    save_tensors::<4, P, half::bf16, B>(path, ctx, tensors, &BPAT_MAGIC_V2_BF16)
}

#[inline]
fn save_tensors<const U: usize, P: AsRef<Path>, F: Default + Pod + Clone, B: GpuBackend>(
    path: P,
    ctx: &GpuContext<B>,
    tensors: &[Tensor<B>],
    magic: &[u8],
) -> Result<(), Error<SerialTensorError>> {
    let mut file = BufWriter::new(File::create(path).map_err(|_| Error {
        msg: "file not found",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::InvalidPath,
    })?);

    file.write(magic).map_err(|_| Error {
        msg: "failed to write to file",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::FailedFileIo,
    })?;
    file.write(&[tensors.len().try_into().map_err(|_| Error {
        msg: "exceeds max tensors of 255",
        kind: ErrorKind::InvalidArgument,
        ctx: SerialTensorError::Unrelated,
    })?])
    .map_err(|_| Error {
        msg: "failed to write to file",
        kind: ErrorKind::SerializationError,
        ctx: SerialTensorError::FailedFileIo,
    })?;

    for t in tensors {
        file.write(&(t.rank()).to_le_bytes()[..U])
            .map_err(|_| Error {
                msg: "failed to write to file",
                kind: ErrorKind::SerializationError,
                ctx: SerialTensorError::FailedFileIo,
            })?;

        let shape_u64 = t
            .shape
            .iter()
            .map(|x| (*x as u64).to_le_bytes()[..U].try_into().unwrap_or([0; U]))
            .collect::<Vec<[u8; U]>>();
        file.write(slice_to_bytes(&shape_u64)).map_err(|_| Error {
            msg: "failed to write to file",
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

        file.write(&buf).map_err(|_| Error {
            msg: "failed to write to file",
            kind: ErrorKind::SerializationError,
            ctx: SerialTensorError::FailedFileIo,
        })?;
    }

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

                // cant allocate this, use empty init and buffer file io using multiple uploads
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

                prev_sync = Some(ctx.upload(tensor, &data, 0).map_err(|err| Error {
                    msg: err.msg,
                    kind: err.kind,
                    ctx: SerialTensorError::Unrelated,
                })?);
            }

            Ok(())
        }
    };
}

impl_load!(
    load_tensors_v0,
    load_into_tensors_v0,
    init_tensor_f64,
    f64,
    u64
);

use half::{bf16, f16};

impl_load!(
    load_tensors_v2_bf16,
    load_into_tensors_v2_bf16,
    init_tensor_bf16,
    bf16,
    u32
);
impl_load!(
    load_tensors_v2_f16,
    load_into_tensors_v2_f16,
    init_tensor_f16,
    f16,
    u32
);
impl_load!(
    load_tensors_v2_f32,
    load_into_tensors_v2_f32,
    init_tensor_f32,
    f32,
    u32
);

#[cfg(all(test, feature = "wgsl"))]
mod tests {
    use std::{
        fs::{File, remove_file},
        io::{BufReader, Read},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        dispatch::{GpuBackend, GpuContext},
        io::headers::BPAT_MAGIC_V0,
        tensor::{Tensor, bf16, f16},
    };

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!(
            "briny_serialization_{name}_{}_{}.bpat",
            std::process::id(),
            nanos
        ))
    }

    fn open_payload(path: &str) -> BufReader<File> {
        let mut file = BufReader::new(File::open(path).unwrap());

        let mut file_start = [0; 4];
        file.read_exact(&mut file_start).unwrap();

        if file_start == BPAT_MAGIC_V0 {
            return file;
        }

        file.seek_relative(4).unwrap();

        file
    }

    fn assert_tensor_data_f32<B: GpuBackend>(
        ctx: &GpuContext<B>,
        tensors: &[Tensor<B>],
        expected: &[(&[u32], &[f32])],
    ) {
        assert_eq!(tensors.len(), expected.len());

        for (tensor, (shape, expected_data)) in tensors.iter().zip(expected) {
            assert_eq!(&tensor.shape[..], *shape);

            let mut actual = vec![0u8; expected_data.len() * size_of::<f32>()];
            ctx.download(tensor, &mut actual).unwrap();

            let actual = briny::raw::cast::slice_from_bytes::<f32>(&actual).unwrap();

            assert_eq!(actual, *expected_data);
        }
    }

    fn assert_tensor_data_f64<B: GpuBackend>(
        ctx: &GpuContext<B>,
        tensors: &[Tensor<B>],
        expected: &[(&[u32], &[f64])],
    ) {
        assert_eq!(tensors.len(), expected.len());

        for (tensor, (shape, expected_data)) in tensors.iter().zip(expected) {
            assert_eq!(&tensor.shape[..], *shape);

            let mut actual = vec![0u8; expected_data.len() * size_of::<f64>()];
            ctx.download(tensor, &mut actual).unwrap();

            let actual = briny::raw::cast::slice_from_bytes::<f64>(&actual).unwrap();

            assert_eq!(actual, *expected_data);
        }
    }

    fn assert_tensor_data_f16<B: GpuBackend>(
        ctx: &GpuContext<B>,
        tensors: &[Tensor<B>],
        expected: &[(&[u32], &[f16])],
    ) {
        assert_eq!(tensors.len(), expected.len());

        for (tensor, (shape, expected_data)) in tensors.iter().zip(expected) {
            assert_eq!(&tensor.shape[..], *shape);

            let mut actual = vec![0u8; expected_data.len() * size_of::<f16>()];
            ctx.download(tensor, &mut actual).unwrap();

            let actual = briny::raw::cast::slice_from_bytes::<f16>(&actual).unwrap();

            assert_eq!(actual, *expected_data);
        }
    }

    fn assert_tensor_data_bf16<B: GpuBackend>(
        ctx: &GpuContext<B>,
        tensors: &[Tensor<B>],
        expected: &[(&[u32], &[bf16])],
    ) {
        assert_eq!(tensors.len(), expected.len());

        for (tensor, (shape, expected_data)) in tensors.iter().zip(expected) {
            assert_eq!(&tensor.shape[..], *shape);

            let mut actual = vec![0u8; expected_data.len() * size_of::<bf16>()];
            ctx.download(tensor, &mut actual).unwrap();

            let actual = briny::raw::cast::slice_from_bytes::<bf16>(&actual).unwrap();

            assert_eq!(actual, *expected_data);
        }
    }

    #[test]
    fn roundtrip_v0_f64() {
        let ctx = GpuContext::new().unwrap();

        let shape1 = [2, 3];
        let data1 = [0.0, 1.0, -2.5, 3.25, 100.0, -999.125];

        let shape2 = [4];
        let data2 = [f64::MIN, -1.0, f64::MAX, 42.5];

        let tensors = vec![
            ctx.init_tensor_f64(shape1.to_vec(), &data1),
            ctx.init_tensor_f64(shape2.to_vec(), &data2),
        ];

        let path = temp_path("v0_f64");
        let path_str = path.to_str().unwrap();

        save_tensors_v0(path_str, &ctx, &tensors).unwrap();

        let mut file = open_payload(path_str);
        let loaded = load_tensors_v0(&mut file, &ctx).unwrap();

        assert_tensor_data_f64(&ctx, &loaded, &[(&shape1, &data1), (&shape2, &data2)]);

        remove_file(path).unwrap();
    }

    #[test]
    fn roundtrip_v2_f32() {
        let ctx = GpuContext::new().unwrap();

        let shape1 = [2, 3];
        let data1 = [0.0, 1.0, -2.5, 3.25, 100.0, -999.125];

        let shape2 = [4];
        let data2 = [f32::MIN, -1.0, f32::MAX, 42.5];

        let tensors = vec![
            ctx.init_tensor_f32(shape1.to_vec(), &data1),
            ctx.init_tensor_f32(shape2.to_vec(), &data2),
        ];

        let path = temp_path("v2_f32");
        let path_str = path.to_str().unwrap();

        save_tensors_v2_f32(path_str, &ctx, &tensors).unwrap();

        let mut file = open_payload(path_str);
        let loaded = load_tensors_v2_f32(&mut file, &ctx).unwrap();

        assert_tensor_data_f32(&ctx, &loaded, &[(&shape1, &data1), (&shape2, &data2)]);

        remove_file(path).unwrap();
    }

    #[test]
    fn roundtrip_v2_f16() {
        let ctx = GpuContext::new().unwrap();

        let shape1 = [2, 3];
        let data1 = [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-2.5),
            f16::from_f32(3.25),
            f16::from_f32(100.0),
            f16::from_f32(-999.125),
        ];

        let shape2 = [4];
        let data2 = [f16::MIN, f16::from_f32(-1.0), f16::MAX, f16::from_f32(42.5)];

        let tensors = vec![
            ctx.init_tensor_f16(shape1.to_vec(), &data1),
            ctx.init_tensor_f16(shape2.to_vec(), &data2),
        ];

        let path = temp_path("v2_f16");
        let path_str = path.to_str().unwrap();

        save_tensors_v2_f16(path_str, &ctx, &tensors).unwrap();

        let mut file = open_payload(path_str);
        let loaded = load_tensors_v2_f16(&mut file, &ctx).unwrap();

        assert_tensor_data_f16(&ctx, &loaded, &[(&shape1, &data1), (&shape2, &data2)]);

        remove_file(path).unwrap();
    }

    #[test]
    fn roundtrip_v2_bf16() {
        let ctx = GpuContext::new().unwrap();

        let shape1 = [2, 3];
        let data1 = [
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(-2.5),
            bf16::from_f32(3.25),
            bf16::from_f32(100.0),
            bf16::from_f32(-999.125),
        ];

        let shape2 = [4];
        let data2 = [
            bf16::MIN,
            bf16::from_f32(-1.0),
            bf16::MAX,
            bf16::from_f32(42.5),
        ];

        let tensors = vec![
            ctx.init_tensor_bf16(shape1.to_vec(), &data1),
            ctx.init_tensor_bf16(shape2.to_vec(), &data2),
        ];

        let path = temp_path("v2_bf16");
        let path_str = path.to_str().unwrap();

        save_tensors_v2_bf16(path_str, &ctx, &tensors).unwrap();

        let mut file = open_payload(path_str);
        let loaded = load_tensors_v2_bf16(&mut file, &ctx).unwrap();

        assert_tensor_data_bf16(&ctx, &loaded, &[(&shape1, &data1), (&shape2, &data2)]);

        remove_file(path).unwrap();
    }
}
