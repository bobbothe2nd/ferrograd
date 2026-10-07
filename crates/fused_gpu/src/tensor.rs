//! Tensors are the core primitive of `fused_gpu`.
//!
//! For more information, see [`Tensor`].

use core::ops::Index;

use std::vec::Vec;

use crate::dispatch::{
    GpuBackend, GpuBuffer,
    backend::{Axis, MetaId},
};

#[inline]
#[must_use]
pub fn build_dims(shape: &[MetaId], meta: &[u32]) -> Vec<u32> {
    let mut dims = Vec::with_capacity(shape.len());

    for dim in shape {
        dims.push(meta[*dim]);
    }

    dims
}

#[inline]
#[must_use]
pub fn calc_grid(shape: &[u32], block: [u32; 3]) -> [u32; 3] {
    let out_rank = shape.len();

    if block[0] == block[1] {
        [
            shape[out_rank - 1].div_ceil(block[0]),
            shape[out_rank - 2].div_ceil(block[1]),
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    } else if block[0] == 1 {
        [
            1,
            shape[out_rank - 1],
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    } else {
        [
            shape[out_rank - 2],
            1,
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Block {
    pub dim3: [u32; 3],
    pub calc_grid: Option<fn(&[u32], [u32; 3]) -> [u32; 3]>,
}

impl Index<usize> for Block {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim3[index]
    }
}

impl Index<Axis> for Block {
    type Output = u32;

    fn index(&self, index: Axis) -> &Self::Output {
        &self.dim3[index as usize]
    }
}

impl Block {
    pub const ZERO3: Self = Self {
        dim3: [0; 3],
        calc_grid: None,
    };
    pub const UNIT: Self = Self {
        dim3: [1; 3],
        calc_grid: None,
    };

    #[inline]
    #[must_use]
    pub const fn new(dim3: [u32; 3]) -> Self {
        Self {
            dim3,
            calc_grid: None,
        }
    }

    #[inline]
    #[must_use]
    pub fn calc_grid(&self, shape: &[u32]) -> [u32; 3] {
        self.calc_grid.map_or_else(
            || calc_grid(shape, self.dim3),
            |calc_grid| (calc_grid)(shape, self.dim3),
        )
    }
}

pub trait ToBuffer<B: GpuBackend> {
    fn to_buffer(self) -> B::Buffer;
    fn as_buffer(&self) -> &B::Buffer;
}

/// Core compute storage primitive.
///
/// The `PartialEq` implementation on this only compares shapes.
///
/// Actively used in all computations, attaching a shape to the generic (and low-level) [`GpuBuffer`].
///
/// Can only be constructed through a [`GpuContext`](`crate::dispatch::GpuContext`) because it
/// requires a buffer to be allocated on the GPU first.
pub struct Tensor<B: GpuBackend = crate::dispatch::backend::GpuContext> {
    pub(crate) shape: Box<[u32]>,
    pub(crate) data: GpuBuffer<B>,
}

impl<B: GpuBackend> ToBuffer<B> for Tensor<B> {
    #[inline]
    fn to_buffer(self) -> <B as GpuBackend>::Buffer {
        self.data.inner
    }

    #[inline]
    fn as_buffer(&self) -> &<B as GpuBackend>::Buffer {
        &self.data.inner
    }
}

impl<B: GpuBackend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}

impl<B: GpuBackend> Eq for Tensor<B> {}

impl<B: GpuBackend> Tensor<B> {
    /// Calculates the required grid provided the block size of the kernel.
    ///
    /// Divides the last two dimensions
    #[inline]
    #[must_use]
    pub fn calc_grid(&self, block: [u32; 3]) -> [u32; 3] {
        calc_grid(&self.shape, block)
    }

    /// Returns the rank (length of shape) of the tensor.
    ///
    /// To get the full shape, use [`Tensor::dims`].
    #[inline]
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the dimensions of the tenosr as raw `u32`.
    ///
    /// This is a borrowed slice into the shape with length `self.rank()`.
    #[inline]
    pub fn dims(&self) -> &[u32] {
        &self.shape
    }

    /// Returns the exact length of the data.
    ///
    /// This will always be equal to `.dims().product()`, only it uses a much faster method by directly checking the length of the buffer.
    ///
    /// This is only correct for `f32` or `u32`. Other sizes (e.g. `f16`/`bf16`/`f64`) are not correct.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.data.size()
    }

    /// Checks if the length is equal to zero, or data has no elements.
    ///
    /// Logically equivalent to `self.len() == 0`.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.size_bytes() == 0
    }
}

pub use half::{bf16, f16};
