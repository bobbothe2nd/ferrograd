use briny::traits::{Pod, StableLayout};
use fused_gpu::dispatch::backend;

macro_rules! id {
    ($name:ident($inner:ty)) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) $inner);

        unsafe impl StableLayout for $name {}
        unsafe impl Pod for $name {}

        impl $name {
            #[must_use]
            #[inline(always)]
            pub const fn to_bits(self) -> $inner {
                self.0
            }

            #[must_use]
            #[inline(always)]
            pub const fn from_bits(bits: $inner) -> Self {
                Self(bits)
            }
        }
    };
}

id!(EdgeId(usize));
id!(NodeId(backend::NodeId));
id!(ParamId(backend::ParamId));
id!(SharedId(backend::SharedId));
id!(MetaId(backend::MetaId));
id!(ValueId(backend::ValueId));
