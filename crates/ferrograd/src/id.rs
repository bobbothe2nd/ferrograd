use briny::traits::{Layout, StableLayout};
use fused_gpu::dispatch::backend;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

unsafe impl StableLayout for EdgeId {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) backend::NodeId);

unsafe impl StableLayout for NodeId {}
unsafe impl Layout<backend::NodeId> for NodeId {}
unsafe impl Layout<NodeId> for backend::NodeId {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub(crate) backend::ParamId);

unsafe impl StableLayout for ParamId {}
unsafe impl Layout<backend::ParamId> for ParamId {}
unsafe impl Layout<ParamId> for backend::ParamId {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedId(pub(crate) backend::SharedId);

unsafe impl StableLayout for SharedId {}
unsafe impl Layout<backend::SharedId> for SharedId {}
unsafe impl Layout<SharedId> for backend::SharedId {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetaId(pub(crate) backend::MetaId);

unsafe impl StableLayout for MetaId {}
unsafe impl Layout<backend::MetaId> for MetaId {}
unsafe impl Layout<MetaId> for backend::MetaId {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub(crate) backend::ValueId);

unsafe impl StableLayout for ValueId {}
unsafe impl Layout<backend::ValueId> for ValueId {}
unsafe impl Layout<ValueId> for backend::ValueId {}
