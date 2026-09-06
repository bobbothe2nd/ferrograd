use std::ops::{Index, IndexMut};

use crate::{
    backend::SavedNodes,
    dispatch::{DType, LossType, OptimType},
    id::{EdgeId, NodeId},
    tensor::{bf16, f16},
};
use briny::raw::alloc::cast_vec;
use fused_gpu::{
    dispatch::{
        CompilationOptions,
        backend::{Graph as InnerGraph, Metadata, Node, kernel::KernelGroup},
    },
    errors::{Error, GraphErrorContext},
};

pub struct Graph<'a>(pub(crate) InnerGraph<'a>);

impl<'a> Index<NodeId> for Graph<'a> {
    type Output = Node<'a>;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.0.nodes[index.0]
    }
}

impl<'a> IndexMut<NodeId> for Graph<'a> {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.0.nodes[index.0]
    }
}

impl<'a> Graph<'a> {
    #[inline]
    #[must_use]
    pub const fn new(loss: LossType, optim: OptimType) -> Self {
        Self(InnerGraph::new(loss, optim))
    }

    #[inline]
    #[must_use]
    pub fn get_adjacent(&self, node: NodeId, user: NodeId) -> Vec<NodeId> {
        cast_vec(self.0.get_adjacent(node.0, user.0))
    }

    #[inline]
    pub fn get_edge(&self, node: NodeId, user: NodeId) -> Option<EdgeId> {
        Some(EdgeId(self.0.get_edge(node.0, user.0)?))
    }

    #[inline]
    #[must_use]
    pub fn is_edge(&self, node: NodeId, user: NodeId, edge: EdgeId) -> bool {
        self.0.is_edge(node.0, user.0, edge.0)
    }

    #[inline]
    #[must_use]
    pub fn is_lhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        self.0.is_lhs_edge(node.0, user.0)
    }

    #[inline]
    #[must_use]
    pub fn is_rhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        self.0.is_rhs_edge(node.0, user.0)
    }

    #[inline]
    #[must_use]
    pub fn compute_saved_nodes(&self) -> SavedNodes {
        SavedNodes(self.0.compute_saved_nodes())
    }

    #[inline]
    pub fn validate(&self, meta: Metadata) -> Result<(), Vec<Error<GraphErrorContext<'a>>>> {
        self.0.validate(meta)
    }

    #[inline]
    pub fn lower(
        &'a mut self,
        meta: Metadata,
        options: &CompilationOptions,
        saved: &SavedNodes,
    ) -> Result<KernelGroup<'a>, Error<GraphErrorContext<'a>>> {
        self.0.topo_sort()?;
        self.0.rebuild_outputs();

        self.0
            .lower(meta, options, &saved.0)
            .map_err(move |e| e.into())
    }
}

impl<'a> Graph<'a> {
    #[inline]
    #[must_use]
    pub const fn define_ops<'b>(&'b mut self) -> DefineOps<'b, 'a> {
        DefineOps { graph: self }
    }
}

pub struct DefineOps<'a, 'b> {
    graph: &'a mut Graph<'b>,
}

impl DefineOps<'_, '_> {
    pub fn input(&mut self, shape: &[usize], dtype: DType) -> NodeId {
        NodeId(self.graph.0.input(shape, dtype))
    }

    pub fn constant_f16(&mut self, data: f16) -> NodeId {
        NodeId(self.graph.0.constant_f16(data))
    }

    pub fn constant_bf16(&mut self, data: bf16) -> NodeId {
        NodeId(self.graph.0.constant_bf16(data))
    }

    pub fn constant_f32(&mut self, data: f32) -> NodeId {
        NodeId(self.graph.0.constant_f32(data))
    }

    pub fn constant_u32(&mut self, data: u32) -> NodeId {
        NodeId(self.graph.0.constant_u32(data))
    }

    pub fn constant_i32(&mut self, data: i32) -> NodeId {
        NodeId(self.graph.0.constant_i32(data))
    }
}

macro_rules! impl_op {
    ($op:ident, $($node:ident),*) => {
        pub fn $op(&mut self, $($node: NodeId),*) -> NodeId {
            NodeId(ferrograd_nn::op::$op(&mut self.graph.0, $($node.0),*))
        }
    };
}

impl DefineOps<'_, '_> {
    impl_op!(mul, a, b);
    impl_op!(add, a, b);
    impl_op!(div, a, b);
    impl_op!(sub, a, b);
    impl_op!(matmul, a, b);
    impl_op!(abs, x);
    impl_op!(exp, x);
    impl_op!(gelu, x);
    impl_op!(relu, x);
    impl_op!(elu, x);
    impl_op!(log, x);
    impl_op!(neg, x);
    impl_op!(tanh, x);
    impl_op!(softmax, x);
}
