use fused_gpu::{
    dispatch::{
        CompilationOptions, TargetFlags,
        backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Node, NodeId, Op, Param, ParamId,
            SimpleDType, ValueId, ValueState,
            kernel::{LinkedKernel, NodeInput, SaveIndicator},
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use std::{format, vec, vec::Vec};

mod generic;
mod wmma;

pub fn lower_matmul_recursive<'a>(
    eval_node: impl Fn(
        NodeId,
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &'a Graph<'a>,
        &[Option<ParamId>],
        &[Option<ParamId>],
        &mut LinkedKernel<'a>,
        ValueId,
        ValueId,
        ValueId,
        ValueId,
        u32,
        ValueId,
        &[Param],
        &mut bool,
        &CompilationOptions,
    ) -> Result<Vec<NodeId>, Error>,
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    backwardness: Option<u8>,
    node_id: NodeId,
    graph: &'a Graph<'a>,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a>,
    base: ValueId,
    _idx: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    params: &[Param],
    stable_iteration_space: &mut bool,
    options: &CompilationOptions,
) -> Result<Vec<NodeId>, Error> {
    *stable_iteration_space = false;

    let row = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::Y }),
    );
    let col = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let node = &graph.nodes[node_id];

    let (a_node, b_node);

    if backwardness == Some(0) {
        let param = saved_params[node.inputs[1]].ok_or(Error {
            msg: "saved input parameter could not be materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        let shape = &graph.nodes[node.inputs[1]].shape;

        a_node = NodeInput::Node(node_id);
        b_node = NodeInput::Param { param, shape };
    } else if backwardness == Some(1) {
        let param = saved_params[node.inputs[0]].ok_or(Error {
            msg: "saved input parameter could not be materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        let shape = &graph.nodes[node.inputs[0]].shape;

        a_node = NodeInput::Param { param, shape };
        b_node = NodeInput::Node(node_id);
    } else {
        a_node = NodeInput::Node(node.inputs[0]);
        b_node = NodeInput::Node(node.inputs[1]);
    }

    let transpose_a = backwardness == Some(1);
    let transpose_b = backwardness == Some(0);

    let mut a_node_shape = match a_node {
        NodeInput::Node(node) => graph.nodes[node].shape.clone(),
        NodeInput::Param { param: _, shape } => shape.to_vec(),
    };

    if transpose_a {
        let len = a_node_shape.len();
        a_node_shape.swap(len - 1, len - 2);
    }

    let mut b_node_shape = match b_node {
        NodeInput::Node(node) => graph.nodes[node].shape.clone(),
        NodeInput::Param { param: _, shape } => shape.to_vec(),
    };

    if transpose_b {
        let len = b_node_shape.len();
        b_node_shape.swap(len - 1, len - 2);
    }

    let m = a_node_shape[a_node_shape.len() - 2];
    let n = b_node_shape[b_node_shape.len() - 1];
    let k = a_node_shape[a_node_shape.len() - 1];

    kernel.register_meta(m);
    kernel.register_meta(n);
    kernel.register_meta(k);

    let m = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::ReadMeta { param: 0, field: m }),
    );
    let n = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::ReadMeta { param: 0, field: n }),
    );
    let k = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::ReadMeta { param: 0, field: k }),
    );

    if backwardness.is_none() {
        kernel
            .raw
            .overwrite_var(out, node.dtype.constant_float(0.0)?);
    }

    let lower = if options.target.flags.contains(TargetFlags::LIN_ACC) {
        wmma::forward_matmul
    } else {
        generic::forward_matmul
    };

    lower(
        eval_node,
        node.dtype,
        root,
        input,
        resolved,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        &a_node,
        &b_node,
        graph,
        out,
        node_params,
        saved_params,
        kernel,
        base,
        row,
        col,
        local_row,
        local_col,
        shared_size,
        tile_size,
        params,
        stable_iteration_space,
        options,
    )
}

pub fn matmul(graph: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    fn save(node_id: NodeId, node: &Node, graph: &Graph, saved: &mut [SaveIndicator]) {
        saved[node_id] |= SaveIndicator::DEFINED_IN_FORWARD
            | SaveIndicator::USED_BY_FORWARD
            | SaveIndicator::DEFINED_IN_BACKWARD
            | SaveIndicator::USED_BY_BACKWARD;

        for &inp in &node.inputs {
            if !graph.nodes[inp].inputs.is_empty() {
                saved[inp] |= SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_FORWARD;
            }
        }
    }

    fn valid_shape<'a>(
        node_id: NodeId,
        node: &Node<'a>,
        graph: &Graph<'a>,
        errors: &mut Vec<Error<GraphErrorContext<'a>>>,
    ) {
        if node.inputs.len() != 2 {
            errors.push(Error {
                msg: "matrix multiplication has invalid input count",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::InvalidInputs {
                    node: node_id,
                    arity: 2,
                    args: node.inputs.len(),
                },
            });

            return;
        }

        let Some(a) = graph.nodes.get(node.inputs[0]) else {
            return;
        };
        let Some(b) = graph.nodes.get(node.inputs[1]) else {
            return;
        };

        let rank_a = a.shape.len();
        let rank_b = b.shape.len();

        if rank_a != rank_b {
            errors.push(Error {
                msg: "matrix multiplication has invalid input rank(s)",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::RankMismatch {
                    node: node_id,
                    all_hand_sides: vec![rank_a, rank_b],
                },
            });
        }

        let k1 = a.shape[a.shape.len() - 1];
        let k2 = b.shape[b.shape.len() - 2];

        if k1 != k2 {
            errors.push(Error {
                msg: "inner dimensions don't match for matrix multiplication",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::ShapeMismatch {
                    node: node_id,
                    all_hand_sides: vec![a.shape.clone(), b.shape.clone()],
                    op: node.op,
                },
            });
        }

        if a.op.is_const() && b.op.is_const() {
            errors.push(Error {
                msg: "binary operation has only constant inputs",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::CannotInferShape {
                    node: node_id,
                    all_hand_sides: vec![a.shape.clone(), b.shape.clone()],
                    op: node.op,
                },
            });
        }
    }

    let mut shape = graph.nodes[a].shape.clone();
    let last_idx = shape.len() - 1;
    shape[last_idx] = graph.nodes[b].shape[last_idx];

    graph.push_node(Node {
        op: GraphOp::Custom {
            lower: lower_matmul_recursive,
            arity: 2,
            need_dims: true,
            stable_iter: false,
            auto_save: true,
            computes_gid: true,
            prefer_separate: false,
            save,
            valid_shape,
            display: |inputs| format!("{:?} @ {:?}", inputs[0], inputs[1]),
            valid_dispatch: DispatchOptions::Any,
        },
        inputs: vec![a, b],
        outputs: Vec::new(),
        shape,
        dtype: graph.nodes[a].dtype,
    })
}
