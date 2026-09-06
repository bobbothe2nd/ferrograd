use fused_gpu::{
    dispatch::{
        CompilationOptions,
        backend::{
            DispatchOptions, Graph, GraphOp, MetaId, Node, NodeId, Op, Param, ParamId, ValueId,
            ValueState,
            kernel::{LinkedKernel, NodeInput, SaveIndicator},
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use std::{format, vec, vec::Vec};

fn valid_binary<'a>(
    node_id: NodeId,
    node: &Node<'a>,
    graph: &Graph<'a>,
    errors: &mut Vec<Error<GraphErrorContext<'a>>>,
) {
    if node.inputs.len() != 2 {
        errors.push(Error {
            msg: "binary operation has invalid input count",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::InvalidInputs {
                node: node_id,
                arity: 2,
                args: node.inputs.len(),
            },
        });
    }

    let a = node.inputs[0];
    let b = node.inputs[1];

    let a_node = &graph.nodes[a];
    let b_node = &graph.nodes[b];

    let a_shape = &a_node.shape;
    let b_shape = &b_node.shape;

    if a_node.op.is_const() && b_node.op.is_const() {
        errors.push(Error {
            msg: "binary operation has only constant inputs",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::CannotInferShape {
                node: node_id,
                all_hand_sides: vec![a_shape.clone(), b_shape.clone()],
                op: node.op,
            },
        });
    }

    if !(a_node.op.is_const() || b_node.op.is_const()) && a_shape != b_shape {
        errors.push(Error {
            msg: "binary operation has different shaped inputs",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::ShapeMismatch {
                node: node_id,
                all_hand_sides: vec![a_shape.clone(), b_shape.clone()],
                op: node.op,
            },
        });
    }
}

fn save_mul_div(_node_id: NodeId, node: &Node, graph: &Graph, saved: &mut [SaveIndicator]) {
    for &inp in &node.inputs {
        if !graph.nodes[inp].inputs.is_empty() {
            saved[inp] |= SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_BACKWARD;
        }
    }
}

pub fn add<'a>(graph: &mut Graph<'a>, a: NodeId, b: NodeId) -> NodeId {
    graph.push_node(Node {
        op: GraphOp::Custom {
            lower: |eval_node: fn(
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
                    idx: ValueId,
                    local_row: ValueId,
                    local_col: ValueId,
                    shared_size: u32,
                    tile_size: ValueId,
                    params: &[Param],
                    stable_iteration_space: &mut bool,
                    options: &CompilationOptions| {
                let mut deepest = Vec::new();
                let dtype = graph.nodes[node_id].dtype;

                match backwardness {
                    None => {
                        let a = graph.nodes[node_id].inputs[0];
                        let b = graph.nodes[node_id].inputs[1];

                        let a_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut a_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(a),
                            a_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let b_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut b_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(b),
                            b_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut a_deep);
                        deepest.append(&mut b_deep);

                        kernel
                            .raw
                            .overwrite_var(out, Op::Add { a: a_val, b: b_val });
                    }

                    Some(0 | 1) => {
                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            out,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut deep);
                    }

                    _ => {
                        return Err(Error {
                            msg: "backwardness must be restricted to input count",
                            kind: ErrorKind::UnresolvedInput,
                            ctx: (),
                        });
                    }
                }

                Ok(deepest)
            },
            display: |inputs| format!("{:?} + {:?}", inputs[0], inputs[1]),
            save: |_, _, _, _| {},
            valid_shape: valid_binary,
            arity: 2,
            need_dims: false,
            stable_iter: true,
            auto_save: true,
            computes_gid: true,
            prefer_separate: false,
            valid_dispatch: DispatchOptions::Any,
        },
        inputs: vec![a, b],
        outputs: Vec::new(),
        shape: get_shape(a, b, graph),
        dtype: graph.nodes[a].dtype,
    })
}

pub fn mul<'a>(graph: &mut Graph<'a>, a: NodeId, b: NodeId) -> NodeId {
    graph.push_node(Node {
        op: GraphOp::Custom {
            lower: |eval_node: fn(
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
                    idx: ValueId,
                    local_row: ValueId,
                    local_col: ValueId,
                    shared_size: u32,
                    tile_size: ValueId,
                    params: &[Param],
                    stable_iteration_space: &mut bool,
                    options: &CompilationOptions| {
                let mut deepest = Vec::new();
                let dtype = graph.nodes[node_id].dtype;

                match backwardness {
                    None => {
                        let a = graph.nodes[node_id].inputs[0];
                        let b = graph.nodes[node_id].inputs[1];

                        let a_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut a_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(a),
                            a_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let b_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut b_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(b),
                            b_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut a_deep);
                        deepest.append(&mut b_deep);

                        kernel
                            .raw
                            .overwrite_var(out, Op::Mul { a: a_val, b: b_val });
                    }

                    Some(0) => {
                        let g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Mut,
                            Some(dtype.constant_float(0.0)?),
                        );

                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            g_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let user = graph.nodes[node_id].inputs[1];

                        let saved =
                            read_saved(kernel, idx, saved_params[user], graph, user, params)?;

                        kernel.raw.accum_var(out, Op::Mul { a: g_val, b: saved });

                        deepest.append(&mut deep);
                    }

                    Some(1) => {
                        let g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Mut,
                            Some(dtype.constant_float(0.0)?),
                        );

                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            g_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let user = graph.nodes[node_id].inputs[0];

                        let saved =
                            read_saved(kernel, idx, saved_params[user], graph, user, params)?;

                        kernel.raw.accum_var(out, Op::Mul { a: g_val, b: saved });

                        deepest.append(&mut deep);
                    }

                    _ => {
                        return Err(Error {
                            msg: "backwardness must be restricted to input count",
                            kind: ErrorKind::UnresolvedInput,
                            ctx: (),
                        });
                    }
                }

                Ok(deepest)
            },
            display: |inputs| format!("{:?} * {:?}", inputs[0], inputs[1]),
            save: save_mul_div,
            valid_shape: valid_binary,
            arity: 2,
            need_dims: false,
            stable_iter: true,
            auto_save: true,
            computes_gid: true,
            prefer_separate: false,
            valid_dispatch: DispatchOptions::Any,
        },
        inputs: vec![a, b],
        outputs: Vec::new(),
        shape: get_shape(a, b, graph),
        dtype: graph.nodes[a].dtype,
    })
}

pub fn sub<'a>(graph: &mut Graph<'a>, a: NodeId, b: NodeId) -> NodeId {
    graph.push_node(Node {
        op: GraphOp::Custom {
            lower: |eval_node: fn(
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
                    idx: ValueId,
                    local_row: ValueId,
                    local_col: ValueId,
                    shared_size: u32,
                    tile_size: ValueId,
                    params: &[Param],
                    stable_iteration_space: &mut bool,
                    options: &CompilationOptions| {
                let mut deepest = Vec::new();
                let dtype = graph.nodes[node_id].dtype;

                match backwardness {
                    None => {
                        let a = graph.nodes[node_id].inputs[0];
                        let b = graph.nodes[node_id].inputs[1];

                        let a_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut a_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(a),
                            a_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let b_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut b_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(b),
                            b_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut a_deep);
                        deepest.append(&mut b_deep);

                        kernel
                            .raw
                            .overwrite_var(out, Op::Sub { a: a_val, b: b_val });
                    }

                    Some(0) => {
                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            out,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut deep);
                    }

                    Some(1) => {
                        let g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Mut,
                            Some(dtype.constant_float(0.0)?),
                        );

                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            g_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        kernel.raw.accum_var(out, Op::Neg { x: g_val });

                        deepest.append(&mut deep);
                    }

                    _ => {
                        return Err(Error {
                            msg: "backwardness must be restricted to input count",
                            kind: ErrorKind::UnresolvedInput,
                            ctx: (),
                        });
                    }
                }

                Ok(deepest)
            },
            display: |inputs| format!("{:?} - {:?}", inputs[0], inputs[1]),
            save: |_, _, _, _| {},
            valid_shape: valid_binary,
            arity: 2,
            need_dims: false,
            stable_iter: true,
            auto_save: true,
            computes_gid: true,
            prefer_separate: false,
            valid_dispatch: DispatchOptions::Any,
        },
        inputs: vec![a, b],
        outputs: Vec::new(),
        shape: get_shape(a, b, graph),
        dtype: graph.nodes[a].dtype,
    })
}

pub fn div<'a>(graph: &mut Graph<'a>, a: NodeId, b: NodeId) -> NodeId {
    graph.push_node(Node {
        op: GraphOp::Custom {
            lower: |eval_node: fn(
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
                    idx: ValueId,
                    local_row: ValueId,
                    local_col: ValueId,
                    shared_size: u32,
                    tile_size: ValueId,
                    params: &[Param],
                    stable_iteration_space: &mut bool,
                    options: &CompilationOptions| {
                let mut deepest = Vec::new();
                let dtype = graph.nodes[node_id].dtype;

                match backwardness {
                    None => {
                        let a = graph.nodes[node_id].inputs[0];
                        let b = graph.nodes[node_id].inputs[1];

                        let a_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut a_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(a),
                            a_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let b_val = kernel.raw.def_var(dtype, ValueState::Mut, None);

                        let mut b_deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(b),
                            b_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        deepest.append(&mut a_deep);
                        deepest.append(&mut b_deep);

                        kernel
                            .raw
                            .overwrite_var(out, Op::Div { a: a_val, b: b_val });
                    }

                    Some(0) => {
                        let g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Mut,
                            Some(dtype.constant_float(0.0)?),
                        );

                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            g_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let user = graph.nodes[node_id].inputs[1];

                        let saved =
                            read_saved(kernel, idx, saved_params[user], graph, user, params)?;

                        kernel.raw.accum_var(out, Op::Div { a: g_val, b: saved });

                        deepest.append(&mut deep);
                    }

                    Some(1) => {
                        let g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Mut,
                            Some(dtype.constant_float(0.0)?),
                        );

                        let mut deep = eval_node(
                            root,
                            input,
                            &NodeInput::Node(node_id),
                            g_val,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            params,
                            stable_iteration_space,
                            options,
                        )?;

                        let a = graph.nodes[node_id].inputs[0];
                        let b = graph.nodes[node_id].inputs[1];

                        let a_val = read_saved(kernel, idx, saved_params[a], graph, a, params)?;
                        let b_val = read_saved(kernel, idx, saved_params[b], graph, b, params)?;

                        let bb = kernel.raw.def_var(
                            dtype,
                            ValueState::Inline,
                            Some(Op::Mul { a: b_val, b: b_val }),
                        );

                        let neg_g_val = kernel.raw.def_var(
                            dtype,
                            ValueState::Inline,
                            Some(Op::Neg { x: g_val }),
                        );

                        let a_div_bb = kernel.raw.def_var(
                            dtype,
                            ValueState::Inline,
                            Some(Op::Div { a: a_val, b: bb }),
                        );

                        kernel.raw.accum_var(
                            out,
                            Op::Mul {
                                a: neg_g_val,
                                b: a_div_bb,
                            },
                        );

                        deepest.append(&mut deep);
                    }

                    _ => {
                        return Err(Error {
                            msg: "backwardness must be restricted to input count",
                            kind: ErrorKind::UnresolvedInput,
                            ctx: (),
                        });
                    }
                }

                Ok(deepest)
            },
            display: |inputs| format!("{:?} / {:?}", inputs[0], inputs[1]),
            save: save_mul_div,
            valid_shape: valid_binary,
            arity: 2,
            need_dims: false,
            stable_iter: true,
            auto_save: true,
            computes_gid: true,
            prefer_separate: false,
            valid_dispatch: DispatchOptions::Any,
        },
        inputs: vec![a, b],
        outputs: Vec::new(),
        shape: get_shape(a, b, graph),
        dtype: graph.nodes[a].dtype,
    })
}

fn get_shape(a: NodeId, b: NodeId, graph: &Graph) -> Vec<MetaId> {
    let a_node = &graph.nodes[a];
    let b_node = &graph.nodes[b];

    if a_node.op.is_const() {
        b_node.shape.clone()
    } else {
        a_node.shape.clone()
    }
}

fn read_saved(
    kernel: &mut LinkedKernel<'_>,
    index: ValueId,
    param: Option<ParamId>,
    graph: &Graph,
    node_id: NodeId,
    params: &[Param],
) -> Result<ValueId, Error> {
    if let Some(pid) = param {
        kernel.register_param(pid);

        Ok(kernel.raw.def_var(
            params[pid].dtype,
            ValueState::Immut,
            Some(Op::ParamLoad { param: pid, index }),
        ))
    } else {
        let dtype = graph.nodes[node_id].dtype;

        Ok(kernel
            .raw
            .def_var(dtype, ValueState::Inline, Some(dtype.constant_float(0.0)?)))
    }
}
