# `fused_gpu`

Advanced graph-based GPU compiler for linear algebra and AI/ML/DL.

## Usage

Should only be used when implementing custom operations. If creating the model, link `ferrograd-nn`. `ferrograd` does this automatically and has stronger correctness guaraantees.

Optimizer:

```rust
pub const STOCHASTIC_GRADIENT_DESCENT: OptimType = OptimType {
    lower: |kernel, dtype, lr, weight_param, grad_param, gid, _, _| {
        let grad = kernel.raw.def_var(
            dtype,
            ValueState::Immut,
            Some(Op::ParamLoad {
                param: grad_param,
                index: gid,
            }),
        );

        let scaled_grad =
            kernel
                .raw
                .def_var(dtype, ValueState::Immut, Some(Op::Mul { a: grad, b: lr }));

        let zero = kernel
            .raw
            .def_var(dtype, ValueState::Inline, Some(dtype.constant_float(0.0)?));

        kernel.raw.param_sub(weight_param, gid, scaled_grad);
        kernel.raw.param_store(grad_param, gid, zero);

        Ok(())
    },
};

pub const STOCHASTIC_GRADIENT_DESCENT_STATE: OptimState<0> = OptimState { shapes: [] };
```

Loss:

```rust
pub const MEAN_SQUARED_ERROR: LossType = LossType {
    lower: |kernel, dtype, pred, target, _, _, _, _| {
        let diff = kernel.raw.def_var(
            dtype,
            ValueState::Immut,
            Some(Op::Sub { a: pred, b: target }),
        );

        let loss_val =
            kernel
                .raw
                .def_var(dtype, ValueState::Immut, Some(Op::Mul { a: diff, b: diff }));

        let two = kernel
            .raw
            .def_var(dtype, ValueState::Inline, Some(dtype.constant_float(2.0)?));

        let grad_val =
            kernel
                .raw
                .def_var(dtype, ValueState::Immut, Some(Op::Mul { a: two, b: diff }));

        Ok((loss_val, grad_val))
    },
};
```

Graph Op:

```rust
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
```

## Backends

Only supports WGPU/WGSL backends. CUDA and ROCm backends planned before `v1.0.0`. Other backends (e.g. CPU) are unlikely. Custom backends are fully supported.
