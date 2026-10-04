use fused_gpu::{
    dispatch::{
        CompilationOptions,
        backend::{
            DType, Graph, NodeId, Op, Param, ParamId,
            SimpleDType, ValueId, ValueState,
            kernel::{LinkedKernel, NodeInput},
        },
    },
    errors::Error,
};
use std::vec::Vec;

pub fn forward_matmul<'a>(
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
    dtype: SimpleDType,
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &'a Graph<'a>,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a>,
    base: ValueId,
    row: ValueId,
    col: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    params: &[Param],
    stable_iteration_space: &mut bool,
    options: &CompilationOptions,
) -> Result<Vec<NodeId>, Error> {
    let mut deepest = Vec::new();

    let a_tile = kernel.raw.new_shared(dtype, shared_size);
    let b_tile = kernel.raw.new_shared(dtype, shared_size);

    let one = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::ConstU32 { value: 1 }),
    );

    let tk = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Mut,
        Some(Op::ConstU32 { value: 0 }),
    );

    let tile_row = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::Mul {
            a: local_row,
            b: tile_size,
        }),
    );
    let shared_idx = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Mut,
        Some(Op::Add {
            a: tile_row,
            b: local_col,
        }),
    );

    let mut a_deepest = Vec::new();
    let mut b_deepest = Vec::new();

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        let a_k = kernel.raw.def_var(
            DType::Simple(SimpleDType::U32),
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_col,
            }),
        );

        let b_k = kernel.raw.def_var(
            DType::Simple(SimpleDType::U32),
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_row,
            }),
        );

        let a_idx = if swap_a {
            let a_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul { a: a_k, b: m }),
            );
            let a_col = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: row }),
            );
            kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        } else {
            let a_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul { a: row, b: k }),
            );
            let a_col = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: a_k }),
            );
            kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        };

        let a_val = kernel.raw.def_var(
            DType::Simple(dtype),
            ValueState::Mut,
            Some(dtype.constant_float(0.0)?),
        );

        a_deepest = eval_node(
            root,
            input,
            a_node,
            a_val,
            resolved,
            graph,
            node_params,
            saved_params,
            kernel,
            a_idx,
            base,
            local_row,
            local_col,
            shared_size,
            tile_size,
            params,
            stable_iteration_space,
            options,
        )?;

        kernel.raw.shared_store(a_tile, shared_idx, a_val);

        let b_idx = if swap_b {
            let b_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul { a: col, b: k }),
            );
            let b_col = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: b_k }),
            );
            kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        } else {
            let b_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul { a: b_k, b: n }),
            );
            let b_col = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: col }),
            );
            kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        };

        let b_val = kernel.raw.def_var(
            DType::Simple(dtype),
            ValueState::Mut,
            Some(dtype.constant_float(0.0)?),
        );

        b_deepest = eval_node(
            root,
            input,
            b_node,
            b_val,
            resolved,
            graph,
            node_params,
            saved_params,
            kernel,
            b_idx,
            base,
            local_row,
            local_col,
            shared_size,
            tile_size,
            params,
            stable_iteration_space,
            options,
        )?;

        kernel.raw.shared_store(b_tile, shared_idx, b_val);

        kernel.raw.push_barrier();

        let inner = kernel.raw.def_var(
            DType::Simple(SimpleDType::U32),
            ValueState::Mut,
            Some(Op::ConstU32 { value: 0 }),
        );

        kernel.push_for_loop(inner, tile_size, one, |kernel| {
            let a_s_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul {
                    a: local_row,
                    b: tile_size,
                }),
            );
            let a_s_idx = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add {
                    a: a_s_row,
                    b: inner,
                }),
            );

            let a_val = kernel.raw.def_var(
                DType::Simple(dtype),
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: a_tile,
                    index: a_s_idx,
                }),
            );

            let b_s_row = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Inline,
                Some(Op::Mul {
                    a: inner,
                    b: tile_size,
                }),
            );
            let b_s_idx = kernel.raw.def_var(
                DType::Simple(SimpleDType::U32),
                ValueState::Immut,
                Some(Op::Add {
                    a: b_s_row,
                    b: local_col,
                }),
            );

            let b_val = kernel.raw.def_var(
                DType::Simple(dtype),
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: b_tile,
                    index: b_s_idx,
                }),
            );

            kernel.raw.overwrite_var(
                out,
                Op::Fma {
                    a: a_val,
                    b: b_val,
                    c: out,
                },
            );

            Ok(())
        })?;

        kernel.raw.push_barrier();

        Ok(())
    })?;

    deepest.append(&mut a_deepest);
    deepest.append(&mut b_deepest);

    Ok(deepest)
}
