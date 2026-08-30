use crate::{
    dispatch::{
        GpuBackend,
        backend::{
            Axis, DType, Graph, Metadata, Op, Param, ParamTy, ValueState,
            kernel::{Kernel, RawKernel},
        },
    },
    errors::Error,
};
use std::{vec, vec::Vec};

#[inline]
pub fn lower_optim<B: GpuBackend>(graph: &Graph<B>, meta: Metadata) -> Result<Kernel, Error> {
    let root = graph.nodes.len() - 1;
    let root_node = &graph.nodes[root];
    let dtype = root_node.dtype;

    let mut kernel = Kernel {
        raw: RawKernel {
            meta,
            shared: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            block: [0; 3],
            root,
            iter_space: root_node.shape.clone(),
        },
        params: Vec::with_capacity(3),
        meta: vec![false; meta.fields],
    };

    if graph
        .nodes
        .iter()
        .all(|x| x.op.is_elementwise() || x.op.is_leaf())
    {
        kernel.raw.block = [256, 1, 1];
    } else {
        kernel.raw.block = [16, 16, 1];
    }

    kernel.params.push(Param {
        dtype: DType::U32,
        ty: ParamTy::Uniform,
        pid: 0,
    });

    let weight_param = kernel.params.len();
    kernel.params.push(Param {
        dtype,
        ty: ParamTy::ReadWrite,
        pid: 1,
    });

    let grad_param = kernel.params.len();
    kernel.params.push(Param {
        dtype,
        ty: ParamTy::ReadWrite,
        pid: 2,
    });

    let mut dims = Vec::new();

    for &meta_index in &root_node.shape {
        let dim_val = kernel.raw.def_var(
            DType::U32,
            ValueState::Immut,
            Some(Op::ReadMeta {
                param: 0,
                field: meta_index,
            }),
        );

        kernel.register_meta(meta_index);

        dims.push(dim_val);
    }

    let gid = kernel.raw.def_var(
        DType::U32,
        ValueState::Mut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let total = dims[0];
    kernel.raw.update_var_state(total, ValueState::Mut);

    if dims.len() > 1 {
        let gid2 = kernel.raw.def_var(DType::U32, ValueState::Mut, None);

        for (i, &d) in dims.iter().enumerate().skip(1) {
            kernel.raw.overwrite_var(
                gid2,
                Op::GlobalId {
                    axis: (i as u8).try_into().unwrap_or(Axis::Z),
                },
            );

            kernel.raw.overwrite_var(gid2, Op::Mul { a: gid2, b: d });

            kernel.raw.overwrite_var(gid, Op::Add { a: gid, b: gid2 });

            kernel.raw.overwrite_var(total, Op::Mul { a: total, b: d });
        }
    }

    let row = kernel.raw.def_var(
        DType::U32,
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::Y }),
    );
    let col = kernel.raw.def_var(
        DType::U32,
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let lr = kernel.raw.def_var(
        DType::F32,
        ValueState::Immut,
        Some(Op::ReadMeta { param: 0, field: 0 }),
    );

    let lr_normalized = match dtype {
        DType::F16 => kernel
            .raw
            .def_var(dtype, ValueState::Immut, Some(Op::CastF16 { id: lr })),
        DType::BF16 => kernel
            .raw
            .def_var(dtype, ValueState::Immut, Some(Op::CastBF16 { id: lr })),
        _ => lr,
    };

    (graph.optim.lower)(
        &mut kernel,
        dtype,
        lr_normalized,
        weight_param,
        grad_param,
        gid,
        row,
        col,
    )?;

    Ok(kernel)
}
