use crate::{
    dispatch::backend::{
        Op, Param, ParamTy, SimpleDType, ValueId, ValueState, codegen::{self, def_var_wgsl, newline, tab}, kernel::RawKernel,
    }, errors::{Error, ErrorKind},
};
use core::fmt::Write;
use std::{string::String, vec::Vec};

pub use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

#[inline]
pub(super) fn generate_layout_desc(params: &[Param]) -> Vec<BindGroupLayoutEntry> {
    params
        .iter()
        .map(|x| {
            let ty = match x.ty {
                ParamTy::Uniform => BufferBindingType::Uniform,
                ParamTy::ReadOnly => BufferBindingType::Storage { read_only: true },
                ParamTy::ReadWrite => BufferBindingType::Storage { read_only: false },
            };

            BindGroupLayoutEntry {
                binding: x.pid as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect::<Vec<_>>()
}

#[inline]
pub(super) fn generate_wgsl(
    kernel: &RawKernel,
    params: &[Param],
    pretty_print: bool,
) -> Result<String, Error> {
    let mut out = "enable f16;".parse().map_err(|_| Error {
        msg: "failed to copy &'static str to String",
        kind: ErrorKind::InternalError,
        ctx: (),
    })?;

    newline(pretty_print, &mut out, 0);

    emit_bindings(kernel, params, &mut out, pretty_print)?;
    newline(pretty_print, &mut out, 0);

    emit_entry(kernel, &mut out, pretty_print)?;

    if pretty_print {
        log::log!(log::Level::Debug, "{out}");
    }

    Ok(out)
}

#[inline]
fn emit_bindings(
    kernel: &RawKernel,
    params: &[Param],
    out: &mut String,
    pretty_print: bool,
) -> Result<(), Error> {
    let _ = write!(out, "struct Meta {{");
    newline(pretty_print, out, 0);
    let _ = write!(out, "f0: f32,");
    for f in 1..kernel.meta.fields {
        newline(pretty_print, out, 1);
        let _ = write!(out, "f{f}: u32,");
    }
    newline(pretty_print, out, 0);
    let _ = write!(out, "}}");

    newline(pretty_print, out, 0);

    for p in params {
        let var_type = match p.ty {
            ParamTy::ReadOnly => "var<storage, read>",
            ParamTy::ReadWrite => "var<storage, read_write>",
            ParamTy::Uniform => "var<uniform>",
        };

        newline(pretty_print, out, 0);

        let pid = p.pid;

        if p.ty == ParamTy::Uniform && p.dtype == SimpleDType::U32 {
            let _ = write!(
                out,
                "@group(0) @binding({pid}) {var_type} param{pid}: Meta;"
            );
        } else {
            let _ = write!(
                out,
                "@group(0) @binding({pid}) {var_type} param{pid}: array<{}>;",
                p.dtype.fmt_wgsl()?,
            );
        }
    }

    newline(pretty_print, out, 0);

    for (i, s) in kernel.shared.iter().enumerate() {
        newline(pretty_print, out, 0);
        let _ = write!(
            out,
            "var<workgroup> shared{i}: array<{}, ({})>;",
            s.dtype.fmt_wgsl()?,
            s.size
        );
    }

    Ok(())
}

#[inline]
fn emit_entry(kernel: &RawKernel, out: &mut String, pretty_print: bool) -> Result<(), Error> {
    newline(pretty_print, out, 0);
    let _ = write!(
        out,
        "@compute @workgroup_size({}, ({}), ({})) ",
        kernel.block[0], kernel.block[1], kernel.block[2]
    );

    newline(pretty_print, out, 0);
    let _ = write!(out, "fn main(");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(local_invocation_id) lid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(workgroup_id) bid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(global_invocation_id) gid: vec3<u32>,");

    newline(pretty_print, out, 0);
    let _ = write!(out, ") {{");

    emit_ops(kernel, out, pretty_print)?;
    newline(pretty_print, out, 0);

    let _ = write!(out, "}}");

    Ok(())
}

#[inline]
fn emit_ops(kernel: &RawKernel, out: &mut String, pretty_print: bool) -> Result<(), Error> {
    let mut nesting = 0;

    for op in &kernel.ops {
        if let Op::DefineVar { id } = op
            && matches!(
                kernel.values[*id].state,
                ValueState::Inline | ValueState::Masked
            )
        {
            continue;
        }

        newline(pretty_print, out, nesting);

        if *op != Op::EndScope {
            tab(pretty_print, out);
        }

        process_op(out, op, &mut nesting, kernel)?;
    }

    Ok(())
}

fn process_op(
    out: &mut String,
    op: &Op,
    nesting: &mut usize,
    kernel: &RawKernel,
) -> Result<(), Error> {
    match op {
        Op::Nop => {}

        Op::DefineVar { id } => def_var_wgsl(out, nesting, *id, kernel, process_op)?,

        Op::OverwriteVar { id, val } => {
            let _ = write!(out, "v{id} = {};", render_val(*val, kernel)?);
        }

        Op::AddAssign { id, val } => {
            let _ = write!(out, "v{id} += {};", render_val(*val, kernel)?);
        }

        Op::MulAssign { id, val } => {
            let _ = write!(out, "v{id} *= {};", render_val(*val, kernel)?);
        }

        Op::DivAssign { id, val } => {
            let _ = write!(out, "v{id} /= {};", render_val(*val, kernel)?);
        }

        Op::SubAssign { id, val } => {
            let _ = write!(out, "v{id} -= {};", render_val(*val, kernel)?);
        }

        Op::ShlAssign { id, val } => {
            let _ = write!(out, "v{id} <<= {};", render_val(*val, kernel)?);
        }

        Op::ShrAssign { id, val } => {
            let _ = write!(out, "v{id} >>= {};", render_val(*val, kernel)?);
        }

        Op::CopyVar { id } => {
            let _ = out.write_str(&render_val(*id, kernel)?);
        }

        Op::ConstF64 { value } => {
            let _ = write!(out, "{value}d");
        }

        Op::ConstF32 { value } => {
            let _ = write!(out, "{value}f");
        }

        Op::ConstF16 { value } => {
            let _ = write!(out, "{value}h");
        }

        Op::ConstBf16 { value: _ } => {
            return Err(Error {
                msg: "cannot evaluate constant `bf16`",
                kind: ErrorKind::InvalidDType,
                ctx: (),
            });
        }

        Op::ConstU32 { value } => {
            let _ = write!(out, "{value}u");
        }

        Op::ConstI32 { value } => {
            let _ = write!(out, "{value}");
        }

        Op::ReadMeta { param, field } => {
            let _ = write!(out, "param{param}.f{field}");
        }

        Op::Eq { a, b } => {
            let _ = write!(
                out,
                "({}) == ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Ne { a, b } => {
            let _ = write!(
                out,
                "({}) != ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Lt { a, b } => {
            let _ = write!(
                out,
                "({}) < ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Gt { a, b } => {
            let _ = write!(
                out,
                "({}) > ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Le { a, b } => {
            let _ = write!(
                out,
                "({}) <= ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Ge { a, b } => {
            let _ = write!(
                out,
                "({}) >= ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::LocalId { axis } => {
            let _ = write!(out, "lid.{axis}");
        }

        Op::BlockId { axis } => {
            let _ = write!(out, "bid.{axis}");
        }

        Op::GlobalId { axis } => {
            let _ = write!(out, "gid.{axis}");
        }

        Op::Add { a, b } => {
            let _ = write!(
                out,
                "({}) + ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Sub { a, b } => {
            let _ = write!(
                out,
                "({}) - ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Mul { a, b } => {
            let _ = write!(
                out,
                "({}) * ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Div { a, b } => {
            let _ = write!(
                out,
                "({}) / ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Mod { a, b } => {
            let _ = write!(
                out,
                "({}) % ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Pow { a, b } => {
            let _ = write!(
                out,
                "pow({}, {})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Shl { a, b } => {
            let _ = write!(
                out,
                "({}) << ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Shr { a, b } => {
            let _ = write!(
                out,
                "({}) >> ({})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Fma { a, b, c } => {
            let _ = write!(
                out,
                "fma({}, {}, {})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?,
                render_val(*c, kernel)?
            );
        }

        Op::Max { a, b } => {
            let _ = write!(
                out,
                "max({}, {})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Min { a, b } => {
            let _ = write!(
                out,
                "min({}, {})",
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Select { cond, a, b } => {
            let _ = write!(
                out,
                "select({}, {}, {})",
                render_val(*cond, kernel)?,
                render_val(*a, kernel)?,
                render_val(*b, kernel)?
            );
        }

        Op::Exp { x } => {
            let _ = write!(out, "exp({})", render_val(*x, kernel)?);
        }

        Op::Abs { x } => {
            let _ = write!(out, "abs({})", render_val(*x, kernel)?);
        }

        Op::Neg { x } => {
            let _ = write!(out, "-({})", render_val(*x, kernel)?);
        }

        Op::Log { x } => {
            let _ = write!(out, "log({})", render_val(*x, kernel)?);
        }

        Op::Tanh { x } => {
            let _ = write!(out, "tanh({})", render_val(*x, kernel)?);
        }

        Op::Sqrt { x } => {
            let _ = write!(out, "sqrt({})", render_val(*x, kernel)?);
        }

        Op::ParamLoad { param, index } => {
            let _ = write!(out, "param{param}[{}]", render_val(*index, kernel)?);
        }

        Op::Not { cond } => {
            let _ = write!(out, "!({})", render_val(*cond, kernel)?);
        }

        Op::CastF64 { id } => {
            let _ = write!(out, "f64({})", render_val(*id, kernel)?);
        }

        Op::CastF32 { id } => {
            let _ = write!(out, "f32({})", render_val(*id, kernel)?);
        }

        Op::CastF16 { id } => {
            let _ = write!(out, "f16({})", render_val(*id, kernel)?);
        }

        Op::CastU32 { id } => {
            let _ = write!(out, "u32({})", render_val(*id, kernel)?);
        }

        Op::CastI32 { id } => {
            let _ = write!(out, "i32({})", render_val(*id, kernel)?);
        }

        Op::CastBF16 { .. } => {
            return Err(Error {
                msg: "bf16 not supported",
                kind: ErrorKind::InvalidDType,
                ctx: (),
            });
        }

        Op::ParamStore {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] = {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamAccum {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] += {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamMul {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] *= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamDiv {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] /= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamSub {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] -= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamShl {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] <<= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ParamShr {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] >>= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedLoad { mem, index } => {
            let _ = write!(out, "shared{mem}[{}]", render_val(*index, kernel)?);
        }

        Op::SharedStore { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] = {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedAccum { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] += {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedMul { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] *= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedDiv { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] /= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedSub { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] -= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedShl { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] <<= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::SharedShr { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] >>= {};",
                render_val(*index, kernel)?,
                render_val(*value, kernel)?
            );
        }

        Op::ForLoopBegin { index, end, step } => {
            let _ = write!(
                out,
                "for (; v{index} < {}; v{index} += {}) {{",
                render_val(*end, kernel)?,
                render_val(*step, kernel)?,
            );
            *nesting += 1;
        }

        Op::ForeverLoopBegin => {
            let _ = write!(out, "loop {{");
            *nesting += 1;
        }

        Op::IfBegin { cond } => {
            let _ = write!(out, "if ({}) {{", render_val(*cond, kernel)?);
            *nesting += 1;
        }

        Op::ElseBegin => {
            let _ = write!(out, "else {{");
            *nesting += 1;
        }

        Op::StartScope => {
            let _ = write!(out, "{{");
        }

        Op::EndScope => {
            let _ = write!(out, "}}");
            *nesting -= 1;
        }

        Op::Barrier => {
            let _ = write!(out, "workgroupBarrier();");
        }

        Op::Return => {
            let _ = write!(out, "return;");
        }

        Op::Continue => {
            let _ = write!(out, "continue;");
        }

        Op::Break => {
            let _ = write!(out, "break;");
        }

        _ => {
            return Err(Error {
                msg: "MMA not supported",
                kind: ErrorKind::UnsupportedFeature,
                ctx: (),
            });
        }
    }

    Ok(())
}

fn render_val(id: ValueId, kernel: &RawKernel) -> Result<String, Error> {
    codegen::render_val(id, kernel, process_op)
}
