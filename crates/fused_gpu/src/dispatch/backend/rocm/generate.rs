use log::debug;

use crate::{dispatch::{
    CompilationOptions, DebugCompilationOptions, backend::{
        DType, Op, Param, ParamTy, SharedAlloc, SimpleDType, ValueId, codegen::{self, def_var_c, newline}, kernel::RawKernel,
    },
}, errors::Error};

use core::fmt::Write;

pub(super) fn generate_hip(src: &RawKernel, params: &[Param], options: &CompilationOptions) -> Result<String, Error> {
    let Ok(mut out) = r#"
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
"#.parse::<String>();

    let pretty_print = options.debug.contains(DebugCompilationOptions::PRETTY_PRINT_IR);

    out.push_str("extern \"C\" __global__\nvoid kernel(");

    gen_args(&mut out, params);

    out.push_str(") {");

    gen_shared(&mut out, &src.shared, pretty_print);

    let mut nesting = 1;

    for op in &src.ops {
        newline(pretty_print, &mut out, nesting);

        process_op(&mut out, op, &mut nesting, src)?;
    }

    newline(pretty_print, &mut out, 0);

    out.push('}');

    if pretty_print {
        debug!("generated kernel:\n{out}");
    }

    Ok(out)
}

fn gen_shared(out: &mut String, shared: &[SharedAlloc], pretty_print: bool) {
    for (i, s) in shared.iter().enumerate() {
        newline(pretty_print, out, 0);

        let _ = write!(
            out,
            "__shared__ {} shared{i}[{}];",
            s.dtype.fmt_c(),
            s.size
        );
    }
}

fn gen_args(out: &mut String, params: &[Param]) {
    for (i, param) in params.iter().enumerate() {
        if i != 0 {
            let _ = out.write_str(", ");
        }

        let read_only = if param.ty == ParamTy::ReadWrite {
            ""
        } else {
            "const "
        };

        let _ = write!(out, "{}{} *param{}", read_only, param.dtype.fmt_c(), param.pid);
    }
}

fn process_op(
    out: &mut String,
    op: &Op,
    nesting: &mut usize,
    kernel: &RawKernel,
) -> Result<(), Error> {
    match op {
        Op::Nop => {}

        Op::DefineVar { id } => def_var_c(out, nesting, *id, kernel, process_op)?,

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
            let _ = write!(out, "{value:?}");
        }

        Op::ConstF32 { value } => {
            let _ = write!(out, "{value:?}f");
        }

        Op::ConstF16 { value } => {
            let _ = write!(out, "__float2half({value:?}f)");
        }

        Op::ConstBf16 { value } => {
            let _ = write!(out, "__float2bfloat16({value:?}f)");
        }

        Op::ConstU32 { value } => {
            let _ = write!(out, "{value}u");
        }

        Op::ConstI32 { value } => {
            let _ = write!(out, "{value}");
        }

        Op::ReadMeta { param, field } => {
            let _ = write!(out, "param{param}[{field}]");
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
            let _ = write!(out, "threadIdx.{axis}");
        }

        Op::BlockId { axis } => {
            let _ = write!(out, "blockIdx.{axis}");
        }

        Op::GlobalId { axis } => {
            let _ = write!(out, "blockIdx.{axis} * {} + threadIdx.{axis}", kernel.block[*axis as usize]);
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
            let symbol = if kernel.values[*a].dtype == DType::Simple(SimpleDType::F32) {
                "fmaf"
            } else {
                "fma"
            }; 
            let _ = write!(
                out,
                "{symbol}({}, {}, {})",
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
                "({} ? {} : {})",
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
            let _ = write!(out, "static_cast<double>({})", render_val(*id, kernel)?);
        }

        Op::CastF32 { id } => {
            let _ = write!(out, "static_cast<float>({})", render_val(*id, kernel)?);
        }

        Op::CastF16 { id } => {
            let _ = write!(out, "static_cast<__half>({})", render_val(*id, kernel)?);
        }

        Op::CastU32 { id } => {
            let _ = write!(out, "static_cast<uint32_t>({})", render_val(*id, kernel)?);
        }

        Op::CastI32 { id } => {
            let _ = write!(out, "static_cast<int32_t>({})", render_val(*id, kernel)?);
        }

        Op::CastBF16 { id } => {
            let _ = write!(out, "static_cast<hip_bfloat16>({})", render_val(*id, kernel)?);
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
            let _ = write!(out, "for (;;) {{");
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
            *nesting += 1;
        }

        Op::EndScope => {
            let _ = write!(out, "}}");
            *nesting -= 1;
        }

        Op::Barrier => {
            let _ = write!(out, "__syncthreads();");
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

        _ => todo!("MMA not supported"),
    }

    Ok(())
}

fn render_val(id: ValueId, kernel: &RawKernel) -> Result<String, Error> {
    codegen::render_val(id, kernel, process_op)
}
