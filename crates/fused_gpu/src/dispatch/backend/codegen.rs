use core::fmt::{self, Display, Write};

use crate::{
    dispatch::backend::{Axis, DType, Op, SimpleDType, ValueId, ValueState, kernel::RawKernel},
    errors::{Error, ErrorKind},
};

impl SimpleDType {
    #[inline]
    #[must_use]
    pub const fn fmt_c(self) -> &'static str {
        match self {
            Self::F64 => "double",
            Self::F32 => "float",
            Self::F16 => "__half",
            Self::BF16 => "hip_bfloat16",
            Self::Bool => "bool",
            Self::I32 => "int32_t",
            Self::U32 => "uint32_t",
        }
    }

    #[inline]
    #[must_use]
    pub const fn fmt_rust(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }

    #[inline]
    pub const fn fmt_wgsl(self) -> Result<&'static str, Error> {
        match self {
            Self::F64 => Ok("f64"),
            Self::F32 => Ok("f32"),
            Self::F16 => Ok("f16"),
            Self::BF16 => Err(Error {
                msg: "bf16 not supported",
                kind: ErrorKind::UnsupportedFeature,
                ctx: (),
            }),
            Self::Bool => Ok("bool"),
            Self::I32 => Ok("i32"),
            Self::U32 => Ok("u32"),
        }
    }
}

impl DType {
    #[inline]
    #[must_use]
    pub const fn fmt_c(self) -> &'static str {
        match self {
            Self::Simple(dtype) => dtype.fmt_c(),
            _ => todo!(),
        }
    }

    #[inline]
    pub const fn fmt_rust(self) -> Result<&'static str, Error> {
        let Self::Simple(dtype) = self else {
            return Err(Error {
                msg: "MMA not supported",
                kind: ErrorKind::UnsupportedFeature,
                ctx: (),
            });
        };

        Ok(dtype.fmt_rust())
    }

    #[inline]
    pub const fn fmt_wgsl(self) -> Result<&'static str, Error> {
        let Self::Simple(dtype) = self else {
            return Err(Error {
                msg: "MMA not supported",
                kind: ErrorKind::UnsupportedFeature,
                ctx: (),
            });
        };

        dtype.fmt_wgsl()
    }
}

impl Display for Axis {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.fmt_vec3())
    }
}

impl Axis {
    #[inline]
    #[must_use]
    pub const fn fmt_vec3(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        }
    }

    #[inline]
    #[must_use]
    pub const fn fmt_idx(self) -> &'static str {
        match self {
            Self::X => "0",
            Self::Y => "1",
            Self::Z => "2",
        }
    }
}

#[inline]
pub fn newline(pretty_print: bool, out: &mut String, nesting: usize) {
    if pretty_print {
        let _ = write!(out, "\n{}", "  ".repeat(nesting));
    } else {
        let _ = write!(out, " ");
    }
}

#[inline]
pub fn tab(pretty_print: bool, out: &mut String) {
    if pretty_print {
        let _ = write!(out, "  ");
    }
}

pub fn def_var_wgsl<F>(
    out: &mut String,
    nesting: &mut usize,
    id: ValueId,
    kernel: &RawKernel,
    process_op: F,
) -> Result<(), Error>
where
    F: FnOnce(&mut String, &Op, &mut usize, &RawKernel) -> Result<(), Error>,
{
    let val = &kernel.values[id];
    match val.state {
        ValueState::Masked | ValueState::Inline => {}
        ValueState::Const => {
            let _ = write!(out, "const v{id}: {}", val.dtype.fmt_wgsl()?);

            if let Some(op) = &val.init {
                let _ = out.write_str(" = ");
                process_op(out, op, nesting, kernel)?;
            }

            let _ = out.write_char(';');
        }
        var => {
            if var == ValueState::Immut {
                let _ = write!(out, "let v{id}: {}", val.dtype.fmt_wgsl()?);
            } else {
                let _ = write!(out, "var v{id}: {}", val.dtype.fmt_wgsl()?);
            }

            if let Some(op) = &val.init {
                let _ = out.write_str(" = ");
                process_op(out, op, nesting, kernel)?;
            }

            let _ = out.write_char(';');
        }
    }

    Ok(())
}

pub fn def_var_c<F>(
    out: &mut String,
    nesting: &mut usize,
    id: ValueId,
    kernel: &RawKernel,
    process_op: F,
) -> Result<(), Error>
where
    F: FnOnce(&mut String, &Op, &mut usize, &RawKernel) -> Result<(), Error>,
{
    let val = &kernel.values[id];

    match val.state {
        ValueState::Masked | ValueState::Inline => {}

        ValueState::Const => {
            let _ = write!(out, "const {} v{id}", val.dtype.fmt_c());

            if let Some(op) = &val.init {
                let _ = out.write_str(" = ");
                process_op(out, op, nesting, kernel)?;
            }

            let _ = out.write_char(';');
        }

        _ => {
            let _ = write!(out, "{} v{id}", val.dtype.fmt_c());

            if let Some(op) = &val.init {
                let _ = out.write_str(" = ");
                process_op(out, op, nesting, kernel)?;
            }

            let _ = out.write_char(';');
        }
    }

    Ok(())
}

pub fn render_val<F>(id: ValueId, kernel: &RawKernel, process_op: F) -> Result<String, Error>
where
    F: FnOnce(&mut String, &Op, &mut usize, &RawKernel) -> Result<(), Error>,
{
    let mut out = String::new();
    let val = &kernel.values[id];
    match val.state {
        ValueState::Inline => {
            if let Some(op) = &val.init {
                process_op(&mut out, op, &mut 0, kernel)?;
            }
        }
        ValueState::Masked => {}
        _ => {
            let _ = write!(out, "v{id}");
        }
    }
    Ok(out)
}
