use crate::dispatch::backend::{Axis, MetaId, ParamId, SharedId, ValueId};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Nop,

    DefineVar {
        id: ValueId,
    },
    OverwriteVar {
        id: ValueId,
        val: ValueId,
    },
    AddAssign {
        id: ValueId,
        val: ValueId,
    },
    MulAssign {
        id: ValueId,
        val: ValueId,
    },
    DivAssign {
        id: ValueId,
        val: ValueId,
    },
    SubAssign {
        id: ValueId,
        val: ValueId,
    },
    ShlAssign {
        id: ValueId,
        val: ValueId,
    },
    ShrAssign {
        id: ValueId,
        val: ValueId,
    },
    CopyVar {
        id: ValueId,
    },

    ConstF32 {
        value: f32,
    },
    ConstF16 {
        value: half::f16,
    },
    ConstBf16 {
        value: half::bf16,
    },
    ConstU32 {
        value: u32,
    },
    ConstI32 {
        value: i32,
    },

    ReadMeta {
        param: ParamId,
        field: MetaId,
    },

    LocalId {
        axis: Axis,
    },
    BlockId {
        axis: Axis,
    },
    GlobalId {
        axis: Axis,
    },

    Add {
        a: ValueId,
        b: ValueId,
    },
    Sub {
        a: ValueId,
        b: ValueId,
    },
    Mul {
        a: ValueId,
        b: ValueId,
    },
    Div {
        a: ValueId,
        b: ValueId,
    },
    Mod {
        a: ValueId,
        b: ValueId,
    },
    Pow {
        a: ValueId,
        b: ValueId,
    },

    Shl {
        a: ValueId,
        b: ValueId,
    },

    Shr {
        a: ValueId,
        b: ValueId,
    },

    /// (Fused) Operation `a * b + c`
    Fma {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    Exp {
        x: ValueId,
    },
    Abs {
        x: ValueId,
    },
    Neg {
        x: ValueId,
    },
    Log {
        x: ValueId,
    },
    Tanh {
        x: ValueId,
    },
    Sqrt {
        x: ValueId,
    },

    ParamLoad {
        param: ParamId,
        index: ValueId,
    },
    ParamStore {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamAccum {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamMul {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamDiv {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamSub {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamShl {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamShr {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },

    SharedLoad {
        mem: SharedId,
        index: ValueId,
    },
    SharedStore {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedAccum {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedMul {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedDiv {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedSub {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedShl {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedShr {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },

    Eq {
        a: ValueId,
        b: ValueId,
    },

    Ne {
        a: ValueId,
        b: ValueId,
    },

    Lt {
        a: ValueId,
        b: ValueId,
    },

    Gt {
        a: ValueId,
        b: ValueId,
    },

    Le {
        a: ValueId,
        b: ValueId,
    },

    Ge {
        a: ValueId,
        b: ValueId,
    },

    Not {
        cond: ValueId,
    },

    Max {
        a: ValueId,
        b: ValueId,
    },
    Min {
        a: ValueId,
        b: ValueId,
    },

    CastF32 {
        id: ValueId,
    },

    CastF16 {
        id: ValueId,
    },

    CastBF16 {
        id: ValueId,
    },

    CastU32 {
        id: ValueId,
    },

    CastI32 {
        id: ValueId,
    },

    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    ForLoopBegin {
        index: ValueId,
        end: ValueId,
        step: ValueId,
    },

    ForeverLoopBegin,

    Continue,

    Break,

    IfBegin {
        cond: ValueId,
    },

    ElseBegin,

    StartScope,

    EndScope,

    Barrier,

    Return,
}

impl Op {
    #[must_use]
    pub fn does_read(&self, value_id: ValueId) -> bool {
        match self {
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF32 { .. }
            | Self::ConstF16 { .. }
            | Self::ConstBf16 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop
            | Self::StartScope => false,
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x }
            | Self::CopyVar { id: x }
            | Self::CastF32 { id: x }
            | Self::CastF16 { id: x }
            | Self::CastBF16 { id: x }
            | Self::CastU32 { id: x }
            | Self::CastI32 { id: x } => x == &value_id,
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => a == &value_id || b == &value_id,
            Self::Fma { a, b, c } => a == &value_id || b == &value_id || c == &value_id,
            Self::AddAssign { val, .. }
            | Self::DivAssign { val, .. }
            | Self::MulAssign { val, .. }
            | Self::ShlAssign { val, .. }
            | Self::ShrAssign { val, .. }
            | Self::SubAssign { val, .. }
            | Self::OverwriteVar { val, .. } => val == &value_id,
            Self::ForLoopBegin { index, end, step } => {
                index == &value_id || end == &value_id || step == &value_id
            }
            Self::IfBegin { cond } => cond == &value_id,
            Self::ParamAccum { index, value, .. }
            | Self::ParamDiv { index, value, .. }
            | Self::ParamMul { index, value, .. }
            | Self::ParamShl { index, value, .. }
            | Self::ParamShr { index, value, .. }
            | Self::ParamStore { index, value, .. }
            | Self::ParamSub { index, value, .. }
            | Self::SharedAccum { index, value, .. }
            | Self::SharedDiv { index, value, .. }
            | Self::SharedMul { index, value, .. }
            | Self::SharedShl { index, value, .. }
            | Self::SharedShr { index, value, .. }
            | Self::SharedStore { index, value, .. }
            | Self::SharedSub { index, value, .. } => index == &value_id || value == &value_id,
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => index == &value_id,
            Self::Select { cond, a, b } => cond == &value_id || a == &value_id || b == &value_id,
        }
    }

    pub const fn replace_usage(&mut self, old_id: ValueId, new_id: ValueId) {
        const fn replace_if_eq(value_id: &mut ValueId, old_id: ValueId, new_id: ValueId) {
            if *value_id == old_id {
                *value_id = new_id;
            }
        }

        match self {
            Self::CopyVar { id } => replace_if_eq(id, old_id, new_id),
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x }
            | Self::CastF32 { id: x }
            | Self::CastF16 { id: x }
            | Self::CastBF16 { id: x }
            | Self::CastU32 { id: x }
            | Self::CastI32 { id: x } => replace_if_eq(x, old_id, new_id),
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => {
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
            }
            Self::Fma { a, b, c } => {
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
                replace_if_eq(c, old_id, new_id);
            }
            Self::AddAssign { val, id }
            | Self::DivAssign { val, id }
            | Self::MulAssign { val, id }
            | Self::ShlAssign { val, id }
            | Self::ShrAssign { val, id }
            | Self::SubAssign { val, id }
            | Self::OverwriteVar { val, id } => {
                replace_if_eq(id, old_id, new_id);
                replace_if_eq(val, old_id, new_id);
            }
            Self::ForLoopBegin { index, end, step } => {
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(end, old_id, new_id);
                replace_if_eq(step, old_id, new_id);
            }
            Self::IfBegin { cond } => replace_if_eq(cond, old_id, new_id),
            Self::ParamAccum { index, value, .. }
            | Self::ParamDiv { index, value, .. }
            | Self::ParamMul { index, value, .. }
            | Self::ParamShl { index, value, .. }
            | Self::ParamShr { index, value, .. }
            | Self::ParamStore { index, value, .. }
            | Self::ParamSub { index, value, .. }
            | Self::SharedAccum { index, value, .. }
            | Self::SharedDiv { index, value, .. }
            | Self::SharedMul { index, value, .. }
            | Self::SharedShl { index, value, .. }
            | Self::SharedShr { index, value, .. }
            | Self::SharedStore { index, value, .. }
            | Self::SharedSub { index, value, .. } => {
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(value, old_id, new_id);
            }
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => {
                replace_if_eq(index, old_id, new_id);
            }
            Self::Select { cond, a, b } => {
                replace_if_eq(cond, old_id, new_id);
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
            }
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF32 { .. }
            | Self::ConstF16 { .. }
            | Self::ConstBf16 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop
            | Self::StartScope => {}
        }
    }

    #[must_use]
    pub fn does_write(&self, value_id: ValueId) -> bool {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. } => id == &value_id,
            _ => false,
        }
    }

    #[must_use]
    pub const fn writes_to(&self) -> Option<ValueId> {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. } => Some(*id),
            _ => None,
        }
    }

    #[must_use]
    pub fn does_mutate(&self, value_id: ValueId) -> bool {
        match self {
            Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. } => id == &value_id,
            _ => false,
        }
    }

    #[must_use]
    pub const fn is_zero(&self) -> bool {
        match self {
            Self::ConstF32 { value } => value.abs() == 0.0,
            Self::ConstI32 { value } => *value == 0,
            Self::ConstU32 { value } => *value == 0,
            _ => false,
        }
    }

    #[must_use]
    pub const fn is_one(&self) -> bool {
        match self {
            Self::ConstF32 { value } => (*value - 1.0).abs() < 1e-9,
            Self::ConstI32 { value } => *value == 1,
            Self::ConstU32 { value } => *value == 1,
            _ => false,
        }
    }
}
