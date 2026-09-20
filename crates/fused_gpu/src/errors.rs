//! Safe error handling for all possibilities.

use crate::dispatch::backend::{GraphOp, MetaId, NodeId};
use core::{
    error::Error as CoreError,
    fmt::{Debug, Display, Formatter, Result},
};
use std::vec::Vec;

#[cfg(feature = "telemetry")]
use gpu_telemetry::errors::{self, ErrorKind as TelemetryErrorKind};

/// Generic error type with message, type, and display.
#[derive(Clone)]
pub struct Error<C = ()> {
    pub msg: &'static str,
    pub kind: ErrorKind,
    pub ctx: C,
}

impl CoreError for Error {}

impl<C: Debug> Debug for Error<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if size_of::<C>() == 0 {
            write!(f, "Error {{ kind: {:?}, msg: {:?} }}", self.kind, self.msg)
        } else {
            write!(
                f,
                "Error {{ kind: {:?}, msg: {:?}, ctx: {:?} }}",
                self.kind, self.msg, self.ctx
            )
        }
    }
}

impl From<Error> for Error<GraphErrorContext<'_>> {
    fn from(err: Error) -> Self {
        Self {
            msg: err.msg,
            kind: err.kind,
            ctx: ().into(),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl Display for Error<GraphErrorContext<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        debug_assert_eq!(self.kind, ErrorKind::ComputeGraphError);

        write!(f, "{}: {}, {}", self.kind, self.msg, self.ctx)
    }
}

#[cfg(feature = "io")]
impl Display for Error<crate::io::SerialTensorError> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        debug_assert_eq!(self.kind, ErrorKind::ComputeGraphError);

        write!(f, "{}: {}, {}", self.kind, self.msg, self.ctx)
    }
}

#[cfg(feature = "telemetry")]
impl From<gpu_telemetry::errors::Error> for Error {
    fn from(value: errors::Error) -> Self {
        let kind = match value.kind {
            TelemetryErrorKind::FailedAdapterCreation => ErrorKind::FailedAdapterCreation,
            TelemetryErrorKind::FailedEventCreation => ErrorKind::FailedEventCreation,
            TelemetryErrorKind::FailedInfoQuery => ErrorKind::FailedInfoQuery,
            TelemetryErrorKind::InvalidEventHandle => ErrorKind::InvalidEventHandle,
            TelemetryErrorKind::InvalidInterval => ErrorKind::InvalidInterval,
            TelemetryErrorKind::SyncError => ErrorKind::SyncError,
            TelemetryErrorKind::UnsupportedFeature => ErrorKind::UnsupportedFeature,
            TelemetryErrorKind::WaitFailed => ErrorKind::WaitFailed,
        };

        Self {
            msg: value.msg,
            kind,
            ctx: (),
        }
    }
}

/// Generic error type for all possiblities within `fused_gpu`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    EnvNotSet,
    UnsupportedLimit,
    UnsupportedFeature,
    ParamNotMaterialized,
    GraphEmpty,
    InvalidDType,
    UnresolvedInput,
    UnresolvedOutput,
    InvalidArgument,
    ComputeGraphError,
    FailedBufferCopy,
    UnresolvedRedirection,
    SyncError,
    InvalidInterval,
    InvalidEventHandle,
    WaitFailed,
    FailedAdapterCreation,
    FailedEventCreation,
    FailedInfoQuery,
    InternalError,
    SerializationError,
    PollFailed,
    InvalidDevice,
    InvalidMemoryOp,
    InconsistentCapture,
    AlreadySet,
    OutOfMemory,
    NotInitialized,
    ProfilerError,
    FileNotFound,
    LaunchFailure,
    PeerAccessError,
    UnresolvedSymbol,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EnvNotSet => f.write_str("environment not set"),
            Self::UnsupportedLimit => f.write_str("unsupported limits"),
            Self::UnsupportedFeature => f.write_str("unsupported feature"),
            Self::ParamNotMaterialized => f.write_str("param not materialized"),
            Self::GraphEmpty => f.write_str("graph empty"),
            Self::InvalidDType => f.write_str("invalid data type"),
            Self::UnresolvedInput => f.write_str("unresolved input"),
            Self::UnresolvedOutput => f.write_str("unresolved output"),
            Self::InvalidArgument => f.write_str("invalid argument"),
            Self::ComputeGraphError => f.write_str("compute graph error"),
            Self::FailedBufferCopy => f.write_str("failed to copy GPU buffers"),
            Self::FailedAdapterCreation => f.write_str("failed adapter creation"),
            Self::FailedEventCreation => f.write_str("failed event creation"),
            Self::FailedInfoQuery => f.write_str("failed info query"),
            Self::InvalidEventHandle => f.write_str("invalid event handle"),
            Self::InvalidInterval => f.write_str("invalid interval"),
            Self::SyncError => f.write_str("synchronization error"),
            Self::WaitFailed => f.write_str("wait failed"),
            Self::UnresolvedRedirection => f.write_str("unresolved redirection"),
            Self::SerializationError => f.write_str("serialization or I/O error"),
            Self::InternalError => f.write_str("internal error (maybe report bug)"),
            Self::PollFailed => f.write_str("poll failed"),
            Self::InvalidDevice => f.write_str("invalid device"),
            Self::InvalidMemoryOp => f.write_str("invalid memory operation"),
            Self::InconsistentCapture => f.write_str("inconsistent capture"),
            Self::AlreadySet => f.write_str("already set"),
            Self::OutOfMemory => f.write_str("out of memory"),
            Self::NotInitialized => f.write_str("not initialized"),
            Self::ProfilerError => f.write_str("profiler error"),
            Self::FileNotFound => f.write_str("file not found"),
            Self::LaunchFailure => f.write_str("launch failure"),
            Self::PeerAccessError => f.write_str("peer access error"),
            Self::UnresolvedSymbol => f.write_str("unresolved symbol"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GraphErrorContext<'a> {
    Unrelated,

    CycleDetected {
        node: NodeId,
        path: Vec<NodeId>,
    },

    InvalidInputs {
        node: NodeId,
        arity: usize,
        args: usize,
    },

    MissingInput {
        node: NodeId,
        input: NodeId,
    },

    ShapeMismatch {
        node: NodeId,
        all_hand_sides: Vec<Vec<MetaId>>,
        op: GraphOp<'a>,
    },

    RankMismatch {
        node: NodeId,
        all_hand_sides: Vec<usize>,
    },

    InvalidAxis {
        node: NodeId,
        axis: usize,
        rank: usize,
    },

    MissingMetadata {
        node: NodeId,
        meta: MetaId,
    },

    LowRank {
        node: NodeId,
        rank: usize,
        required: usize,
    },

    CannotInferShape {
        node: NodeId,
        all_hand_sides: Vec<Vec<MetaId>>,
        op: GraphOp<'a>,
    },
}

impl From<()> for GraphErrorContext<'_> {
    #[allow(clippy::ignored_unit_patterns)]
    fn from(_: ()) -> Self {
        Self::Unrelated
    }
}

impl Display for GraphErrorContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Unrelated => {
                write!(f, "unrelated")
            }

            Self::CycleDetected { node, path } => {
                write!(f, "cycle detected at node {node} following {path:?}")
            }

            Self::InvalidAxis { node, axis, rank } => {
                write!(f, "invalid axis {axis} for rank of {rank} at node {node}")
            }

            Self::InvalidInputs { node, arity, args } => {
                write!(
                    f,
                    "invalid input count {args} for arity of {arity} at node {node}"
                )
            }

            Self::MissingInput { node, input } => {
                write!(f, "missing input at node {node} on input node {input}")
            }

            Self::MissingMetadata { node, meta } => {
                write!(f, "missing metadata at node {node} on field f{meta}")
            }

            Self::ShapeMismatch {
                node,
                all_hand_sides,
                op,
            } => write!(
                f,
                "shape mismatch at node {node} {}",
                op.debug(all_hand_sides)
            ),

            Self::RankMismatch {
                node,
                all_hand_sides,
            } => {
                let closure = |f: &mut Formatter, node: &usize, all_hand_sides: &[usize]| {
                    write!(f, "rank mismatch at node {node}")?;

                    write!(f, "{}", all_hand_sides[0])?;

                    for hand_side in all_hand_sides.iter().skip(1) {
                        write!(f, "== {hand_side}")?;
                    }

                    Ok(())
                };

                closure(f, node, all_hand_sides)
            }

            Self::LowRank {
                node,
                rank,
                required,
            } => write!(f, "rank too low at node {node} ({rank} >= {required})"),

            Self::CannotInferShape {
                node,
                all_hand_sides,
                op,
            } => write!(
                f,
                "cannot infer shape of node {node} with op {}",
                op.debug(all_hand_sides)
            ),
        }
    }
}

impl CoreError for GraphErrorContext<'_> {}
