pub mod dtw;
pub mod f0;
pub mod feedback;
pub mod scoring;
pub mod types;
pub mod utils;

pub use types::{
    F0Result, FeedbackItem, NoteEvent, PhraseResult, PitchData, RpaResult, ScoringOptions,
    ScoringResult, SignalQuality, WeightedRpaResult,
};

pub use f0::{compute_confidence_weighted_rpa, compute_rpa, hz_to_cent};
pub use dtw::{constrained_dtw, hierarchical_dtw, segment_phrases};
pub use scoring::{
    adaptive_tolerance, assess_signal_quality, compute_singing_score, estimate_passaggio,
    perceptual_tolerance, score_singing,
};
pub use feedback::generate_attribute_feedback;
