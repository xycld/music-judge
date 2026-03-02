use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Result {
    pub timestamps: Vec<f64>,
    /// Frequencies in Hz, 0 = unvoiced
    pub frequencies: Vec<f64>,
    /// Confidence values in [0, 1]
    pub confidences: Vec<f64>,
    pub voicing: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteEvent {
    pub onset: f64,
    pub offset: f64,
    pub pitch_hz: f64,
    pub midi_note: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseResult {
    pub phrase_idx: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_range: Option<(usize, usize)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_range: Option<(usize, usize)>,
    pub cost: f64,
    pub status: String,
    pub path_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackItem {
    #[serde(rename = "type")]
    pub feedback_type: String,
    pub time_sec: f64,
    pub message: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchData {
    pub ref_cents: Vec<f64>,
    pub user_cents: Vec<f64>,
    pub timestamps: Vec<f64>,
    pub cent_diffs: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResult {
    pub pitch_score: f64,
    pub rhythm_score: f64,
    pub stability_score: f64,
    pub expression_score: f64,
    pub total_score: f64,
    #[serde(default)]
    pub dtw_cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passaggio_hz: Option<f64>,
    #[serde(default)]
    pub phrase_results: Vec<PhraseResult>,
    #[serde(default)]
    pub feedback: Vec<FeedbackItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch_data: Option<PitchData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpaResult {
    pub rpa: f64,
    pub rca: f64,
    pub voicing_recall: f64,
    pub voicing_false_alarm: f64,
    pub n_voiced_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedRpaResult {
    pub rpa_weighted: f64,
    pub rpa_unweighted: f64,
    pub mean_confidence: f64,
    pub n_voiced_frames: usize,
}

#[derive(Debug, Clone)]
pub struct ScoringOptions {
    pub sr: usize,
    pub dtw_window: usize,
    pub pitch_zero_cent: f64,
    pub use_hierarchical: bool,
    pub downsample: usize,
    pub max_pitch_points: usize,
}

impl Default for ScoringOptions {
    fn default() -> Self {
        Self {
            sr: 16000,
            dtw_window: 200,
            pitch_zero_cent: 300.0,
            use_hierarchical: true,
            downsample: 5,
            max_pitch_points: 500,
        }
    }
}
