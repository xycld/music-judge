use crate::dtw::{constrained_dtw, hierarchical_dtw};
use crate::f0::hz_to_cent_default;
use crate::feedback::generate_attribute_feedback;
use crate::types::{F0Result, PitchData, ScoringOptions, ScoringResult, SignalQuality};
use crate::utils::{mean, median, polyfit_3, polyval_3, rfft_magnitudes, std_dev, variance};

/// Frequency-dependent perceptual tolerance (cents) based on Zwicker & Fastl.
/// Low (<500 Hz): 50 cents, Mid (500-2000 Hz): 25 cents, High (>2000 Hz): 40 cents.
pub fn perceptual_tolerance(freq_hz: f64) -> f64 {
    if freq_hz < 500.0 {
        50.0
    } else if freq_hz < 2000.0 {
        25.0
    } else {
        40.0
    }
}

/// Sliding-window std deviation to detect vocal break point (passaggio).
/// Looks for F0 jitter peaks in the 200-800 Hz range.
pub fn estimate_passaggio(
    f0_series: &[f64],
    confidence: &[f64],
    window_frames: usize,
) -> Option<f64> {
    let valid_f0: Vec<f64> = f0_series
        .iter()
        .zip(confidence.iter())
        .filter(|&(&f, &c)| c > 0.5 && f > 0.0)
        .map(|(&f, _)| f)
        .collect();

    if valid_f0.len() < window_frames * 3 {
        return None;
    }

    let half_w = window_frames / 2;
    let mut local_stds = vec![0.0f64; valid_f0.len()];
    for i in half_w..valid_f0.len().saturating_sub(half_w) {
        let chunk = &valid_f0[i - half_w..i + half_w];
        local_stds[i] = std_dev(chunk);
    }

    // Only consider 200-800 Hz range
    let candidates: Vec<f64> = local_stds
        .iter()
        .zip(valid_f0.iter())
        .map(|(&s, &f)| if f > 200.0 && f < 800.0 { s } else { 0.0 })
        .collect();

    let max_candidate = candidates.iter().cloned().fold(0.0f64, f64::max);

    if max_candidate < 5.0 {
        return None;
    }

    let peak_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;

    Some(valid_f0[peak_idx])
}

/// JND base + passaggio bonus, hard cap 65 cents.
pub fn adaptive_tolerance(freq_hz: f64, passaggio_hz: Option<f64>) -> f64 {
    let mut base = perceptual_tolerance(freq_hz);

    if let Some(p) = passaggio_hz {
        let dist = (freq_hz - p).abs();
        if dist < 50.0 {
            base += 15.0 * (1.0 - dist / 50.0);
        }
    }

    base.min(65.0)
}

fn detect_expression(
    user_f0: &F0Result,
    vibrato_freq_range: (f64, f64),
    vibrato_amp_range_cent: (f64, f64),
    hop_sec: f64,
) -> f64 {
    let voiced_cent: Vec<f64> = user_f0
        .frequencies
        .iter()
        .zip(user_f0.voicing.iter())
        .filter(|&(_, &v)| v)
        .map(|(&f, _)| f)
        .collect();
    let voiced_cent = hz_to_cent_default(&voiced_cent);

    if voiced_cent.len() < 50 {
        return 50.0;
    }

    // Detrend with degree-3 polynomial
    let x: Vec<f64> = (0..voiced_cent.len()).map(|i| i as f64).collect();
    let coefs = polyfit_3(&x, &voiced_cent);
    let trend = polyval_3(&coefs, &x);
    let residual: Vec<f64> = voiced_cent
        .iter()
        .zip(trend.iter())
        .map(|(&v, &t)| v - t)
        .collect();

    let (fft_vals, freqs) = rfft_magnitudes(&residual, hop_sec);

    let mut vib_max = 0.0f64;
    let mut any_vib = false;
    for (&mag, &freq) in fft_vals.iter().zip(freqs.iter()) {
        if (vibrato_freq_range.0..=vibrato_freq_range.1).contains(&freq) {
            any_vib = true;
            if mag > vib_max {
                vib_max = mag;
            }
        }
    }

    if !any_vib {
        return 50.0;
    }

    let total_energy = if fft_vals.len() > 1 {
        mean(&fft_vals[1..])
    } else {
        1.0
    };
    let vibrato_amp = std_dev(&residual) * 2.0;

    let mut score = 50.0;

    if vib_max > total_energy * 1.5 {
        if vibrato_amp >= vibrato_amp_range_cent.0 && vibrato_amp <= vibrato_amp_range_cent.1 {
            score += 30.0;
        } else if vibrato_amp < vibrato_amp_range_cent.0 {
            score += 10.0;
        }
    }

    // Portamento: large smooth transitions
    let mut large_transitions = 0usize;
    for i in 0..voiced_cent.len().saturating_sub(1) {
        let diff = (voiced_cent[i + 1] - voiced_cent[i]).abs();
        if diff > 50.0 && diff < 300.0 {
            large_transitions += 1;
        }
    }
    if large_transitions > 0 {
        score += (large_transitions as f64 * 2.0).min(20.0);
    }

    score.clamp(0.0, 100.0)
}

/// Assess input signal quality from F0 data to detect noise, silence, or non-singing input.
///
/// Returns a `SignalQuality` with a `quality_multiplier` in [0, 1] that dampens scores
/// for unreliable input. For genuine singing: voiced_ratio > 0.3, mean_confidence > 0.5,
/// pitch_coherence > 0.4. Noise/silence fails on all three.
pub fn assess_signal_quality(user_f0: &F0Result) -> SignalQuality {
    let n = user_f0.frequencies.len();
    if n == 0 {
        return SignalQuality {
            voiced_ratio: 0.0,
            mean_confidence: 0.0,
            pitch_coherence: 0.0,
            quality_multiplier: 0.0,
            is_valid: false,
        };
    }

    // 1) Voiced ratio: what fraction of frames are voiced?
    let n_voiced = user_f0.voicing.iter().filter(|&&v| v).count();
    let voiced_ratio = n_voiced as f64 / n as f64;

    // 2) Mean confidence of voiced frames
    let voiced_confidences: Vec<f64> = user_f0
        .confidences
        .iter()
        .zip(user_f0.voicing.iter())
        .filter(|&(_, &v)| v)
        .map(|(&c, _)| c)
        .collect();
    let mean_confidence = if voiced_confidences.is_empty() {
        0.0
    } else {
        mean(&voiced_confidences)
    };

    // 3) Pitch coherence: fraction of consecutive voiced frames with |Δcent| < 200
    //    Real singing has smooth pitch; noise has random jumps.
    let voiced_freqs: Vec<f64> = user_f0
        .frequencies
        .iter()
        .zip(user_f0.voicing.iter())
        .filter(|&(_, &v)| v)
        .map(|(&f, _)| f)
        .collect();
    let pitch_coherence = if voiced_freqs.len() < 2 {
        0.0
    } else {
        let voiced_cents = hz_to_cent_default(&voiced_freqs);
        let n_transitions = voiced_cents.len() - 1;
        let n_smooth = voiced_cents
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() < 200.0)
            .count();
        n_smooth as f64 / n_transitions as f64
    };

    // Combined quality multiplier — each factor in [0, 1], multiplied together
    // with individual sigmoid-like ramps so there's a smooth transition.
    let vr_factor = ((voiced_ratio - 0.10) / 0.20).clamp(0.0, 1.0); // ramp 0.10→0.30
    let mc_factor = ((mean_confidence - 0.20) / 0.30).clamp(0.0, 1.0); // ramp 0.20→0.50
    let pc_factor = ((pitch_coherence - 0.20) / 0.30).clamp(0.0, 1.0); // ramp 0.20→0.50

    let quality_multiplier = vr_factor * mc_factor * pc_factor;

    // Binary gate: need at least voiced_ratio > 0.15 AND mean_confidence > 0.25
    let is_valid = voiced_ratio > 0.15 && mean_confidence > 0.25;

    SignalQuality {
        voiced_ratio: (voiced_ratio * 1000.0).round() / 1000.0,
        mean_confidence: (mean_confidence * 1000.0).round() / 1000.0,
        pitch_coherence: (pitch_coherence * 1000.0).round() / 1000.0,
        quality_multiplier: (quality_multiplier * 1000.0).round() / 1000.0,
        is_valid,
    }
}

/// Four dimensions: pitch (40%), rhythm (25%), stability (15%), expression (20%).
/// Total score mapped to [40, 100], then dampened by signal quality.
pub fn compute_singing_score(
    ref_f0: &F0Result,
    user_f0: &F0Result,
    ref_audio: Option<&[f64]>,
    user_audio: Option<&[f64]>,
    opts: &ScoringOptions,
) -> ScoringResult {
    let ds = opts.downsample;

    let passaggio = estimate_passaggio(&user_f0.frequencies, &user_f0.confidences, 50);

    let (path, dtw_cost, phrase_results) = if opts.use_hierarchical && ref_audio.is_some() {
        hierarchical_dtw(ref_f0, user_f0, ref_audio, user_audio, opts.sr, ds, 0.15)
    } else {
        let ref_cent = hz_to_cent_default(&ref_f0.frequencies);
        let user_cent = hz_to_cent_default(&user_f0.frequencies);
        let ref_voiced: Vec<f64> = ref_cent
            .iter()
            .zip(ref_f0.voicing.iter())
            .map(|(&c, &v)| if v { c } else { 0.0 })
            .collect();
        let user_voiced: Vec<f64> = user_cent
            .iter()
            .zip(user_f0.voicing.iter())
            .map(|(&c, &v)| if v { c } else { 0.0 })
            .collect();
        let ref_ds: Vec<f64> = ref_voiced.iter().step_by(ds).copied().collect();
        let user_ds: Vec<f64> = user_voiced.iter().step_by(ds).copied().collect();
        let (path, cost) = constrained_dtw(&ref_ds, &user_ds, opts.dtw_window / ds);
        (path, cost, vec![])
    };

    // Pitch Score (40%) — octave-invariant chroma distance + adaptive tolerance
    let ref_cent_full = hz_to_cent_default(&ref_f0.frequencies);
    let user_cent_full = hz_to_cent_default(&user_f0.frequencies);
    let ref_voiced_cent: Vec<f64> = ref_cent_full
        .iter()
        .zip(ref_f0.voicing.iter())
        .map(|(&c, &v)| if v { c } else { 0.0 })
        .collect();
    let user_voiced_cent: Vec<f64> = user_cent_full
        .iter()
        .zip(user_f0.voicing.iter())
        .map(|(&c, &v)| if v { c } else { 0.0 })
        .collect();
    let ref_ds: Vec<f64> = ref_voiced_cent.iter().step_by(ds).copied().collect();
    let user_ds: Vec<f64> = user_voiced_cent.iter().step_by(ds).copied().collect();

    let mut cent_diffs = Vec::new();
    let mut tolerances = Vec::new();
    let mut pitch_ref_cents = Vec::new();
    let mut pitch_user_cents = Vec::new();
    let mut pitch_timestamps = Vec::new();
    let mut pitch_cent_diffs = Vec::new();

    for &(ri, ui) in &path {
        if ri < ref_ds.len() && ui < user_ds.len()
            && ref_ds[ri] != 0.0 && user_ds[ui] != 0.0
        {
            let raw_diff = (ref_ds[ri] - user_ds[ui]).abs();
            let chroma_diff = raw_diff % 1200.0;
            let diff = chroma_diff.min(1200.0 - chroma_diff);
            cent_diffs.push(diff);

            let user_freq_idx = (ui * ds).min(user_f0.frequencies.len() - 1);
            let freq_hz = user_f0.frequencies[user_freq_idx];
            if freq_hz > 0.0 {
                tolerances.push(adaptive_tolerance(freq_hz, passaggio));
            } else {
                tolerances.push(50.0);
            }

            let ref_ts_idx = (ri * ds).min(ref_f0.timestamps.len() - 1);
            pitch_ref_cents.push(ref_ds[ri]);
            pitch_user_cents.push(user_ds[ui]);
            pitch_timestamps.push(ref_f0.timestamps[ref_ts_idx]);
            pitch_cent_diffs.push(diff);
        }
    }

    let pitch_score = if !cent_diffs.is_empty() {
        let frame_scores: Vec<f64> = cent_diffs
            .iter()
            .zip(tolerances.iter())
            .map(|(&d, &t)| {
                (100.0 * (1.0 - (d - t).max(0.0) / (opts.pitch_zero_cent - t + 1e-6)))
                    .clamp(0.0, 100.0)
            })
            .collect();
        mean(&frame_scores)
    } else {
        0.0
    };

    // Rhythm Score (25%) — windowed local tempo deviation
    let rhythm_score = if path.len() > 1 {
        let window = 50usize;
        let stride = window / 2;
        let mut rhythm_segments = Vec::new();
        let mut i = 0;
        while i + window <= path.len() {
            let chunk_start = &path[i];
            let chunk_end = &path[i + window - 1];
            let ref_span = chunk_end.0 as f64 - chunk_start.0 as f64;
            let user_span = chunk_end.1 as f64 - chunk_start.1 as f64;
            if ref_span > 0.0 && user_span > 0.0 {
                let local_slope = user_span / ref_span;
                let deviation = (local_slope - 1.0).abs();
                rhythm_segments.push((100.0 * (1.0 - deviation * 2.0)).clamp(0.0, 100.0));
            }
            i += stride;
        }
        if rhythm_segments.is_empty() {
            50.0
        } else {
            mean(&rhythm_segments)
        }
    } else {
        50.0
    };

    // Stability Score (15%) — median variance (robust to note transitions)
    let user_cent_all = hz_to_cent_default(&user_f0.frequencies);
    let stability_score = if user_cent_all.len() > 10 {
        let voiced_cent: Vec<f64> = user_cent_all
            .iter()
            .zip(user_f0.voicing.iter())
            .filter(|&(_, &v)| v)
            .map(|(&c, _)| c)
            .collect();
        let win = 5;
        let mut local_vars = Vec::new();
        let mut i = 0;
        while i + win <= voiced_cent.len() {
            let chunk = &voiced_cent[i..i + win];
            local_vars.push(variance(chunk));
            i += win;
        }
        if local_vars.is_empty() {
            50.0
        } else {
            let median_var = median(&local_vars);
            (100.0 * (1.0 - median_var / 2500.0)).clamp(0.0, 100.0)
        }
    } else {
        50.0
    };

    // Expression Score (20%)
    let expression_score = detect_expression(user_f0, (4.0, 8.0), (20.0, 80.0), 0.01);

    // Signal quality gate — suppress scores for noise/silence/non-singing
    let quality = assess_signal_quality(user_f0);
    let qm = quality.quality_multiplier;

    // Apply quality multiplier to each dimension.
    // For invalid signals, scores collapse toward 0 instead of defaulting to 50.
    let pitch_final = pitch_score * qm;
    let rhythm_final = rhythm_score * qm;
    let stability_final = stability_score * qm;
    let expression_final = expression_score * qm;

    // Weighted total → [40, 100] only for valid signals; invalid → [0, ~40]
    let total = pitch_final * 0.40
        + rhythm_final * 0.25
        + stability_final * 0.15
        + expression_final * 0.20;

    // Floor of 40 only applies when the signal is valid singing
    let total_mapped = if quality.is_valid {
        40.0 + total * 0.6
    } else {
        // No floor — garbage input gets the raw (near-zero) weighted total
        total
    };

    // Downsample pitch data for visualization
    let max_points = opts.max_pitch_points;
    let (p_ref, p_user, p_ts, p_diffs) = if pitch_timestamps.len() > max_points {
        let step = pitch_timestamps.len() / max_points;
        (
            pitch_ref_cents.iter().step_by(step).take(max_points).copied().collect(),
            pitch_user_cents.iter().step_by(step).take(max_points).copied().collect(),
            pitch_timestamps.iter().step_by(step).take(max_points).copied().collect(),
            pitch_cent_diffs.iter().step_by(step).take(max_points).copied().collect(),
        )
    } else {
        (pitch_ref_cents, pitch_user_cents, pitch_timestamps, pitch_cent_diffs)
    };

    let feedback = generate_attribute_feedback(user_f0, pitch_final, &phrase_results, 0.01);

    ScoringResult {
        pitch_score: (pitch_final * 10.0).round() / 10.0,
        rhythm_score: (rhythm_final * 10.0).round() / 10.0,
        stability_score: (stability_final * 10.0).round() / 10.0,
        expression_score: (expression_final * 10.0).round() / 10.0,
        total_score: (total_mapped * 10.0).round() / 10.0,
        dtw_cost: (dtw_cost * 100.0).round() / 100.0,
        passaggio_hz: passaggio.map(|p| (p * 10.0).round() / 10.0),
        phrase_results,
        feedback,
        pitch_data: Some(PitchData {
            ref_cents: p_ref,
            user_cents: p_user,
            timestamps: p_ts,
            cent_diffs: p_diffs,
        }),
        signal_quality: Some(quality),
    }
}

pub fn score_singing(
    ref_f0: &F0Result,
    user_f0: &F0Result,
    ref_audio: Option<&[f64]>,
    user_audio: Option<&[f64]>,
) -> ScoringResult {
    compute_singing_score(ref_f0, user_f0, ref_audio, user_audio, &ScoringOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_tolerance_ranges() {
        assert!((perceptual_tolerance(200.0) - 50.0).abs() < 1e-10);
        assert!((perceptual_tolerance(1000.0) - 25.0).abs() < 1e-10);
        assert!((perceptual_tolerance(3000.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_tolerance_no_passaggio() {
        assert!((adaptive_tolerance(440.0, None) - 50.0).abs() < 1e-10);
        assert!((adaptive_tolerance(1000.0, None) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_tolerance_with_passaggio() {
        // 1000 Hz mid-band (25) + passaggio bonus (15) = 40
        let tol = adaptive_tolerance(1000.0, Some(1000.0));
        assert!((tol - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_tolerance_cap() {
        // Low freq (50 base) + passaggio bonus (15) = 65 (at cap)
        let tol = adaptive_tolerance(200.0, Some(200.0));
        assert!((tol - 65.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_passaggio_none_for_stable() {
        let f0 = vec![440.0; 500];
        let conf = vec![1.0; 500];
        assert!(estimate_passaggio(&f0, &conf, 50).is_none());
    }

    #[test]
    fn test_score_perfect_match() {
        let n = 500;
        let f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        let result = score_singing(&f0, &f0, None, None);
        assert!(result.pitch_score > 90.0, "pitch={}", result.pitch_score);
        assert!(result.total_score >= 40.0);
        assert!(result.total_score <= 100.0);
    }

    #[test]
    fn test_score_total_range() {
        let n = 500;
        let ref_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        let user_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![220.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        let result = score_singing(&ref_f0, &user_f0, None, None);
        // Valid singing input still gets floor of 40
        assert!(result.total_score >= 40.0, "total={}", result.total_score);
        assert!(result.total_score <= 100.0, "total={}", result.total_score);
        assert!(result.signal_quality.as_ref().unwrap().is_valid);
    }

    #[test]
    fn test_score_json_serialization() {
        let n = 200;
        let f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        let result = score_singing(&f0, &f0, None, None);
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("pitch_score"));
        assert!(json.contains("signal_quality"));
        let parsed: ScoringResult = serde_json::from_str(&json).unwrap();
        assert!((parsed.pitch_score - result.pitch_score).abs() < 1e-10);
    }

    /// Noise input: all frames unvoiced, zero confidence → score near 0
    #[test]
    fn test_noise_rejected() {
        let n = 500;
        let ref_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        // Simulate noise: F0 extractor marks everything unvoiced with zero confidence
        let noise_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![0.0; n],
            confidences: vec![0.0; n],
            voicing: vec![false; n],
        };
        let result = score_singing(&ref_f0, &noise_f0, None, None);
        let q = result.signal_quality.as_ref().unwrap();
        assert!(!q.is_valid, "Noise should fail quality gate");
        assert!(q.voiced_ratio < 0.01, "voiced_ratio={}", q.voiced_ratio);
        assert!(
            result.total_score < 10.0,
            "Noise total_score should be <10, got {}",
            result.total_score
        );
    }

    /// Sparse voicing with low confidence: simulates noisy F0 extraction
    #[test]
    fn test_sparse_low_confidence_rejected() {
        let n = 500;
        let ref_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        // 10% voiced, low confidence, random-ish pitches
        let mut freqs = vec![0.0; n];
        let mut voicing = vec![false; n];
        let mut confidences = vec![0.05; n];
        for i in (0..n).step_by(10) {
            freqs[i] = 200.0 + (i as f64 * 37.0) % 600.0; // pseudo-random pitches
            voicing[i] = true;
            confidences[i] = 0.15;
        }
        let noisy_f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: freqs,
            confidences,
            voicing,
        };
        let result = score_singing(&ref_f0, &noisy_f0, None, None);
        let q = result.signal_quality.as_ref().unwrap();
        assert!(!q.is_valid, "Sparse noisy F0 should fail quality gate");
        assert!(
            result.total_score < 20.0,
            "Sparse noisy total should be <20, got {}",
            result.total_score
        );
    }

    /// Valid singing still scores well after the quality gate
    #[test]
    fn test_quality_gate_passes_real_singing() {
        let n = 500;
        let f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![0.9; n],
            voicing: vec![true; n],
        };
        let result = score_singing(&f0, &f0, None, None);
        let q = result.signal_quality.as_ref().unwrap();
        assert!(q.is_valid, "Good singing should pass quality gate");
        assert!(
            q.quality_multiplier > 0.8,
            "Good singing qm should be high, got {}",
            q.quality_multiplier
        );
        assert!(
            result.total_score > 70.0,
            "Perfect match should score >70, got {}",
            result.total_score
        );
    }

    /// Quality assessment function directly
    #[test]
    fn test_assess_signal_quality_empty() {
        let f0 = F0Result {
            timestamps: vec![],
            frequencies: vec![],
            confidences: vec![],
            voicing: vec![],
        };
        let q = assess_signal_quality(&f0);
        assert!(!q.is_valid);
        assert!(q.quality_multiplier < 0.001);
    }

    #[test]
    fn test_assess_signal_quality_perfect() {
        let n = 500;
        let f0 = F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; n],
            confidences: vec![0.95; n],
            voicing: vec![true; n],
        };
        let q = assess_signal_quality(&f0);
        assert!(q.is_valid);
        assert!(q.voiced_ratio > 0.99);
        assert!(q.mean_confidence > 0.9);
        assert!(q.pitch_coherence > 0.99);
        assert!(q.quality_multiplier > 0.95);
    }
}
