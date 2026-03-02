use crate::f0::hz_to_cent_default;
use crate::types::{F0Result, PhraseResult};

/// Sakoe-Chiba constrained DTW.
/// Returns `(path, normalized_cost)` where path is a vec of `(ref_idx, user_idx)` pairs.
pub fn constrained_dtw(ref_cent: &[f64], user_cent: &[f64], window: usize) -> (Vec<(usize, usize)>, f64) {
    let m = ref_cent.len();
    let n = user_cent.len();

    if m == 0 || n == 0 {
        return (vec![], 0.0);
    }

    let inf: f64 = 1e18;
    let w = window.max(m.abs_diff(n));

    let mut d = vec![vec![inf; n + 1]; m + 1];
    d[0][0] = 0.0;

    for i in 1..=m {
        let j_start = if i > w { i - w } else { 1 };
        let j_end = (i + w).min(n);
        for j in j_start..=j_end {
            let cost = (ref_cent[i - 1] - user_cent[j - 1]).abs();
            let prev = d[i - 1][j].min(d[i][j - 1]).min(d[i - 1][j - 1]);
            d[i][j] = cost + prev;
        }
    }

    let mut path = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = d[i - 1][j - 1];
        let up = d[i - 1][j];
        let left = d[i][j - 1];
        let min_val = diag.min(up).min(left);
        if (diag - min_val).abs() < 1e-15 {
            i -= 1;
            j -= 1;
        } else if (up - min_val).abs() < 1e-15 {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    path.reverse();
    let normalized_cost = if !path.is_empty() {
        d[m][n] / path.len() as f64
    } else {
        0.0
    };

    (path, normalized_cost)
}

/// Phrase segmentation based on energy valleys + silence detection.
pub fn segment_phrases(
    f0: &F0Result,
    audio: &[f64],
    sr: usize,
    min_silence_sec: f64,
    energy_threshold_db: f64,
    hop_sec: f64,
) -> Vec<(usize, usize)> {
    let hop_samples = (sr as f64 * hop_sec) as usize;
    let n_frames = f0.frequencies.len();

    let mut frame_energy = vec![0.0f64; n_frames];
    for (i, energy) in frame_energy.iter_mut().enumerate().take(n_frames) {
        let start = i * hop_samples;
        let end = (start + hop_samples).min(audio.len());
        if start < audio.len() {
            let frame = &audio[start..end];
            let rms = (frame.iter().map(|&s| s * s).sum::<f64>() / frame.len().max(1) as f64
                + 1e-10)
                .sqrt();
            *energy = 20.0 * (rms + 1e-10).log10();
        }
    }

    let is_silent: Vec<bool> = frame_energy.iter().map(|&e| e < energy_threshold_db).collect();
    let min_silence_frames = (min_silence_sec / hop_sec) as usize;

    let mut phrases = Vec::new();
    let mut phrase_start = 0usize;
    let mut silent_count = 0usize;

    for (i, &silent) in is_silent.iter().enumerate().take(n_frames) {
        if silent {
            silent_count += 1;
        } else {
            if silent_count >= min_silence_frames && i > phrase_start + min_silence_frames {
                let phrase_end = i - silent_count;
                if phrase_end > phrase_start {
                    phrases.push((phrase_start, phrase_end));
                }
                phrase_start = i;
            }
            silent_count = 0;
        }
    }

    if phrase_start < n_frames.saturating_sub(1) {
        phrases.push((phrase_start, n_frames - 1));
    }

    if phrases.is_empty() {
        phrases.push((0, n_frames.saturating_sub(1)));
    }

    phrases
}

/// Hierarchical DTW: phrase-level segmentation + per-phrase constrained DTW.
/// Falls back to fixed-size chunking when `ref_audio` is `None`.
pub fn hierarchical_dtw(
    ref_f0: &F0Result,
    user_f0: &F0Result,
    ref_audio: Option<&[f64]>,
    _user_audio: Option<&[f64]>,
    sr: usize,
    downsample: usize,
    window_ratio: f64,
) -> (Vec<(usize, usize)>, f64, Vec<PhraseResult>) {
    let ref_cent = hz_to_cent_default(&ref_f0.frequencies);
    let user_cent = hz_to_cent_default(&user_f0.frequencies);

    // Voicing mask: unvoiced → 0
    let ref_voiced_cent: Vec<f64> = ref_cent
        .iter()
        .zip(ref_f0.voicing.iter())
        .map(|(&c, &v)| if v { c } else { 0.0 })
        .collect();
    let user_voiced_cent: Vec<f64> = user_cent
        .iter()
        .zip(user_f0.voicing.iter())
        .map(|(&c, &v)| if v { c } else { 0.0 })
        .collect();

    let ref_ds: Vec<f64> = ref_voiced_cent.iter().step_by(downsample).copied().collect();
    let user_ds: Vec<f64> = user_voiced_cent.iter().step_by(downsample).copied().collect();

    let phrases_ds: Vec<(usize, usize)> = if let Some(audio) = ref_audio {
        let phrases = segment_phrases(ref_f0, audio, sr, 0.3, -40.0, 0.01);
        phrases
            .iter()
            .map(|&(s, e)| (s / downsample, (e / downsample).min(ref_ds.len().saturating_sub(1))))
            .collect()
    } else {
        let chunk_size = 100;
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < ref_ds.len() {
            let end = (i + chunk_size).min(ref_ds.len().saturating_sub(1));
            if end > i {
                chunks.push((i, end));
            }
            i += chunk_size;
        }
        if chunks.is_empty() {
            chunks.push((0, ref_ds.len().saturating_sub(1)));
        }
        chunks
    };

    let scale = user_ds.len() as f64 / ref_ds.len().max(1) as f64;

    let mut full_path: Vec<(usize, usize)> = Vec::new();
    let mut phrase_results = Vec::new();
    let mut total_cost = 0.0;
    let mut total_path_len = 0usize;

    for (phrase_idx, &(ps, pe)) in phrases_ds.iter().enumerate() {
        let ref_phrase: Vec<f64> = if pe < ref_ds.len() {
            ref_ds[ps..=pe].to_vec()
        } else {
            ref_ds[ps..].to_vec()
        };

        let us = (ps as f64 * scale) as usize;
        let ue = ((pe as f64 * scale) as usize).min(user_ds.len().saturating_sub(1));
        let user_phrase: Vec<f64> = if ue < user_ds.len() && us <= ue {
            user_ds[us..=ue].to_vec()
        } else if us < user_ds.len() {
            user_ds[us..].to_vec()
        } else {
            vec![]
        };

        if ref_phrase.len() < 3 || user_phrase.len() < 3 {
            phrase_results.push(PhraseResult {
                phrase_idx,
                ref_range: None,
                user_range: None,
                cost: 0.0,
                status: "skipped".to_string(),
                path_len: 0,
                reason: Some("too_short".to_string()),
            });
            continue;
        }

        let w = (ref_phrase.len() as f64 * window_ratio) as usize;
        let w = w.max(ref_phrase.len().abs_diff(user_phrase.len()));
        let (path, cost) = constrained_dtw(&ref_phrase, &user_phrase, w);

        let cost_threshold = 80.0;
        let status = if cost < cost_threshold {
            "aligned"
        } else {
            "misaligned"
        };

        if !path.is_empty() {
            let global_path: Vec<(usize, usize)> =
                path.iter().map(|&(r, u)| (r + ps, u + us)).collect();
            total_cost += cost * path.len() as f64;
            total_path_len += path.len();
            full_path.extend(global_path);
        }

        phrase_results.push(PhraseResult {
            phrase_idx,
            ref_range: Some((ps * downsample, pe * downsample)),
            user_range: Some((us * downsample, ue * downsample)),
            cost: (cost * 100.0).round() / 100.0,
            status: status.to_string(),
            path_len: path.len(),
            reason: None,
        });
    }

    let avg_cost = if total_path_len > 0 {
        total_cost / total_path_len as f64
    } else {
        0.0
    };

    (full_path, avg_cost, phrase_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrained_dtw_identical() {
        let seq = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let (path, cost) = constrained_dtw(&seq, &seq, 5);
        assert_eq!(path.len(), 5);
        assert!(cost.abs() < 1e-10);
        for (i, &(r, u)) in path.iter().enumerate() {
            assert_eq!(r, i);
            assert_eq!(u, i);
        }
    }

    #[test]
    fn test_constrained_dtw_shifted() {
        let ref_seq = vec![0.0, 100.0, 200.0, 300.0, 400.0];
        let user_seq = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let (path, cost) = constrained_dtw(&ref_seq, &user_seq, 5);
        assert!(!path.is_empty());
        assert!(cost > 0.0);
        assert!(cost < 200.0);
    }

    #[test]
    fn test_constrained_dtw_different_lengths() {
        let ref_seq = vec![100.0, 200.0, 300.0];
        let user_seq = vec![100.0, 150.0, 200.0, 250.0, 300.0];
        let (path, cost) = constrained_dtw(&ref_seq, &user_seq, 5);
        assert!(!path.is_empty());
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_constrained_dtw_empty() {
        let (path, cost) = constrained_dtw(&[], &[1.0, 2.0], 5);
        assert!(path.is_empty());
        assert!(cost.abs() < 1e-10);
    }

    #[test]
    fn test_segment_phrases_single() {
        let f0 = F0Result {
            timestamps: (0..100).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![440.0; 100],
            confidences: vec![1.0; 100],
            voicing: vec![true; 100],
        };
        let audio = vec![0.5; 16000];
        let phrases = segment_phrases(&f0, &audio, 16000, 0.3, -40.0, 0.01);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_hierarchical_dtw_no_audio() {
        let make_f0 = |n: usize, freq: f64| F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![freq; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        };
        let ref_f0 = make_f0(500, 440.0);
        let user_f0 = make_f0(500, 440.0);
        let (path, cost, phrases) = hierarchical_dtw(&ref_f0, &user_f0, None, None, 16000, 5, 0.15);
        assert!(!path.is_empty());
        assert!(cost.abs() < 1e-6);
        assert!(!phrases.is_empty());
    }
}
