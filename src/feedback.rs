use crate::f0::hz_to_cent_default;
use crate::types::{F0Result, FeedbackItem, PhraseResult};
use crate::utils::{polyfit_3, polyval_3, rfft_magnitudes, std_dev};

pub fn generate_attribute_feedback(
    user_f0: &F0Result,
    pitch_score: f64,
    phrase_results: &[PhraseResult],
    hop_sec: f64,
) -> Vec<FeedbackItem> {
    let mut feedbacks = Vec::new();

    let voiced_cent: Vec<f64> = user_f0
        .frequencies
        .iter()
        .zip(user_f0.voicing.iter())
        .filter(|&(_, &v)| v)
        .map(|(&f, _)| f)
        .collect();
    let voiced_cent = hz_to_cent_default(&voiced_cent);

    if voiced_cent.len() < 100 {
        return feedbacks;
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

    // Vibrato analysis (4-8 Hz)
    let win_frames = (2.0 / hop_sec) as usize;
    let half_step = win_frames / 2;
    let mut i = 0;
    while i + win_frames <= residual.len() {
        let chunk = &residual[i..i + win_frames];
        let (fft_vals, freqs) = rfft_magnitudes(chunk, hop_sec);

        if fft_vals.len() > 1 {
            let mut vib_max = 0.0f64;
            let mut vib_peak_freq = 0.0f64;
            let mut any_vib = false;

            let total_energy: f64 =
                fft_vals[1..].iter().sum::<f64>() / (fft_vals.len() - 1).max(1) as f64;

            for (k, (&mag, &freq)) in fft_vals.iter().zip(freqs.iter()).enumerate() {
                if k == 0 {
                    continue;
                }
                if (4.0..=8.0).contains(&freq) {
                    any_vib = true;
                    if mag > vib_max {
                        vib_max = mag;
                        vib_peak_freq = freq;
                    }
                }
            }

            if any_vib {
                let vib_amp = std_dev(chunk) * 2.0;
                let time_sec = i as f64 * hop_sec;

                if vib_max > total_energy * 2.0 && (20.0..=80.0).contains(&vib_amp) {
                    if (5.5..=6.5).contains(&vib_peak_freq) {
                        feedbacks.push(FeedbackItem {
                            feedback_type: "vibrato_pro".to_string(),
                            time_sec: (time_sec * 10.0).round() / 10.0,
                            message: format!(
                                "在 {:.1}s 处的颤音频率非常稳定 ({:.1}Hz), 接近专业水准",
                                time_sec, vib_peak_freq
                            ),
                            confidence: 0.9,
                        });
                    } else if vib_amp > 40.0 {
                        feedbacks.push(FeedbackItem {
                            feedback_type: "vibrato_expressive".to_string(),
                            time_sec: (time_sec * 10.0).round() / 10.0,
                            message: format!(
                                "在 {:.1}s 处的颤音幅度饱满, 表现力很强",
                                time_sec
                            ),
                            confidence: 0.8,
                        });
                    }
                }
            }
        }

        i += half_step;
    }

    // Portamento analysis
    let mut portamento_count = 0usize;
    for i in 0..voiced_cent.len().saturating_sub(1) {
        let diff = (voiced_cent[i + 1] - voiced_cent[i]).abs();
        if diff > 50.0 && diff < 300.0 {
            portamento_count += 1;
            if portamento_count <= 3 {
                let time_sec = i as f64 * hop_sec;
                feedbacks.push(FeedbackItem {
                    feedback_type: "portamento".to_string(),
                    time_sec: (time_sec * 10.0).round() / 10.0,
                    message: format!(
                        "在 {:.1}s 处的滑音处理得很顺滑, 给歌曲增添了细腻感",
                        time_sec
                    ),
                    confidence: 0.7,
                });
            }
        }
    }

    if portamento_count > 5 {
        let total_phrases = if phrase_results.is_empty() {
            10
        } else {
            phrase_results.len()
        };
        let pct = ((portamento_count as f64 / total_phrases as f64 * 100.0) as usize).min(100);
        feedbacks.push(FeedbackItem {
            feedback_type: "portamento_summary".to_string(),
            time_sec: 0.0,
            message: format!("你捕捉到了原唱中约 {}% 的滑音细节", pct),
            confidence: 0.6,
        });
    }

    // Sustained note stability
    let win = 50;
    let mut stable_count = 0usize;
    let mut idx = 0;
    while idx + win <= voiced_cent.len() {
        let chunk = &voiced_cent[idx..idx + win];
        if std_dev(chunk) < 8.0 {
            stable_count += 1;
        }
        idx += win;
    }

    if stable_count >= 4 {
        feedbacks.push(FeedbackItem {
            feedback_type: "stability".to_string(),
            time_sec: 0.0,
            message: format!("多段长音的稳定性令人印象深刻 (共 {} 段)", stable_count),
            confidence: 0.85,
        });
    }

    if pitch_score > 85.0 {
        feedbacks.push(FeedbackItem {
            feedback_type: "pitch_excellence".to_string(),
            time_sec: 0.0,
            message: "整体音准控制非常出色, 大部分音符都精准命中".to_string(),
            confidence: 0.9,
        });
    }

    feedbacks.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap_or(std::cmp::Ordering::Equal));
    feedbacks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stable_f0(n: usize, freq: f64) -> F0Result {
        F0Result {
            timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
            frequencies: vec![freq; n],
            confidences: vec![1.0; n],
            voicing: vec![true; n],
        }
    }

    #[test]
    fn test_feedback_stable_singer() {
        let f0 = make_stable_f0(1000, 440.0);
        let fb = generate_attribute_feedback(&f0, 90.0, &[], 0.01);
        let has_stability = fb.iter().any(|f| f.feedback_type == "stability");
        let has_pitch_exc = fb.iter().any(|f| f.feedback_type == "pitch_excellence");
        assert!(has_stability, "Expected stability feedback");
        assert!(has_pitch_exc, "Expected pitch excellence feedback");
    }

    #[test]
    fn test_feedback_too_short() {
        let f0 = make_stable_f0(50, 440.0);
        let fb = generate_attribute_feedback(&f0, 60.0, &[], 0.01);
        assert!(fb.is_empty(), "Short signal should produce no feedback");
    }

    #[test]
    fn test_feedback_chinese_utf8() {
        let f0 = make_stable_f0(1000, 440.0);
        let fb = generate_attribute_feedback(&f0, 90.0, &[], 0.01);
        for item in &fb {
            assert!(!item.message.is_empty());
            let _ = item.message.as_bytes();
        }
    }
}
