use crate::types::{F0Result, RpaResult, WeightedRpaResult};

/// `cent = 1200 * log2(f0 / ref_hz)`, f0 <= 0 replaced by `ref_hz`.
pub fn hz_to_cent(f0_hz: &[f64], ref_hz: f64) -> Vec<f64> {
    f0_hz
        .iter()
        .map(|&f| {
            let safe = if f > 0.0 { f } else { ref_hz };
            1200.0 * (safe / ref_hz).log2()
        })
        .collect()
}

pub fn hz_to_cent_default(f0_hz: &[f64]) -> Vec<f64> {
    hz_to_cent(f0_hz, 10.0)
}

/// Raw Pitch Accuracy (RPA) and Raw Chroma Accuracy (RCA).
/// RPA: fraction of mutually-voiced frames where |est - ref| < tolerance.
/// RCA: same but with octave folding.
pub fn compute_rpa(ref_f0: &F0Result, est_f0: &F0Result, cent_tolerance: f64) -> RpaResult {
    let n = ref_f0.frequencies.len().min(est_f0.frequencies.len());

    let mut both_voiced_ref = Vec::new();
    let mut both_voiced_est = Vec::new();
    let mut n_ref_voiced: usize = 0;
    let mut n_ref_unvoiced: usize = 0;
    let mut n_est_voiced_and_ref_unvoiced: usize = 0;

    for i in 0..n {
        let rv = ref_f0.voicing[i];
        let ev = est_f0.voicing[i];
        if rv {
            n_ref_voiced += 1;
        } else {
            n_ref_unvoiced += 1;
            if ev {
                n_est_voiced_and_ref_unvoiced += 1;
            }
        }
        if rv && ev {
            both_voiced_ref.push(ref_f0.frequencies[i]);
            both_voiced_est.push(est_f0.frequencies[i]);
        }
    }

    let n_both = both_voiced_ref.len();
    if n_both == 0 {
        return RpaResult {
            rpa: 0.0,
            rca: 0.0,
            voicing_recall: 0.0,
            voicing_false_alarm: 0.0,
            n_voiced_frames: 0,
        };
    }

    let ref_cent = hz_to_cent_default(&both_voiced_ref);
    let est_cent = hz_to_cent_default(&both_voiced_est);

    let cent_diff: Vec<f64> = ref_cent
        .iter()
        .zip(est_cent.iter())
        .map(|(r, e)| (e - r).abs())
        .collect();

    let rpa = cent_diff.iter().filter(|&&d| d < cent_tolerance).count() as f64 / n_both as f64;

    let rca = cent_diff
        .iter()
        .filter(|&&d| {
            let chroma = d % 1200.0;
            chroma.min(1200.0 - chroma) < cent_tolerance
        })
        .count() as f64
        / n_both as f64;

    let voicing_recall = if n_ref_voiced > 0 {
        n_both as f64 / n_ref_voiced as f64
    } else {
        0.0
    };

    let voicing_fa = if n_ref_unvoiced > 0 {
        n_est_voiced_and_ref_unvoiced as f64 / n_ref_unvoiced as f64
    } else {
        0.0
    };

    RpaResult {
        rpa,
        rca,
        voicing_recall,
        voicing_false_alarm: voicing_fa,
        n_voiced_frames: n_both,
    }
}

/// Confidence-weighted RPA: `RPA_w = sum(c_i * 1(|df_i| < d)) / sum(c_i)` where `c = min(c_ref, c_est)`.
pub fn compute_confidence_weighted_rpa(
    ref_f0: &F0Result,
    est_f0: &F0Result,
    cent_tolerance: f64,
) -> WeightedRpaResult {
    let n = ref_f0.frequencies.len().min(est_f0.frequencies.len());

    let mut both_voiced_ref_freq = Vec::new();
    let mut both_voiced_est_freq = Vec::new();
    let mut joint_conf = Vec::new();

    for i in 0..n {
        if ref_f0.voicing[i] && est_f0.voicing[i] {
            both_voiced_ref_freq.push(ref_f0.frequencies[i]);
            both_voiced_est_freq.push(est_f0.frequencies[i]);
            joint_conf.push(ref_f0.confidences[i].min(est_f0.confidences[i]));
        }
    }

    let n_both = both_voiced_ref_freq.len();
    if n_both == 0 {
        return WeightedRpaResult {
            rpa_weighted: 0.0,
            rpa_unweighted: 0.0,
            mean_confidence: 0.0,
            n_voiced_frames: 0,
        };
    }

    let ref_cent = hz_to_cent_default(&both_voiced_ref_freq);
    let est_cent = hz_to_cent_default(&both_voiced_est_freq);

    let cent_diff: Vec<f64> = ref_cent
        .iter()
        .zip(est_cent.iter())
        .map(|(r, e)| (e - r).abs())
        .collect();

    let rpa_unweighted =
        cent_diff.iter().filter(|&&d| d < cent_tolerance).count() as f64 / n_both as f64;

    let conf_sum: f64 = joint_conf.iter().sum();
    let rpa_weighted = if conf_sum > 0.0 {
        cent_diff
            .iter()
            .zip(joint_conf.iter())
            .map(|(&d, &c)| if d < cent_tolerance { c } else { 0.0 })
            .sum::<f64>()
            / conf_sum
    } else {
        0.0
    };

    let mean_conf = conf_sum / n_both as f64;

    WeightedRpaResult {
        rpa_weighted,
        rpa_unweighted,
        mean_confidence: mean_conf,
        n_voiced_frames: n_both,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_cent_basic() {
        let cents = hz_to_cent(&[440.0], 10.0);
        let expected = 1200.0 * (440.0_f64 / 10.0).log2();
        assert!((cents[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hz_to_cent_zero_protection() {
        let cents = hz_to_cent(&[0.0, -5.0], 10.0);
        assert!((cents[0]).abs() < 1e-6);
        assert!((cents[1]).abs() < 1e-6);
    }

    #[test]
    fn test_hz_to_cent_octave() {
        let c1 = hz_to_cent(&[440.0], 10.0);
        let c2 = hz_to_cent(&[880.0], 10.0);
        assert!((c2[0] - c1[0] - 1200.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rpa_perfect() {
        let f0 = F0Result {
            timestamps: vec![0.0, 0.01, 0.02],
            frequencies: vec![440.0, 440.0, 440.0],
            confidences: vec![1.0, 1.0, 1.0],
            voicing: vec![true, true, true],
        };
        let result = compute_rpa(&f0, &f0, 50.0);
        assert!((result.rpa - 1.0).abs() < 1e-10);
        assert!((result.rca - 1.0).abs() < 1e-10);
        assert_eq!(result.n_voiced_frames, 3);
    }

    #[test]
    fn test_compute_rpa_octave_off() {
        let ref_f0 = F0Result {
            timestamps: vec![0.0, 0.01, 0.02],
            frequencies: vec![440.0, 440.0, 440.0],
            confidences: vec![1.0, 1.0, 1.0],
            voicing: vec![true, true, true],
        };
        let est_f0 = F0Result {
            timestamps: vec![0.0, 0.01, 0.02],
            frequencies: vec![880.0, 880.0, 880.0],
            confidences: vec![1.0, 1.0, 1.0],
            voicing: vec![true, true, true],
        };
        let result = compute_rpa(&ref_f0, &est_f0, 50.0);
        // RPA = 0 (1200 cents off), RCA = 1.0 (octave folded = 0)
        assert!((result.rpa - 0.0).abs() < 1e-10);
        assert!((result.rca - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_weighted_rpa() {
        let ref_f0 = F0Result {
            timestamps: vec![0.0, 0.01, 0.02],
            frequencies: vec![440.0, 440.0, 440.0],
            confidences: vec![1.0, 0.5, 0.1],
            voicing: vec![true, true, true],
        };
        let est_f0 = F0Result {
            timestamps: vec![0.0, 0.01, 0.02],
            frequencies: vec![440.0, 440.0, 440.0],
            confidences: vec![0.9, 0.8, 0.2],
            voicing: vec![true, true, true],
        };
        let result = compute_confidence_weighted_rpa(&ref_f0, &est_f0, 50.0);
        assert!((result.rpa_weighted - 1.0).abs() < 1e-10);
        assert!((result.rpa_unweighted - 1.0).abs() < 1e-10);
        assert_eq!(result.n_voiced_frames, 3);
    }
}
