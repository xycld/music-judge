//! Compare Rust scoring results with Python on the same F0 data.
//!
//! Usage: cargo run --release --example compare

use music_judge::{F0Result, ScoringOptions, compute_singing_score};
use std::fs;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct F0Export {
    reference: F0Data,
    user: F0Data,
}

#[derive(serde::Deserialize)]
struct F0Data {
    timestamps: Vec<f64>,
    frequencies: Vec<f64>,
    confidences: Vec<f64>,
    voicing: Vec<bool>,
}

#[derive(serde::Deserialize)]
struct PyResult {
    pitch_score: f64,
    rhythm_score: f64,
    stability_score: f64,
    expression_score: f64,
    total_score: f64,
    dtw_cost: f64,
    passaggio_hz: Option<f64>,
    time_ms: f64,
}

fn main() {
    let f0_path = "../benchmark/f0_data.json";
    let py_result_path = "../benchmark/py_result_simple.json";

    println!("{}", "=".repeat(60));
    println!("Music-Judge: Rust Scorer");
    println!("{}", "=".repeat(60));

    // 1. Load F0 data
    println!("\n[1/3] Loading F0 data from JSON...");
    let f0_json = fs::read_to_string(f0_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}\n  Run Python first: cd .. && python benchmark/compare.py", f0_path, e));
    let f0_export: F0Export = serde_json::from_str(&f0_json).expect("Failed to parse F0 JSON");

    let ref_f0 = F0Result {
        timestamps: f0_export.reference.timestamps,
        frequencies: f0_export.reference.frequencies,
        confidences: f0_export.reference.confidences,
        voicing: f0_export.reference.voicing,
    };
    let user_f0 = F0Result {
        timestamps: f0_export.user.timestamps,
        frequencies: f0_export.user.frequencies,
        confidences: f0_export.user.confidences,
        voicing: f0_export.user.voicing,
    };

    println!("  Reference: {} frames", ref_f0.frequencies.len());
    println!("  User:      {} frames", user_f0.frequencies.len());

    // 2. Score with Rust (simple DTW, no audio — matches Python simple mode)
    println!("\n[2/3] Scoring with Rust...");
    let opts = ScoringOptions {
        use_hierarchical: false,
        ..ScoringOptions::default()
    };

    let t0 = Instant::now();
    let result = compute_singing_score(&ref_f0, &user_f0, None, None, &opts);
    let elapsed = t0.elapsed();

    println!("\n{}", "─".repeat(60));
    println!("  Rust Scoring Results (took {:.1} ms)", elapsed.as_secs_f64() * 1000.0);
    println!("{}", "─".repeat(60));
    println!("  Pitch Score:      {:.1}", result.pitch_score);
    println!("  Rhythm Score:     {:.1}", result.rhythm_score);
    println!("  Stability Score:  {:.1}", result.stability_score);
    println!("  Expression Score: {:.1}", result.expression_score);
    println!("  Total Score:      {:.1}", result.total_score);
    println!("  DTW Cost:         {:.2}", result.dtw_cost);
    match result.passaggio_hz {
        Some(p) => println!("  Passaggio:        {:.1} Hz", p),
        None => println!("  Passaggio:        None"),
    }
    println!("  Phrases:          {}", result.phrase_results.len());
    println!("  Feedback items:   {}", result.feedback.len());

    // 3. Compare with Python results
    println!("\n[3/3] Comparing with Python...");
    if let Ok(py_json) = fs::read_to_string(py_result_path) {
        if let Ok(py) = serde_json::from_str::<PyResult>(&py_json) {
            println!("\n{}", "─".repeat(60));
            println!("  {:20} {:>10} {:>10} {:>10}", "Dimension", "Python", "Rust", "Diff");
            println!("{}", "─".repeat(60));

            let comparisons = [
                ("Pitch Score", py.pitch_score, result.pitch_score),
                ("Rhythm Score", py.rhythm_score, result.rhythm_score),
                ("Stability Score", py.stability_score, result.stability_score),
                ("Expression Score", py.expression_score, result.expression_score),
                ("Total Score", py.total_score, result.total_score),
                ("DTW Cost", py.dtw_cost, result.dtw_cost),
            ];

            for (name, py_val, rs_val) in &comparisons {
                let diff = rs_val - py_val;
                let marker = if diff.abs() < 0.2 { "✓" }
                    else if diff.abs() < 1.0 { "~" }
                    else { "✗" };
                println!("  {:20} {:>10.1} {:>10.1} {:>+9.2} {}", name, py_val, rs_val, diff, marker);
            }

            // Passaggio comparison
            match (py.passaggio_hz, result.passaggio_hz) {
                (Some(p), Some(r)) => {
                    let diff = r - p;
                    let marker = if diff.abs() < 5.0 { "✓" } else { "✗" };
                    println!("  {:20} {:>10.1} {:>10.1} {:>+9.2} {}", "Passaggio Hz", p, r, diff, marker);
                }
                (None, None) => println!("  {:20} {:>10} {:>10} {:>10}", "Passaggio Hz", "None", "None", "match ✓"),
                (p, r) => println!("  {:20} {:>10?} {:>10?}  mismatch ✗", "Passaggio Hz", p, r),
            }

            println!("{}", "─".repeat(60));

            // Performance comparison
            let rust_ms = elapsed.as_secs_f64() * 1000.0;
            let speedup = py.time_ms / rust_ms;
            println!("\n  Performance:");
            println!("    Python: {:.1} ms", py.time_ms);
            println!("    Rust:   {:.1} ms", rust_ms);
            println!("    Speedup: {:.0}x", speedup);
        } else {
            println!("  Could not parse Python results JSON");
        }
    } else {
        println!("  Python results not found at {}", py_result_path);
        println!("  Run Python first: cd .. && python benchmark/compare.py");
    }

    // Save Rust result as JSON
    let rust_json = serde_json::to_string_pretty(&result).unwrap();
    fs::write("../benchmark/rs_result.json", &rust_json).ok();
    println!("\n  Rust results saved to benchmark/rs_result.json");
}
