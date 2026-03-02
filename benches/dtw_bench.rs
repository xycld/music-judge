//! Performance benchmarks for DTW alignment.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use music_judge::{constrained_dtw, F0Result, hierarchical_dtw};

fn bench_constrained_dtw_1000(c: &mut Criterion) {
    // 1000 x 1000 DTW — target: <10ms
    let n = 1000;
    let ref_cent: Vec<f64> = (0..n).map(|i| 440.0 + (i as f64 * 0.1).sin() * 50.0).collect();
    let user_cent: Vec<f64> = (0..n).map(|i| 442.0 + (i as f64 * 0.1).cos() * 45.0).collect();

    c.bench_function("constrained_dtw 1000x1000", |b| {
        b.iter(|| constrained_dtw(black_box(&ref_cent), black_box(&user_cent), 200))
    });
}

fn bench_constrained_dtw_500(c: &mut Criterion) {
    let n = 500;
    let ref_cent: Vec<f64> = (0..n).map(|i| 440.0 + (i as f64 * 0.1).sin() * 50.0).collect();
    let user_cent: Vec<f64> = (0..n).map(|i| 442.0 + (i as f64 * 0.1).cos() * 45.0).collect();

    c.bench_function("constrained_dtw 500x500", |b| {
        b.iter(|| constrained_dtw(black_box(&ref_cent), black_box(&user_cent), 100))
    });
}

fn bench_hierarchical_dtw(c: &mut Criterion) {
    let n = 2000;
    let make_f0 = |freq_base: f64| F0Result {
        timestamps: (0..n).map(|i| i as f64 * 0.01).collect(),
        frequencies: (0..n)
            .map(|i| freq_base + (i as f64 * 0.05).sin() * 30.0)
            .collect(),
        confidences: vec![0.9; n],
        voicing: vec![true; n],
    };

    let ref_f0 = make_f0(440.0);
    let user_f0 = make_f0(442.0);

    c.bench_function("hierarchical_dtw 2000 frames (no audio)", |b| {
        b.iter(|| {
            hierarchical_dtw(
                black_box(&ref_f0),
                black_box(&user_f0),
                None,
                None,
                16000,
                5,
                0.15,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_constrained_dtw_1000,
    bench_constrained_dtw_500,
    bench_hierarchical_dtw,
);
criterion_main!(benches);
