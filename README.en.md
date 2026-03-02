# music-judge

Multi-dimensional singing scoring engine — pitch, rhythm, stability, and expression.

## Features

- **Pitch Score (40%)** — Octave-invariant chroma distance with adaptive perceptual tolerance (Zwicker & Fastl)
- **Rhythm Score (25%)** — Windowed local tempo deviation analysis
- **Stability Score (15%)** — Median variance, robust to note transitions
- **Expression Score (20%)** — Detrended FFT spectral energy analysis
- **Passaggio Detection** — Sliding-window standard deviation peak localization
- **DTW Alignment** — Sakoe-Chiba constrained dynamic time warping with hierarchical support
- **Attribute Feedback** — Vibrato, portamento, sustained note stability feedback generation

## Usage

```rust
use music_judge::{F0Result, ScoringOptions, compute_singing_score};

let ref_f0 = F0Result { /* reference pitch data */ };
let user_f0 = F0Result { /* user pitch data */ };
let opts = ScoringOptions::default();

let result = compute_singing_score(&ref_f0, &user_f0, None, None, &opts);
println!("Total: {:.1}", result.total_score);
```

## License

Apache-2.0
