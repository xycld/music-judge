# music-judge

Given reference and user vocal F0 (fundamental frequency) sequences, produces a composite score with per-phrase detail.

Scoring covers four dimensions: pitch (40%), rhythm (25%), stability (15%), and expression (20%). It also detects the passaggio (vocal break point) and generates singing attribute feedback (vibrato, portamento, sustained note stability, etc.).

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
