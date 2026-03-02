# music-judge

给定参考人声和用户人声的 F0（基频）序列，输出一个综合评分和逐乐句的详细结果。

评分包含四个维度：音准 (40%)、节奏 (25%)、稳定性 (15%)、表现力 (20%)。同时会检测换声点位置，并生成中文的演唱属性反馈（颤音、滑音、长音稳定性等）。

## 用法

```rust
use music_judge::{F0Result, ScoringOptions, compute_singing_score};

let ref_f0 = F0Result { /* 参考音高数据 */ };
let user_f0 = F0Result { /* 用户音高数据 */ };
let opts = ScoringOptions::default();

let result = compute_singing_score(&ref_f0, &user_f0, None, None, &opts);
println!("总分: {:.1}", result.total_score);
```

## 许可证

Apache-2.0
