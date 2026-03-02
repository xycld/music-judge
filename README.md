# music-judge

多维度歌唱评分引擎 — 音准、节奏、稳定性、表现力。

## 特性

- **音准评分 (40%)** — 基于八度不变色度距离 + 自适应容差（Zwicker & Fastl 感知模型）
- **节奏评分 (25%)** — 滑动窗口局部速度偏差分析
- **稳定性评分 (15%)** — 中位数方差，对音符过渡鲁棒
- **表现力评分 (20%)** — 去趋势后 FFT 频谱能量分析
- **换声点检测** — 滑动窗口标准差峰值定位
- **DTW 对齐** — Sakoe-Chiba 约束动态时间规整，支持分层对齐
- **属性反馈** — 颤音、滑音、长音稳定性等中文反馈生成

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
