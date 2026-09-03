# Stage C5 v4 execution results

日期：2026-09-03

## 结论

Stage C5 robust-connected v4 depth smoke 通过。6/6 个 v4 cache 完整性校验通过；五个候选在 12k/50k、depth tolerance 0.02/0.05/0.10、pixel radius 0/1 的 12 个连接性变体中 overlap 始终非零。

C 和 D 分支在四个非 control 候选 `00369, 00384, 00065, 00437` 上，于 5 cm 和 10 cm 均产生正 novel covered-count gain。因此当前结果支持继续评估 VGGT depth + known-pose fusion，但按 tips 不自动扩展 audit20、其他场景或 policy 训练。

## 运行

- v4 cache：6（1 baseline + 5 candidate）
- VGGT reconstruction forward/cache：6
- cache-only 生成时间：31.05 s
- 进程总运行时间：38.68 s
- 峰值 GPU 显存：6,158,511,104 bytes（约 5.74 GiB）
- cache 总大小：112,289,729 bytes
- 官方 `vggt/`：未修改

## Novel covered-count gain

| Candidate | A @5/10cm | B @5/10cm | C @5/10cm | D @5/10cm |
|---|---:|---:|---:|---:|
| 00018 control | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 00369 | 2 / 11 | 85 / 231 | 113 / 239 | 106 / 248 |
| 00384 | 12 / 40 | 98 / 274 | 130 / 311 | 112 / 277 |
| 00065 | 7 / 51 | 76 / 217 | 77 / 244 | 64 / 223 |
| 00437 | -3 / -2 | 33 / 105 | 105 / 217 | 104 / 223 |

## Observed retention gain @5/10cm

| Candidate | A | B | C | D |
|---|---:|---:|---:|---:|
| 00018 control | -7 / 8 | 9 / 35 | 6 / 23 | 21 / 32 |
| 00369 | 6 / 36 | 52 / 106 | 90 / 83 | 92 / 91 |
| 00384 | -48 / -74 | -61 / -126 | -84 / -165 | -80 / -183 |
| 00065 | -27 / -43 | -51 / -112 | -85 / -143 | -72 / -136 |
| 00437 | -14 / -13 | -36 / -12 | -24 / -33 | -24 / -13 |

## 验证

- H20 Python unittest：69 passed
- compileall：通过
- git diff --check：通过
- cache fingerprint、schema、shape、view order、per-view offsets、preprocess transforms：6/6 通过

Observed retention 在 `00384`、`00065` 上有明显下降，因此 smoke 通过不等于 oracle label 已完全校准。下一步应先分析 retention/outlier 与尺度稳定性，再决定是否批准更大范围数据生成。
