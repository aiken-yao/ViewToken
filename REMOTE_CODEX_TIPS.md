# 给远程 Codex 的 ViewToken 执行说明

## 你的角色

你运行在带 GPU 的远程服务器上。当前首要目标不是扩展完整 NBV 系统，而是让
ViewToken Phase-0 在真实 VGGT-1B 上可靠运行，并产出可检查的 world-aligned
token cache。

开始前请完整阅读：

1. `README.md`
2. `docs/feasibility_protocol.md`
3. `viewtoken/backbones/vggt_extractor.py`
4. `viewtoken/memory/scene_tokens.py`
5. `scripts/extract_vggt_features.py`

不要改变 ViewToken 的问题定义：token 用于下一最佳视角选择，不是 token
pruning，也不是计算预算分配。

## 不可违反的研究约束

1. 决策时只能使用已经观测到的图像、由其构建的 Scene Token Memory，以及候选
   相机 pose/rays。
2. 候选视角的真实 RGB、深度、VGGT feature 或可见性不能作为 NBV 模型输入。
3. 候选图像只能在离线阶段用于计算 oracle reconstruction-gain 标签。
4. 第一阶段冻结全部 VGGT 参数，不进行 fine-tuning。
5. 不修改 `vggt/` 中的官方实现。优先修复 `viewtoken/` adapter；只有确认官方接口
   本身存在问题时才提出修改建议。
6. 暂时不做 RL、不做跨视角 token 合并、不做完整 counterfactual transformer。

## 任务一：同步并检查环境

```bash
cd /mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken
git status --short
git pull --ff-only origin main

python --version
python - <<'PY'
import numpy
import torch

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

不要为了安装本项目而盲目升级或降级服务器已有的 CUDA/PyTorch。若 VGGT 已能运行，
只补充缺少的轻量依赖：

```bash
python -m pip install -e ./vggt
python -m pip install "PyYAML>=6.0"
```

若必须修改环境，先报告当前版本、错误和拟执行的命令。

## 任务二：运行不需要权重的测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q viewtoken scripts tests
```

预期所有测试通过。测试失败时：

1. 先确认是代码错误还是环境错误；
2. 给出最小复现；
3. 只修改必要文件；
4. 修复后重新运行完整测试。

## 任务三：选择 Phase-0 烟雾测试图像

需要同一静态场景或物体的 3--4 张不同视角 RGB 图像。优先使用用户明确指定的
数据目录。若没有给出目录，先询问用户，不要递归扫描整个共享存储。

图片要求：

- `.jpg`、`.jpeg` 或 `.png`；
- 同一场景、存在足够视角重叠；
- 第一轮不要超过 4 张，以免浪费 VGGT-1B 推理时间和显存。

## 任务四：运行真实 VGGT-1B 提取

保持网络代理/SSH 反向隧道开启。如果 Hugging Face 权重已缓存，应优先复用缓存。

```bash
python scripts/extract_vggt_features.py \
  --images /path/view_000.png /path/view_001.png /path/view_002.png \
  --output-dir outputs/token_cache/smoke_test
```

默认使用：

- backbone：`facebook/VGGT-1B`；
- aggregator layer：23；
- patch feature：frame/global concat 后的 2048 维 raw token；
- token storage dtype：float16；
- geometry storage dtype：float32；
- VGGT：`eval()` 且全部参数 `requires_grad=False`。

提取器通过临时 forward hook 捕获 aggregator 输出，因此一次运行只能有一次 VGGT
aggregator forward。不要改成先调用 `model.aggregator(images)`、再调用
`model(images)` 的双前向实现。

## 任务五：验证输出

确认目录包含：

```text
patch_tokens.pt
depth.pt
depth_conf.pt
world_points.pt
world_points_conf.pt
pose_enc.pt
patch_positions.pt
patch_confidence.pt
patch_valid_mask.pt
metadata.json
```

运行下面的检查，必要时补成仓库脚本：

```bash
python - <<'PY'
import json
from pathlib import Path

import torch

root = Path("outputs/token_cache/smoke_test")
metadata = json.loads((root / "metadata.json").read_text())
print(json.dumps(metadata, indent=2))

for path in sorted(root.glob("*.pt")):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    finite = torch.isfinite(tensor).float().mean().item() if tensor.is_floating_point() else None
    print(path.name, tuple(tensor.shape), tensor.dtype, "finite_ratio=", finite)
PY
```

必须核对：

1. `patch_tokens` 为 `[B, S, N, 2048]`；
2. `patch_positions` 为 `[B, S, N, 3]`；
3. `patch_confidence` 和 `patch_valid_mask` 为 `[B, S, N]`；
4. `N == patch_grid_height * patch_grid_width`；
5. batch 通常为 1，`S` 等于输入图像数；
6. patch token 已去除 camera/register special tokens；
7. 有效 patch 的 position 必须有限；
8. 报告 valid ratio、confidence 的 min/median/max、运行时间、峰值显存和缓存大小。

如果输入为 518 x 518，patch size 14，则 patch grid 应为 37 x 37，`N=1369`。
非方形输入以 `metadata.json` 中的实际 patch grid 为准。

## Scene Token 的当前定义

每个 VGGT patch token 对应同一图像 patch 内世界点的置信度加权质心：

```text
position_i = sum_p(conf_p * world_point_p) / sum_p(conf_p)
```

非有限世界点和低于阈值的 confidence 必须被屏蔽。当前保留 `[B,S,N,...]`，不要
跨视角平均 feature，也不要做 voxel merge。

## Phase-0 验收条件

只有满足以下条件才能进入 oracle-gain 数据生成：

- 两组单元测试全部通过；
- 真实 3--4 视角提取成功；
- token、position 和 dense geometry shape 一致；
- 输出没有大面积 NaN/Inf；
- 有效 patch 比例合理并已报告；
- 确认整个过程只有一次 aggregator forward；
- 未使用任何候选视角图像作为模型输入。

## Phase-0 之后做什么

完成验收后，先报告结果并等待确认。下一任务是实现 oracle-gain 数据生成：

1. 固定 3--4 个初始观测视角；
2. 对每个 held-out candidate 分别重建；
3. 计算 Chamfer、F-score 和 coverage 的真实增益；
4. 保存 candidate pose、当前 memory ID 和 gain 标签；
5. 严格按 scene/object 划分训练和测试。

不要直接跳到完整 ViewToken policy。首先训练等容量 diagnostic probes，比较：

```text
pose only
geometry/visibility
xyz + confidence
VGGT feature + xyz + confidence + candidate pose/rays
```

只有 token 版本在未见场景上稳定优于 `xyz + confidence`，才继续构建
counterfactual ViewToken policy。

## 修改与汇报规范

- 不执行 `git reset --hard`、强制推送或批量删除。
- 不提交数据集、模型权重、`outputs/`、日志或密钥。
- 修改前运行 `git status --short`，保留用户已有变更。
- 完成后运行测试并展示 `git diff --stat`。
- 除非用户明确要求，否则不要自行 commit 或 push。

最终汇报必须包含：

```text
完成了什么
修改了哪些文件
运行了哪些命令
测试结果
真实输出 shape/statistics
仍存在的风险或阻塞
下一步建议
```

