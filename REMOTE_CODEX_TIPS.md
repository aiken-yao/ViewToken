# 给远程 Codex 的 ViewToken 执行说明

## 你的角色

你运行在带 GPU 的远程服务器上。ViewToken Phase-0 已经通过真实 VGGT-1B
烟雾测试。当前首要目标不是扩展完整 NBV 系统，而是审计 oracle gain 是否形成
稳定、可信、可排序的候选视角分布。

开始前请完整阅读：

1. `README.md`
2. `docs/feasibility_protocol.md`
3. `viewtoken/backbones/vggt_extractor.py`
4. `viewtoken/memory/scene_tokens.py`
5. `scripts/extract_vggt_features.py`

不要改变 ViewToken 的问题定义：token 用于下一最佳视角选择，不是 token
pruning，也不是计算预算分配。

## 当前状态（2026-09-01）

以下结论已经验证，不要重复从零排查：

1. 完整 VGGT-1B 权重位于：

   ```text
   /group/cjy/model_weights/vggt/model.pt
   ```

   权重包含 aggregator、camera、depth、point 和 track，可以 strict load。

2. H20 上实际可用的 Python 环境是：

   ```text
   /group/cjy/vggt/.venv/vggt-nv-sys/bin/python
   ```

   torch 2.3.1 的原 Conda 环境运行 4-view VGGT 会触发 SIGFPE。当前阶段不要继续
   使用或修复该环境，除非用户明确要求环境诊断。

3. 4 张同场景 ScanNet 图像的真实 VGGT 提取已经通过：

   ```text
   patch_tokens = [1, 4, 1036, 2048]
   patch_grid = 28 x 37
   finite ratio = 正常
   aggregator_forward_count = 1
   peak GPU memory ≈ 6.16 GB
   ```

4. observed 3-view 加 candidate 1-view 的 oracle-gain 链路已经跑通，candidate
   图像只用于离线标签，没有进入策略输入。

5. candidate `00030` 的首条标签为负：

   ```text
   Chamfer gain = -0.0820
   coverage gain = -0.00075
   F-score@0.1 gain = -0.00499
   ```

   单条负 gain 不代表实现错误。保留负值，不要 clamp 为 0。只有当一组候选视角
   整体无法形成合理分布时，才优先怀疑 alignment 或 metric pipeline。

## 当前唯一主任务：Oracle Gain Audit

在同一个 observed 3-view state 下评估至少 20 个 held-out candidate views。完成
审计前，不训练 ViewToken policy，也不增加复杂模型。

### 必须固定的实验条件

所有候选必须使用相同的：

```text
observed view IDs
reference/input image order
random seed
Sim(3) alignment protocol
confidence threshold
voxel-downsample size
metric point-sampling count
ground-truth preprocessing
```

candidate 必须始终追加在 observed views 后面，不要因候选不同而改变 reference
image。比较 observed reconstruction 与 observed+candidate reconstruction 时，两者都
必须按同一协议对齐到 GT。

### 必须新增或报告的指标

不要只报告对称 Chamfer。将其拆分为：

```text
accuracy:     prediction -> GT
completeness: GT -> prediction
```

同时计算：

```text
coverage
F-score@0.02
F-score@0.05
F-score@0.10
```

阈值必须在对齐后的同一尺度下解释。如果 ScanNet GT 使用米制，明确说明 Sim(3)
对齐后阈值仍以米为单位。

### 每项 gain 的统计汇总

至少输出：

```text
candidate count
min / median / mean / max
positive-gain ratio
oracle-best candidate and gain
random-candidate mean gain
```

另外报告不同指标之间的 Spearman rank correlation，检查 Chamfer、coverage 和
F-score 是否对候选排序产生严重冲突。不要在看到分布前拍脑袋组合多个指标成一个
加权总分。

### 三组 sanity checks

必须显式加入：

1. 重复一个已经 observed 的视角：gain 应接近 0；
2. 与 observed 高重叠的相邻视角：gain 通常较小；
3. 能观察明显新区域的视角：至少部分 coverage/completeness 指标应为正。

候选视角真实 RGB 仍然只能用于离线重建和标签计算，不能进入任何候选评分模型。

### 审计结果判断

- 如果 gain 有合理正负分布，且 oracle-best 明显好于随机候选平均值，可以开始构建
  小规模 oracle-gain dataset。
- 如果几乎所有候选均为负或彼此无法区分，先检查 Sim(3)、输入顺序、离群点、点云
  密度、confidence filtering 和 point-sampling 公平性。
- 如果 accuracy 变差但 completeness/coverage 变好，不要简单判为 pipeline 失败；
  这可能是新增表面同时引入离群点，需要分别报告两个方向。
- 完成审计后先汇报，不要直接开始完整 policy。

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

本节保留用于复现实验。当前优先使用上面已经验证的 Python 与权重路径。

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

## Oracle audit 之后做什么

只有当前 Oracle Gain Audit 通过后，才实现小规模 oracle-gain dataset：

1. 先扩展到至少 5 个场景，每个场景固定 3--4 个初始观测视角；
2. 对每个 held-out candidate 分别重建；
3. 保存 Chamfer accuracy/completeness、F-score 和 coverage 的独立增益；
4. 保存 candidate pose、当前 memory ID、view IDs、alignment 和 metric 配置；
5. 严格按 scene/object 划分训练和测试；
6. 对每个 scene/state 保存完整候选集合，后续训练 ranking probe，而不是只保存最佳
   candidate。

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
