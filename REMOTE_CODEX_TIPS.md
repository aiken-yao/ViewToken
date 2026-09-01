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

## 2026-09-02 更新：Oracle Gain Audit 已完成，但未通过可信度验收

真实审计输出位于：

```text
outputs/oracle_gain/scannet_scene0000_00_audit20/
```

审计使用 observed `00000, 00010, 00020`，评估了 20 个 held-out candidate，另有
1 个 repeat-observed sanity case。主要结果为：

```text
held-out candidates = 20
Chamfer gain mean = -0.1252, positive ratio = 0.05
accuracy gain mean = +0.1300, positive ratio = 0.80
completeness gain mean = -0.3803, positive ratio = 0.05
coverage gain mean = -0.00190, positive ratio = 0.40
F-score@0.02 gain mean = +0.00042, positive ratio = 0.65
F-score@0.05 gain mean = +0.00088, positive ratio = 0.60
F-score@0.10 gain mean = -0.00999, positive ratio = 0.35
```

当前结论不是“VGGT 加入视角必然变差”，而是现有 oracle label 测量协议尚未校准。
在完成下述修复前，不扩展 oracle-gain dataset，不训练 policy，也不据此决定 ViewToken
选题是否可行。

### 已确认的高优先级问题

1. 当前 repeat-observed case 实际输入为
   `[00000, 00010, 00020, 00010]`，不是对同一 baseline 重复计算。重复图像会改变
   VGGT 的多视图 attention 和几何预测，因此它只能叫
   `duplicate_input_sensitivity`，不能用来要求 gain 等于 0，也不能作为合法 NBV
   candidate。
2. baseline 与每个 candidate 当前分别使用自由尺度 Sim(3) ICP 对齐到完整 ScanNet
   GT。部分点云对完整场景进行无约束 scale/translation/rotation 拟合，可能收缩到
   GT 的局部区域，从而造成 accuracy 变好、completeness/coverage 变差。当前 ICP
   还缺少 correspondence trimming、outlier rejection、overlap 约束以及变换诊断。
3. 点云在对齐前随机截取到 50k，计算指标时再次随机采样到 12k。相同 seed 对
   3-view 和 4-view 点云不会产生等价空间子集，小幅 gain 可能只是采样噪声。
4. reconstruction cache 没有配置 fingerprint。改变图像顺序、checkpoint、confidence
   threshold、seed、最大点数或代码后，`reuse-reconstructions: true` 仍可能静默复用
   旧结果。
5. `min-world-point-confidence: 0.0` 基本没有过滤任何点。阈值实验必须在 cache、
   alignment 和 sampling 修好以后进行，不能先在单场景上调参。
6. 当前 high-overlap/new-area 标签主要由相机平移距离指定，没有验证视锥重叠、朝向
   和可见新表面比例，因此不能作为强 sanity 结论。

## 当前唯一主任务：校准 Oracle 测量协议

阶段 A 已于 2026-09-02 完成，结论为“部分通过，但整体校准未通过”。当前只执行
阶段 B 的相机锚定公平对齐。不要运行新的 VGGT 重建；优先复用 audit20 和 Stage A
已有 cached reconstructions。不要扩展 dataset，也不要训练 policy。

### 阶段 A：已完成——隔离 metric/alignment 噪声

报告位置：

```text
outputs/oracle_calibration/scannet_scene0000_00_stage_a/stage_a_metric_stability.json
outputs/oracle_calibration/scannet_scene0000_00_stage_a/run_log.md
docs/audits/scannet_scene0000_00_stage_a/
```

已完成：alignment diagnostics、identical-cloud 端到端检查、seed `0..9` 稳定性、
synthetic Sim(3) full/partial correspondence 测试、cache fingerprint，以及把重复输入
重命名为 `duplicate_input_sensitivity`。当前测试结果为 22 tests OK，`compileall` 和
`git diff --check` 均通过，官方 `vggt/` 未修改。

关键结果：

```text
identical-cloud max_abs_gain = 0.0

multi-seed metric std:
chamfer      = 0.00795
accuracy     = 0.00285
completeness = 0.01814
coverage     = 0.00151
F@0.02       = 0.00044
F@0.05       = 0.00157
F@0.10       = 0.00212

std / old audit20 candidate-gain range:
chamfer      = 0.0209
completeness = 0.0176
coverage     = 0.0950
F@0.02       = 0.1185

free Sim(3) ICP diagnostics:
scale mean                  = 5.12
rotation angle mean         = 11.90 deg
translation norm mean       = 8.44
residual RMSE mean          = 0.669 m
inlier ratio @ 0.05 m mean  = 0.00869
```

解释：identical-cloud 和 synthetic Sim(3) 表明 gain 算术、确定性链路和基础 Sim(3)
数学实现可以站住；但真实 VGGT partial reconstruction 对完整 ScanNet GT 的自由 ICP
几乎没有形成有效几何对应。`scale=5.12` 单独看不一定是错误，因为 VGGT 有尺度不确定
性；真正否决旧 alignment 的证据是 `0.669 m` residual 与仅 `0.87%` 的 5 cm inlier
ratio。旧 audit20 标签因此仍不可用于训练。

coverage 和 F@0.02 的 sampling std 分别达到旧 candidate range 的约 9.5% 和 11.9%，
并非完全可忽略，因此阶段 C 仍然必须完成；但当前第一阻塞是阶段 B 的 alignment。

以下保留为阶段 A 已执行要求和回归测试规范：

1. 使用完全相同的 baseline cached points 作为 baseline 和 candidate 输入，完整执行
   两次对齐与 metric，所有 gain 必须严格为 0 或仅有浮点误差。新增端到端测试。
2. 对同一 cached cloud 使用 seed `0..9` 重复评估，报告每项 metric 的 mean/std，
   以及 alignment scale、rotation、translation、residual 和 inlier ratio。评估标准差
   必须显著小于真实 candidate 的 gain 差异。
3. 新增已知变换的 synthetic Sim(3) recovery test，分别覆盖完整重叠与部分重叠。
4. 为 reconstruction cache 增加 fingerprint，至少包含 checkpoint 身份、ordered image
   paths、preprocess、layer、confidence threshold、max points、seed 和 cache schema
   version。实现前设置 `reuse-reconstructions: false`，或每次使用全新输出目录。
5. 不把 observed view 放入真实 NBV action set。保留重复图像实验时，将记录类型明确
   标为 `duplicate_input_sensitivity`。

### 阶段 B：代码路径已完成，当前阻塞于旧 cache 缺少 pose encoding

2026-09-02 已完成 camera-anchor alignment 基础实现、Stage B 审计脚本、v3 cache
schema 和 pose convention/camera alignment 单测。实际审计没有运行新的 VGGT，报告位于：

```text
docs/audits/scannet_scene0000_00_stage_b/run_log.md
docs/audits/scannet_scene0000_00_stage_b/stage_b_camera_anchor_report.json
```

结果为：

```text
status = blocked_missing_pose_enc
attempted records = 21
missing pose_enc.pt = 22
did_run_vggt = false
tests = 27 OK
```

22 份缺失文件对应 1 份共享 baseline cache 和 21 份 candidate cache，数量正确。旧
audit20 cache 只有 `points.pt`、`confidence.pt` 和 `metadata.json`，无法从中恢复 VGGT
预测相机。因此停止而不偷偷运行 VGGT 是正确行为。不要把后来单独推理得到的
`pose_enc.pt` 补写进旧 cache；points 与 pose 必须来自同一次、同配置的重建并保存在
新的 v3 输出目录。

在请求用户批准新的 VGGT 推理前，先完成以下四项代码补强；这些工作不需要运行 VGGT：

1. **强制 v3 artifact 完整性。** 新建 oracle reconstruction 时，若
   `features.pose_enc is None` 必须立即失败，不能仍写成 v3 cache。复用时必须同时检查
   `points.pt`、`confidence.pt`、`pose_enc.pt`、`metadata.json`、schema version 和
   fingerprint。Stage B 载入 cache 时也要验证 schema/fingerprint，不能只检查
   `pose_enc.pt` 是否存在。
2. **补齐 camera alignment 后的几何诊断。** 对 baseline/candidate 分别报告点云到 GT
   的 residual mean/median/RMSE/max，以及 inlier ratio@0.02/0.05/0.10m，并与旧
   free-ICP 结果比较。只有 camera RMSE 不足以证明点云对齐有效。
3. **对退化 anchor fail closed。** 同时记录 predicted 和 GT camera-center condition
   number，检查 pose/center 全部有限。超过明确阈值或近共线时输出
   `blocked_degenerate_camera_anchors`，不得继续生成可训练 label。不能使用 candidate
   camera 补足约束；应改用 4 个 observed views 或加入 observed-camera orientation。
4. **修正并测试 proper-rotation Sim(3)。** 当前 reflection correction 后的 scale
   必须使用带符号奇异值和，而不是无条件对全部 singular values 求和。新增任意 3D
   rotation/scale/translation、非共面 4-point 和 reflection-rejection 测试，避免从
   3 个共面相机扩展到 4 个 anchor 时出现隐藏误差。

上述四项完成后运行 `compileall`、完整单测和 `git diff --check`，先汇报修改与测试，
不要自行开始 GPU 推理。

以下保留为 Stage B 的设计和验收要求：

1. 从每次 VGGT 重建保存的 `pose_enc` 解码 extrinsics/intrinsics。优先复用官方
   `pose_encoding_to_extri_intri`，不要修改 `vggt/`。
2. 明确外参约定并计算预测相机中心；若为 world-to-camera，则
   `C = -R^T t`。增加 pose convention 单元测试。
3. 将相同 view ID 的预测相机中心与 ScanNet GT camera-to-world pose 对应起来。
4. baseline 与 `observed+candidate` 分别根据各自预测的相机中心估计 Sim(3)，但两边
   必须使用完全相同的共享 observed IDs。candidate 自身绝不能成为 alignment anchor。
5. 将各自的相机锚定 Sim(3) 应用于对应预测世界点。第一轮只报告纯 camera-Sim(3)
   结果，不要立即混入点云 ICP；必要时后续只能追加固定 scale 的 robust rigid ICP，
   并使用 trimming/outlier rejection。
6. 每条 record 保存并汇报 scale、rotation、translation、每个 anchor 的误差、camera
   RMSE/max error、shared anchor IDs，以及相机中心几何的 condition number。若三个
   observed camera centers 近共线，不要用 candidate 补足约束；改用 4 个初始 observed
   views，或纳入相机朝向约束。
7. 使用新 alignment 重新计算旧 audit20 cached reconstructions 的 metrics，并比较
   camera-Sim(3) 与旧 free-ICP 的 gain 分布、scale 漂移、residual 和 inlier ratio。
   本阶段不生成新的 VGGT reconstruction。

阶段 B 必须满足：相机 anchor residual 明显降低；baseline/candidate scale 不发生异常
塌缩；candidate 加入后 observed-camera alignment 不剧烈漂移；新协议下
identical-cloud gain 仍为 0；点云对 GT 的 residual/inlier 明显优于旧 free-ICP。完成后
先汇报并等待确认，再进入阶段 C。

在四项补强通过且用户明确批准重建后，不要立即重跑全部 22 份。先在新的 v3 output
directory 中重建 baseline 和 3--4 个代表性 candidate：

```text
duplicate-input sensitivity
high-overlap
old oracle-best
connected new-area
```

先验证 cache 完整性、camera anchor condition、scale 漂移、camera residual、point-cloud
residual/inlier 和 gain 分布。这个小规模 Stage B smoke 通过后，再请求批准重建完整
audit20。旧 cache 继续只读保留，不能覆盖。

### 阶段 C：确定性几何评估

1. 移除对齐前的随机 50k 截断；先做 finite/confidence filtering，再对齐，再做确定性
   voxel downsample。
2. 所有 candidate 共用一份缓存且固定的 GT point set。优先评估全部 voxel centroids；
   若必须限点，使用确定性的空间/hash sampling 或报告多 seed 误差条。
3. 基于 ScanNet GT pose、depth 或 mesh 计算 frustum overlap 与 novel visible surface
   fraction，重新定义 high-overlap 和 new-area audit case。GT 可见性仅用于离线审计和
   oracle label，不能进入 NBV policy 输入。
4. alignment 与 sampling 稳定后，再比较 confidence threshold，例如绝对阈值或固定
   分位数，并同时报告保留点数、coverage 和各项 metric。

### 校准通过条件

只有同时满足以下条件，才能重新运行完整 audit20 并考虑扩展到 5 个场景：

- identical-cloud gain 接近 0；
- 多 seed metric/alignment 标准差远小于 candidate gain 间隔；
- synthetic full/partial-overlap Sim(3) 测试通过；
- baseline 与 candidate 的 scale 不发生异常漂移或塌缩；
- cache 配置变化能正确失效；
- duplicate-input sensitivity 已与 NBV candidate 分离；
- 至少部分具有足够连接性的 novel view 能稳定改善 coverage/completeness；
- 所有新增单元测试、`compileall` 和原有测试通过。

完成以上校准后先提交一份对比报告：旧 ICP 对齐 vs 相机锚定对齐、随机采样 vs
确定性 voxel 评估，以及各自的 gain 分布和稳定性。等待用户确认后再扩展数据集。

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

### 四组 sanity checks

必须显式加入并区分：

1. identical-cloud：同一份 cached reconstruction 重复对齐、采样和评估，gain 应接近
   0；这是 metric pipeline 的确定性检查。
2. duplicate-input sensitivity：把已经 observed 的图像再次送入 VGGT，研究多视图
   推理对重复输入的敏感性；该结果不要求为 0，也不能作为 NBV candidate。
3. 与 observed 高重叠的相邻视角：gain 通常较小，但必须用 GT frustum/surface
   overlap 验证“高重叠”，不能只看相机平移距离。
4. 能观察明显新区域且与当前观测保持足够连接性的视角：至少部分
   coverage/completeness 指标应稳定为正。

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
