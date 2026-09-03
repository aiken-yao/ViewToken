# Stage C5 v4 implementation status

日期：2026-09-03

## 当前结论

已按 C5 tips 完成 v4 cache schema 的读取/校验基础、depth artifact 归一化保存、固定 smoke 配置和 fail-closed 审计入口。按用户要求，本轮不运行 VGGT，因此没有生成新的 6 个 v4 cache，也没有使用旧 v3/audit20 cache 冒充 C5 结果。

审计器当前对缺失 cache 返回 `blocked_missing_v4_cache`，并列出 baseline 与 5 个 candidate 的预期 view order。

## 固定输入

- observed: `00000, 00010, 00020`
- candidates: `00018, 00369, 00384, 00065, 00437`
- preprocess: `crop`
- schema: `v4`
- reconstruction sampling: `none`
- random seed: `0`

## 验证

- pytest（禁用环境插件自动加载）：64 passed
- compileall：通过
- git diff --check：通过
- 实际 VGGT forward/cache 数：0
- 官方 `vggt/`：未修改

## 本轮继续更新

审计器已扩展为真正的 cache-only A/B/C/D 分支比较：

- `scripts/generate_oracle_gain.py --cache-only` 只生成并校验 reconstruction cache，
  不计算旧的 free-ICP oracle gain；
- `scripts/audit_stage_c5_v4_branches.py` 对每个候选共享同一 GT target、visibility mask
  和 deterministic sampling，输出 novel coverage gain、observed retention、outlier、
  depth/intrinsics 与 held-out pose diagnostics；
- v4 fingerprint 现在包含 calibrated intrinsics，避免内参配置变化时错误复用 cache；
- 新增 `tests/test_v4_branches.py`，覆盖 observed-only scale、C/D 内参分支、identical
  reconstruction gain 和 candidate pose held-out 诊断。

本机 `compileall` 与 `git diff --check` 通过。Windows 默认 Python 的 NumPy namespace
损坏（`numpy.ndarray` 缺失）导致 torch 单测无法启动；远程 H20 环境仍需重新运行完整
测试。真实 v4 cache 和 VGGT forward 仍为 0，尚未启动 GPU。
