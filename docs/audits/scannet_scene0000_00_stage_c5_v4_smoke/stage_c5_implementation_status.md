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
