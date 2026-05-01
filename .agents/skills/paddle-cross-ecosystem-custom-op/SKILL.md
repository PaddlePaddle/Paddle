---
name: paddle-cross-ecosystem-custom-op
description: 将原生 PyTorch 自定义算子库、Torch extension、生态库（TorchCodec/FlashInfer/DeepEP 等）以及 Kernel DSL 生态（Triton/TileLang/TVM FFI 等）以最小修改方式接入 PaddlePaddle。遇到以下场景务必使用：迁移外部算子库到 Paddle；分析 PFCCLab fork 与上游的兼容差异；处理 paddle.enable_compat、paddle.utils.cpp_extension、TORCH_LIBRARY、torch.ops、at::Tensor/c10 compat 问题；为 compat gap 设计最小 workaround 并准备 Paddle issue 最小复现。
---

# Paddle 跨生态自定义算子迁移

## 任务定义

这个 skill 只做一件事：让上游 PyTorch 自定义算子仓库在 Paddle 上按原来的调用路径跑起来，同时保持后续 rebase / sync upstream 的能力。

一次完整的输出应该覆盖四个方面：

- 迁移方案
- 最小修改边界
- 验证路径
- compat gap 处理策略

## 核心约束

- **最小修改**：不做额外格式化、优化、重构，不主动改公共 API。
- **上游同步**：所有改动都要考虑后续 rebase / sync upstream 的便利性。
- **compat 优先**：优先使用 Paddle 现有的 compat 机制，让 compat 层承担兼容职责。
- **缺口要明确**：compat gap 要分类清楚、标明边界，并准备最小复现。
- **验证要闭环**：至少跑通一条最小 build/test 路径。

## 工作顺序

1. 识别上游仓库、当前 fork、默认分支和实际迁移分支。PFCCLab 适配仓库的默认分支通常是 `paddle`，比较前先确认 parent 和默认分支。
2. 按控制面把仓库分成四层：
   - 框架无关的内核 / 算法
   - 构建与打包
   - C++ compat API / 注册
   - Python 包装 / runtime glue / tests
3. 如果任务是分析多个 PFCCLab fork，且用户明确要求并行，按仓库拆分并行分析；每个子任务都要输出 parent、比较分支、四层 diff 归类和可复用模式。
4. 先确定第一轮改动的位置。首轮补丁通常集中在 build、runtime glue、device / stream / distributed 边界。
5. 沿最小路径逐步推进验证：build → import → 最小功能测试 → 运行时对照。

## 默认改动边界

通常不需要动的部分：

- CUDA/C++ 核心 kernel 与算法逻辑
- 原有 schema 定义
- 大部分 `TORCH_LIBRARY` / pybind11 注册代码
- 上游目录结构与 Python package 形状

通常需要先检查的部分：

- `setup.py` / `pyproject.toml`
- 入口脚本、测试脚本、示例脚本
- `torch.ops` / `torch.library` / `torch._dynamo` / `torch.profiler` 使用点
- device / stream / distributed / DLPack / custom op registration glue

## 具体规则

- `setup.py` / `pyproject.toml`：优先加 `paddle.enable_compat()`，保留原有 `from torch.utils import cpp_extension` 的写法；只有代理路径覆盖不到时，才最小化地切到 `paddle.utils.cpp_extension` 或局部调整 include / lib / flags。
- `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` / pybind11：默认先保持原样，等编译或运行时真正失败了再定位具体缺口。
- `at::Tensor` / `c10::TensorOptions` / `torch::empty` 等 C++ API：优先依赖 compat headers；遇到缺口时只桥接单个 API 点。
- Python 入口与测试：运行时优先使用 `paddle.enable_compat(scope={...})`；短生命周期的 build script 可以用全局 `paddle.enable_compat()`。
- 分布式 / stream / device：先把运行时上下文边界接上，再看是否需要深入 `phi::GPUContext`、`ProcessGroup`、DLPack 或 stream wrapper。
- 分析 PFCCLab fork：输出要提炼成可复用的模式，覆盖 build / C++ / Python / tests 四层。

## 按需读取参考材料

不要一次性读取全部 reference。按当前任务只打开需要的文件：

| 当前任务 | 读取文件 |
|---|---|
| 先理解跨生态机制和分层口径 | [机制总览](references/mechanism-overview.md) |
| 实际迁移一个新仓库 | [迁移手册](references/migration-playbook.md) |
| 把错误定位到 Paddle 仓库内部 | [Paddle 内部锚点](references/paddle-internals.md) |
| 分析 PFCCLab fork 或复用既有迁移经验 | [生态库差异模式](references/ecosystem-diff-patterns.md) |
| 判断 compat gap、workaround、issue MRE | [compat 缺口处理](references/compat-gap-policy.md) |
| build/import 已通但运行时行为不一致 | [运行时调试](references/runtime-debugging.md) |

## 输出要求

- 明确列出哪些文件不需要动、哪些文件需要改、每一处改动对应哪一层。
- 如果需要 workaround，必须写清楚覆盖范围、删除条件，以及是否需要提 Paddle issue。
- 如果问题进入运行时对照阶段，要指出第一次差异出现在哪一行、哪个调用点、属于哪一层。
- 如果分析的是现有 fork，要总结出可复用的迁移顺序，并把 diff 提炼成稳定模式。

## 完成前检查

- 没有无关的格式化、清理、重命名。
- 保留了上游目录结构和主要 API 形状。
- 运行时的 `enable_compat` 已尽量限定 `scope`；build script 的全局 compat 只用在构建入口。
- build/test 至少跑通了一条最小路径。
- compat gap 已经准备了 issue MRE，或在结果中明确写出了缺口与临时 workaround。
