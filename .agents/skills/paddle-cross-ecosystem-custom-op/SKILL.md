---
name: paddle-cross-ecosystem-custom-op
description: 将原生 PyTorch 自定义算子库、Torch extension、TorchCodec/FlashInfer/DeepEP 这类生态库，以及 Triton/TileLang/TVM FFI 等 Kernel DSL 生态，以最小修改方式接入 PaddlePaddle 的技能。遇到以下场景务必使用：迁移外部算子库到 Paddle；分析 PFCCLab fork 相对上游的兼容改动；处理 paddle.enable_compat、paddle.utils.cpp_extension、TORCH_LIBRARY、torch.ops、at::Tensor/c10 compat 问题；为 compat gap 设计最小 workaround 与 Paddle issue 最小复现。
---

# Paddle 跨生态自定义算子迁移

## 目标

- 在不破坏上游同步能力的前提下，将原生 PyTorch 生态的自定义算子库接入 Paddle。
- 默认产出应同时包含迁移方案、最小修改边界、验证路径，以及 compat gap 处理策略。

## 核心原则

- **最小修改**：不主动格式化，不主动优化，不主动重构，不主动改公共 API。
- **上游优先**：所有改动都要兼顾后续 rebase / sync upstream。
- **兼容层优先**：优先复用 Paddle 现有 compat 机制，不要一开始就把整仓库 `torch` API 人工翻译成 `paddle` API。
- **缺口显式化**：遇到 compat gap 时，不要静默绕过或隐藏问题。先判断是 Paddle compat 缺口，还是生态库自己依赖了 PyTorch 私有行为。必要时准备 Paddle issue 最小复现，并保留最小 TODO + workaround。
- **验证闭环**：至少完成一条最小 build/test 路径验证，不只看 diff。

## 首轮动作

1. 识别上游仓库、当前 fork、默认分支和实际迁移分支；PFCCLab fork 常见默认分支是 `paddle`，不要盲目比较 `main...main`。
2. 并行分析当前 fork 相对上游的 diff；如果是全新迁移，则先按 build / C++ / Python / tests 分层分析源仓库。
3. 把代码切成四层：
   - 框架无关内核 / 算法
   - 构建与打包
   - C++ compat API / 注册
   - Python 包装 / runtime glue / tests
4. 默认只改后 3 层。第 1 层除非有明确 bug，否则不动。

## 迁移时如何判断哪些文件该动

通常不动：

- CUDA/C++ 核心 kernel 与算法逻辑
- 原有 schema 定义
- 大部分 `TORCH_LIBRARY` / pybind11 注册代码
- 上游目录结构与 Python package 形状

通常先看：

- `setup.py` / `pyproject.toml`
- 入口脚本、测试脚本、示例脚本
- `torch.ops` / `torch.library` / `torch._dynamo` / `torch.profiler` 使用点
- device / stream / distributed / DLPack / custom op registration glue

## 执行流程

1. 先读 [机制总览](references/mechanism-overview.md)。
2. 要开始实际迁移时，读 [迁移手册](references/migration-playbook.md)。
3. 要理解 Paddle 仓库里的真实锚点时，读 [Paddle 内部锚点](references/paddle-internals.md)。
4. 要从现有生态库提炼模式时，读 [生态库差异模式](references/ecosystem-diff-patterns.md)。
5. compat 报错或行为不一致时，读 [compat 缺口处理](references/compat-gap-policy.md)。
6. 如果已经能编译和导入，但运行时结果、device、stream、分布式行为或性能路径与原始 PyTorch 不一致，读 [运行时调试](references/runtime-debugging.md)。

## 具体规则

- `setup.py` / `pyproject.toml`：优先只加 `paddle.enable_compat()`，保留原有 `torch.utils.cpp_extension` import；只有代理路径不能覆盖时，才最小切到 `paddle.utils.cpp_extension` 或调整 include / lib / flags。
- `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` / pybind11：先保持原样，只有编译或运行真的失败时才改。
- `at::Tensor` / `c10::TensorOptions` / `torch::empty` 这类 C++ API：优先依赖 compat headers；遇到缺口时只对单个 API 做桥接，不做大规模 C++ 重写。
- Python 入口与测试：运行时优先 `paddle.enable_compat(scope={...})`，避免污染整个进程；短生命周期的 build script 可以使用全局 `paddle.enable_compat()` 来接住原有 `torch.utils.cpp_extension` 和其他顶层 `torch` import。
- 分布式 / stream / device：先查现有库是否已经有 Paddle 特定 hook 或 workaround，再决定是否需要接入 `phi::GPUContext`、`ProcessGroup`、DLPack 或 stream wrapper。
- 分析现有 PFCCLab fork 时：最终必须提炼可复用模式，不要只罗列 diff。

## 输出要求

- 明确列出“哪些文件不该动、哪些文件该动、为什么”。
- 如果需要 workaround，必须说明：
  - 为什么 compat 层当前不够
  - workaround 的最小边界
  - 后续可删除条件
  - 是否需要 Paddle issue
- 如果分析现有 fork，相比上游要总结出 build / C++ / Python / tests 四层上的稳定迁移模式。
- 如果处理的是运行时问题，要尽量指出第一次观察到差异的是哪一行、哪个调用点，以及它更像落在四层机制中的哪一层。

## 完成前检查

- 没有无关格式化、清理、重命名。
- 保留上游目录结构和主要 API 形状。
- 运行时 `enable_compat` 尽可能限定 `scope`；build script 若必须全局启用，要确认作用域只停留在构建入口。
- build/test 至少跑过一个最小路径。
- 若发现 compat gap，已经准备 issue MRE，或者在结果中明确指出缺失与临时 workaround。
