# 机制总览

跨生态迁移要维持一条完整的控制路径：先让 build 和 import 跑通，再让 Python wrapper 正确地把参数传到注册层，注册层把调用分发到 C++ 实现，最后 C++ compat 层维持 Tensor 和设备语义的一致性。官方文档把这套机制组织成四层。

## 官方口径的四层兼容机制

| 迁移问题 | 对应层 | 主要锚点 | 常见信号 |
|---|---|---|---|
| C++ 调用能否编译并保持 Tensor 语义 | C++ API 兼容层 | `paddle/phi/api/include/compat/` | `at::*` / `torch::*` / `c10::*` 缺失，或 device / dtype / place 语义偏差 |
| 算子如何注册与调度 | 算子注册兼容层 | `torch/library.h`、`torch/library.cpp`、`torch_compat.h` | `torch.ops` 找不到算子、schema 不匹配、dispatch 错位 |
| Python wrapper 是否保持参数与 metadata 语义 | Python 接口兼容层 | 外部库自己的 wrapper 与 helper | 参数重排、shape / dtype / place 在进入 C++ 前已经偏了 |
| `import torch` 如何映射到 Paddle | Python API 代理层 | `paddle.enable_compat()`、`python/paddle/compat/proxy.py` | import 行为异常、scope 边界不对、代理模块缺失 |

### 1. C++ API 兼容层

这一层负责处理 C++ 侧常见的 `at::*`、`torch::*`、`c10::*` 调用，让上游自定义算子代码尽量不改就能继续编译。

- 代码锚点：`paddle/phi/api/include/compat/`
- 常见入口：
  - `ATen/Functions.h`
  - `ATen/core/TensorBody.h`
  - `c10/core/TensorOptions.h`

需要注意的一点：compat `at::Tensor` 的底层包装对象是 `paddle::Tensor`。因此上游代码的调用方式通常可以保持不变，真正需要关注的是底层张量与设备语义如何映射到 Paddle。

迁移时动作要克制：先让现有 C++ 代码用上 compat 头文件，再定位具体缺失的 API，最后只桥接单个缺口。

### 2. 算子注册兼容层

这一层负责 schema、namespace、dispatch、runtime lookup，以及 `TORCH_LIBRARY` / `torch.ops` 路径能否继续正常工作。

- 代码锚点：
  - `paddle/phi/api/include/compat/torch/library.h`
  - `paddle/phi/api/include/compat/torch/library.cpp`
  - `paddle/fluid/pybind/torch_compat.h`

这也是为什么不少仓库在保留 `TORCH_LIBRARY`、`TORCH_LIBRARY_IMPL` 和 pybind11 入口的情况下，依然可以在 Paddle 下跑通最小路径。

判断标准很直接：如果 build 和 import 都已经跑通，但 `torch.ops.xxx` 查找失败、schema 对不上、dispatch 落错实现，第一轮检查就应该放在注册层。

### 3. Python 接口兼容层

这一层负责 Python wrapper、辅助函数、张量预处理与后处理的行为一致性。

控制面通常分散在外部库自己的 Python 文件里，尤其是：

- wrapper 里的参数重排、cast、reshape、split、pack / unpack
- device / place / stream helper
- `torch._dynamo`、`torch.profiler`、`torch.library` 等私有 glue

自定义算子的实际调用路径通常会先经过 Python wrapper，再进入 `torch.ops` 或 pybind glue。如果运行时问题在进入 C++ 之前就出现了 `shape`、`dtype`、`place` 偏差，优先检查这一层。

### 4. Python API 代理层

这一层负责 `import torch` 与命名空间映射，让外部库在 Paddle 环境下仍能按原始导入路径运行。

- 对外入口：`python/paddle/__init__.py` 暴露的 `paddle.enable_compat()` / `paddle.disable_compat()`
- 具体实现：`python/paddle/compat/proxy.py`

迁移时通常会在入口脚本、测试脚本或构建脚本中先启用 compat。运行时入口更适合 `scope={...}`；build script 作为短生命周期的入口，可以使用全局 compat 来接住顶层 `torch` import。

## 四层之外的两个支撑点

### 构建与扩展工具

官方四层没有把 build system 单独列成一层，但实际迁移里 build 往往是第一落点。

- 代码锚点：
  - `python/paddle/utils/cpp_extension/cpp_extension.py`
  - `python/paddle/utils/cpp_extension/extension_utils.py`
- 主要作用：
  - 映射或直接替代 `torch.utils.cpp_extension`
  - 注入 Paddle include / lib 路径
  - 把 compat 头目录加入 include path

这也是为什么很多仓库的第一处修改只需要在 build 入口前加一行 `paddle.enable_compat()`。

### TVM FFI / DLPack 支撑

对 TVM FFI、TileLang、Triton 一类生态，DLPack、current device、current stream、runtime preload 才是高频控制面。Paddle 已经对 DLPack 提供了较好的支撑，因此这类库的第一轮补丁往往集中在 adapter 和 runtime glue。

## 用四层判断问题归属

### 编译期

- 缺 `at::*` / `torch::*` / `c10::*` → 先查 **C++ API 兼容层**
- `TORCH_LIBRARY`、`torch.ops` 路径编译失败 → 先查 **算子注册兼容层**
- `setup.py`、安装行为、include / lib 注入异常 → 先查 **构建支撑点**

### 运行期

- import 行为、模块作用域、proxy 边界异常 → 先查 **Python API 代理层**
- wrapper 把参数或 `place` 改歪了 → 先查 **Python 接口兼容层**
- `torch.ops` 找不到算子或 dispatch 错位 → 先查 **算子注册兼容层**
- 进入 C++ 后 tensor metadata、dtype、device、layout、pointer 语义不一致 → 先查 **C++ API 兼容层**

## 一个快速判断例子

如果一个简单 custom op 仓库已经满足：

- `pip install . --no-build-isolation` 成功
- `import extension` 成功
- Python wrapper 调用时报"找不到 `torch.ops.extension_cpp.muladd_cpp`"

第一落点应放在 **算子注册兼容层**。此时需要核对 namespace、schema、operator name 和 dispatch 路径；然后再回看 Python wrapper 调用名是否与注册层一致。

## 参考验证点

- Python 代理测试：`test/compat/test_torch_proxy.py`
- `TORCH_LIBRARY` 兼容测试：`test/cpp/compat/torch_library_test.cc`
- dispatch 兼容测试：`test/cpp/compat/torch_library_dispatch_test.cc`
- cpp_extension 测试：`test/cpp_extension/`

如果需要继续深入 Paddle 仓库内部文件，转去 [Paddle 内部锚点](paddle-internals.md)。如果运行时已经进入逐段对照阶段，继续看 [运行时调试](runtime-debugging.md)。
