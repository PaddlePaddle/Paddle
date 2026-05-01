# 机制总览

本节尽量与官方文档《原理和迁移方式》保持同一套口径。核心点不是“把 PyTorch 代码改写成 Paddle 代码”，而是理解 Paddle 已经提供了哪些兼容层，让你知道问题更可能落在哪一层。

## 官方口径的四层兼容机制

官方文档按自底向上描述了四层支持。

### 1. C++ API 兼容层

这一层解决的是 C++ 侧常见的 `at::*`、`torch::*`、`c10::*` 调用问题，让已有的自定义算子实现尽量不必大改。

- 代码锚点：`paddle/phi/api/include/compat/`
- 常见入口：
  - `ATen/Functions.h`
  - `ATen/core/TensorBody.h`
  - `c10/core/TensorOptions.h`
- 迁移含义：
  - 先让已有 C++ 代码吃 compat 头。
  - 如果编译失败，再定位是哪个具体 API 没覆盖。
  - 只对缺失的点做最小桥接，不反向重写整段 tensor 逻辑。

一个关键事实是：`ATen/core/TensorBody.h` 里的 compat `at::Tensor` 底层包装的是 `paddle::Tensor`。所以很多上游代码的“形状”可以保留，真正变化的是底层映射，而不是调用者写法。

### 2. 算子注册兼容层

这一层解决的是“算子怎么被注册和调度”的问题。

- 代码锚点：
  - `paddle/phi/api/include/compat/torch/library.h`
  - `paddle/phi/api/include/compat/torch/library.cpp`
  - `paddle/fluid/pybind/torch_compat.h`
- 迁移含义：
  - 对 pybind11 注册的自定义算子，通常不需要主动改注册代码。
  - 对 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` / `torch.ops` 路径，先假设 compat 层能接住。
  - 只有在 schema、dispatch、class registration 或 runtime lookup 明确失败时，才把它当成注册层问题来修。

这也是为什么不少库即使完全保留 `TORCH_LIBRARY` 宏定义，仍然能在 Paddle 下被正确调用。

### 3. Python 接口兼容层

这一层解决的是 Python 端算子封装、wrapper、辅助函数、张量预处理与后处理中的接口兼容问题。

- 典型表现：
  - 包装代码里仍在用 PyTorch 风格的张量方法、device 语义、helper 函数。
  - 自定义算子并不是直接从测试调到 C++，中间还隔了一层较厚的 Python glue。
- 迁移含义：
  - 运行时问题经常不是 kernel 错了，而是 Python wrapper 的语义先偏了。
  - 如果一个库 heavily 依赖 `torch._dynamo`、`torch.profiler`、`torch.library` 或其他内部 Python glue，优先在这层做最小 shim。

这一层没有单一入口文件，因为它往往分散在外部库自己的 Python 封装中；真正要做的是识别“问题是不是还没进 C++ 就已经发生了”。

### 4. Python API 代理层

这一层解决的是导入和命名空间映射问题，也就是 `import torch` 如何在 Paddle 环境下继续工作。

- 对外入口：`python/paddle/__init__.py` 暴露的 `paddle.enable_compat()` / `paddle.disable_compat()`
- 具体实现：`python/paddle/compat/proxy.py`
- 迁移含义：
  - 对大多数外部库，最先做的是在入口脚本、测试脚本或构建脚本中加 `paddle.enable_compat()`。
  - 默认优先使用 `scope={...}`，把代理范围收敛到目标 package。
  - 这一层解决的是“import 和 Python 调用形状能不能继续沿用”，不是替代底层全部兼容工作。

## 四层之外，但迁移中经常最先动到的支撑点

### 构建与扩展工具

官方四层里没有把 build system 单独列成一层，但实际迁移时它往往是第一个改动点。

- 代码锚点：
  - `python/paddle/utils/cpp_extension/cpp_extension.py`
  - `python/paddle/utils/cpp_extension/extension_utils.py`
- 作用：
  - 通过 proxy 映射或直接替代 `torch.utils.cpp_extension`
  - 自动注入 Paddle include / lib 路径
  - 把 compat 头目录加入 include path

所以实践中常见顺序是：

1. 先在 build 入口加 `paddle.enable_compat()`，尽量保留原有 `torch.utils.cpp_extension` import 形状。
2. 再看 C++ API compat 是否已足够
3. 如果 proxy 路径不能覆盖，再最小切到 `paddle.utils.cpp_extension` 或局部调整 include / lib / flags
4. 最后看注册层和 Python 两层是否还存在语义差异

### TVM FFI / DLPack 支撑

官方文档还特别提到：对 TVM FFI 生态，Paddle 已经对 DLPack 协议提供了较好的支持，因此这类生态很多时候主要工作不在 kernel 本身，而在 Python 端 wrapper、current device/current stream，以及导入阶段的 runtime glue。

## 用这四层判断问题归属

### 编译期问题更像哪层

- 缺 `at::*` / `torch::*` / `c10::*`：优先看 **C++ API 兼容层**
- `TORCH_LIBRARY` / `torch.ops` 编译不过：优先看 **算子注册兼容层**
- `setup.py` / 安装行为不对：先看 build 支撑点，而不是四层本身

### 运行期问题更像哪层

- `import torch`、模块作用域、代理范围不对：优先看 **Python API 代理层**
- Python wrapper 结果已经偏了，但还没真正进 C++：优先看 **Python 接口兼容层**
- `torch.ops` 找不到算子、dispatch 不对、调用没落到预期实现：优先看 **算子注册兼容层**
- 进入 C++ 后 tensor metadata、dtype、device、layout 语义不一致：优先看 **C++ API 兼容层**

### 一个快速判断例子

假设你把一个简单 custom op 仓库迁到一半，现象是：

- `pip install . --no-build-isolation` 能过
- `import extension` 能过
- 但 `extension.muladd(x, y, z)` 在 Python 侧直接报“找不到 `torch.ops.extension_cpp.muladd_cpp`”

这时就不要先去怀疑 kernel 或 `at::Tensor`。更高概率是：

- schema/namespace/dispatch 没接好
- 或 Python wrapper 调用的 operator 名称与注册层不一致

也就是说，第一落点应当是 **算子注册兼容层**，然后才是 Python wrapper。本质上是先问“算子有没有被正确注册和找到”，而不是先问“算子算得对不对”。

## 参考验证点

- Python 代理测试：`test/compat/test_torch_proxy.py`
- `TORCH_LIBRARY` 兼容测试：`test/cpp/compat/torch_library_test.cc`
- dispatch 兼容测试：`test/cpp/compat/torch_library_dispatch_test.cc`
- cpp_extension 测试：`test/cpp_extension/`

如果你需要把一个具体报错进一步定位到 Paddle 仓库内部文件，直接转去 [Paddle 内部锚点](paddle-internals.md)。如果问题已经进入运行时对比阶段，继续看 [运行时调试](runtime-debugging.md)。
