# Paddle 内部锚点

当你需要判断一个兼容问题到底归谁管，先看下面这些锚点。

## Python 代理层

| 路径 | 作用 | 什么时候看 |
|---|---|---|
| `python/paddle/__init__.py` | 对外暴露 `enable_compat` / `disable_compat` | 想确认公开 API 入口时 |
| `python/paddle/compat/proxy.py` | `torch` import 代理、scope、blocked modules、guard 实现 | Python import 行为不对、scope 不生效、某模块不该被代理时 |
| `test/compat/test_torch_proxy.py` | Python 代理层测试 | 想确认现有代理语义和边界时 |

## build / cpp_extension 层

| 路径 | 作用 | 什么时候看 |
|---|---|---|
| `python/paddle/utils/cpp_extension/cpp_extension.py` | `setup`、`CppExtension`、`CUDAExtension`、`BuildExtension` 实现 | 迁移 `setup.py`、build 安装行为不对、shared library 命名问题 |
| `python/paddle/utils/cpp_extension/extension_utils.py` | include/lib 注入、compat include path、生成 Python stub、custom op registration | include path、link flags、custom op Python 包装不对时 |
| `test/cpp_extension/` | cpp_extension 相关测试 | 想确认 build/install/JIT 的既有行为时 |

### build 层一个关键事实

`extension_utils.py` 会把 `paddle/phi/api/include/compat/` 相关目录加入 include path，所以很多 PyTorch 风格头文件其实是由 Paddle 的 compat 头接住的。

## C++ compat 头层

| 路径 | 作用 | 什么时候看 |
|---|---|---|
| `paddle/phi/api/include/compat/ATen/Functions.h` | 常见 `ATen` 函数入口 | 某个 `at::*` 函数找不到时 |
| `paddle/phi/api/include/compat/ATen/core/TensorBody.h` | `at::Tensor` 的 compat 包装；底层包的是 `paddle::Tensor` | tensor 方法 / data_ptr / sizes / device / dtype 行为问题 |
| `paddle/phi/api/include/compat/c10/core/TensorOptions.h` | compat 版 `TensorOptions` | `options()` / device / dtype / memory_format 问题 |
| `paddle/phi/api/include/compat/torch/library.h` | `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` 宏与注册接口 | operator 注册、dispatch、schema 问题 |
| `paddle/phi/api/include/compat/torch/library.cpp` | compat operator/class registry 的实现 | 运行时 lookup / dispatch / class registration 行为不对时 |

## Python 到 C++ 的调度桥

| 路径 | 作用 | 什么时候看 |
|---|---|---|
| `paddle/fluid/pybind/torch_compat.h` | 运行时把 Python 参数转成 compat `IValue` / `FunctionArgs`，并通过 registry 调度 | `torch.ops` / `TORCH_LIBRARY` 注册成功但调用异常时 |

## compat 测试锚点

| 路径 | 作用 |
|---|---|
| `test/cpp/compat/torch_library_test.cc` | 基本 `TORCH_LIBRARY` / class registration 行为 |
| `test/cpp/compat/torch_library_dispatch_test.cc` | dispatch key 选择与 fallback 行为 |
| `test/cpp/compat/CMakeLists.txt` | compat 测试入口清单 |

## 遇到问题时怎么定位

### 场景 1：`import torch` 行为不对

先看：

1. `python/paddle/compat/proxy.py`
2. `test/compat/test_torch_proxy.py`

### 场景 2：`setup.py` / `pip install .` 行为不对

先看：

1. `cpp_extension.py`
2. `extension_utils.py`
3. `test/cpp_extension/`

### 场景 3：C++ 编译时报某个 `at::*` / `c10::*` API 不存在

先看：

1. `ATen/Functions.h`
2. `ATen/core/TensorBody.h`
3. `c10/core/TensorOptions.h`

如果 compat 头没有，再考虑最小桥接。

### 场景 4：`TORCH_LIBRARY` 编译通过但运行失败

先看：

1. `torch/library.h`
2. `torch/library.cpp`
3. `torch_compat.h`
4. `test/cpp/compat/torch_library_test.cc`

### 场景 5：库依赖 device / stream / distributed 的 PyTorch 私有行为

先看仓库本身的 glue layer，再决定是否要进入 Paddle 内部。

很多生态库真正需要改的是：

- 当前 device / current stream 获取
- distributed group 和 communicator 初始化
- profiler / dynamo / custom op registration 的私有 API 假设

这类问题往往不在 kernel 本体里。
