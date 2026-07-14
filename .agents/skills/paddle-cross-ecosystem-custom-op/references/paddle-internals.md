# Paddle 内部锚点

当迁移问题需要深入 Paddle 仓库内部时，先按所属抽象层定位。下面这些锚点对应的是 compat 机制里最稳定的控制面。

## Python 代理层

| 路径 | 作用 | 适用场景 |
|---|---|---|
| `python/paddle/__init__.py` | 对外暴露 `enable_compat` / `disable_compat` | 确认公开 API 入口 |
| `python/paddle/compat/proxy.py` | `torch` import 代理、scope、blocked modules、guard 实现 | import 行为异常、scope 不生效、模块代理边界不对 |
| `test/compat/test_torch_proxy.py` | Python 代理层测试 | 核对现有代理语义与边界 |

## build / cpp_extension 层

| 路径 | 作用 | 适用场景 |
|---|---|---|
| `python/paddle/utils/cpp_extension/cpp_extension.py` | `setup`、`CppExtension`、`CUDAExtension`、`BuildExtension` 实现 | `setup.py` 迁移、build 安装行为异常、shared library 命名问题 |
| `python/paddle/utils/cpp_extension/extension_utils.py` | include/lib 注入、compat include path、生成 Python stub、custom op registration | include path、link flags、custom op Python 包装异常 |
| `test/cpp_extension/` | cpp_extension 相关测试 | 核对 build / install / JIT 既有行为 |

`extension_utils.py` 会把 `paddle/phi/api/include/compat/` 相关目录加入 include path，因此很多 PyTorch 风格头文件会直接被 compat 头接住。

## C++ compat 头层

| 路径 | 作用 | 适用场景 |
|---|---|---|
| `paddle/phi/api/include/compat/ATen/Functions.h` | 常见 `ATen` 函数入口 | 某个 `at::*` 函数找不到 |
| `paddle/phi/api/include/compat/ATen/core/TensorBody.h` | `at::Tensor` 的 compat 包装，底层包的是 `paddle::Tensor` | tensor 方法、`data_ptr`、sizes、device、dtype 行为问题 |
| `paddle/phi/api/include/compat/c10/core/TensorOptions.h` | compat 版 `TensorOptions` | `options()`、device、dtype、memory_format 问题 |
| `paddle/phi/api/include/compat/torch/library.h` | `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` 宏与注册接口 | operator 注册、dispatch、schema 问题 |
| `paddle/phi/api/include/compat/torch/library.cpp` | compat operator/class registry 实现 | runtime lookup、dispatch、class registration 行为异常 |

## Python 到 C++ 的调度桥

| 路径 | 作用 | 适用场景 |
|---|---|---|
| `paddle/fluid/pybind/torch_compat.h` | 运行时把 Python 参数转成 compat `IValue` / `FunctionArgs`，并通过 registry 调度 | `torch.ops` / `TORCH_LIBRARY` 注册成功但调用异常 |

## compat 测试锚点

| 路径 | 作用 |
|---|---|
| `test/cpp/compat/torch_library_test.cc` | 基本 `TORCH_LIBRARY` / class registration 行为 |
| `test/cpp/compat/torch_library_dispatch_test.cc` | dispatch key 选择与 fallback 行为 |
| `test/cpp/compat/CMakeLists.txt` | compat 测试入口清单 |

## 典型路由方式

### 场景 1：`import torch` 行为异常

优先查看：

1. `python/paddle/compat/proxy.py`
2. `test/compat/test_torch_proxy.py`

### 场景 2：`setup.py` / `pip install .` 行为异常

优先查看：

1. `cpp_extension.py`
2. `extension_utils.py`
3. `test/cpp_extension/`

### 场景 3：C++ 编译时报某个 `at::*` / `c10::*` API 不存在

优先查看：

1. `ATen/Functions.h`
2. `ATen/core/TensorBody.h`
3. `c10/core/TensorOptions.h`

compat 头当前没有覆盖时，再决定是否加入最小桥接。

### 场景 4：`TORCH_LIBRARY` 编译通过但运行失败

优先查看：

1. `torch/library.h`
2. `torch/library.cpp`
3. `torch_compat.h`
4. `test/cpp/compat/torch_library_test.cc`

### 场景 5：device / stream / distributed 的私有行为依赖

先回到外部库自己的 glue layer，看当前 device / current stream、distributed group / communicator、profiler / dynamo / custom op registration 的私有假设如何组织；只有需要确认 compat 机制边界时，再进入 Paddle 内部文件。
