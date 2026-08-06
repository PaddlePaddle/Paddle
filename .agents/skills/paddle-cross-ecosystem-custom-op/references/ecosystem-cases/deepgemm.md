# DeepGEMM：控制面在 build 与 runtime header

这类库的 kernel 与算法主体通常高度框架无关，迁移价值主要来自把补丁收敛在 build、宏和 runtime header 边界。

### 第一落点

- `setup.py`
- 编译标志
- runtime header

### 具体例子

上游用 `torch/version.h` 的版本宏做能力探测，Paddle compat 没有这个头。处理方式不是仿造版本宏，而是把每个 capability 假定显式固化并逐条确认（长期保留类别，除非 compat 提供版本宏）：

```diff
--- csrc/utils/compatibility.hpp
-#include <torch/version.h>
 // `torch::kFloat8_e4m3fn` is supported since PyTorch 2.1
-#define DG_FP8_COMPATIBLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 1))
+#define DG_FP8_COMPATIBLE true
 // `cuTensorMapEncodeTiled` is supported since CUDA Driver API 12.1
-#define DG_TENSORMAP_COMPATIBLE (CUDA_VERSION >= 12010)
+#define DG_TENSORMAP_COMPATIBLE true
```

runtime header 的前提宏是另一个固定落点：compat headers 里 `gpuStream_t` 的声明依赖 `PADDLE_WITH_CUDA`，在 include 之前补一行 define，再显式引入 stream 头（长期保留类别）：

```diff
--- csrc/jit/device_runtime.hpp
+#define PADDLE_WITH_CUDA // make sure gpuStream_t declaration
 #include <cublasLt.h>
-#include <torch/version.h>
 #include <ATen/cuda/CUDAContext.h>
+#include <c10/cuda/CUDAStream.h>
```

Python 侧唯一的成规模改动来自扩展产物命名差异：`paddle.utils.cpp_extension` 的产物是顶层模块而不是包内 `_C`，`__init__.py` 里改 import 并留一个 `_C = deep_gemm_cpp` alias，让上游其他子模块（如 `.mega`）不用跟着改（长期保留类别）。

### 优先查看的文件

- `setup.py`：看 build 入口如何最小切换（`enable_compat` + paddle CUDAExtension + `-DPADDLE_WITH_CUDA`）
- `csrc/jit/device_runtime.hpp`：看 device runtime、stream、环境前提
- `csrc/jit/compiler.hpp`：看 JIT 编译边界需要哪些宏与运行时条件
- `csrc/utils/compatibility.hpp`：看 capability 假定如何显式固化
- `csrc/python_api.cpp`：看 Python 到 C++ 的入口形状
- `tests/`：看最小验证是否覆盖真实用户路径

### 可复用结论

- 框架无关核心越多，patch 面越要克制
- build / header 经常就是足够的迁移边界
- capability 假定要通过最小验证逐条确认
