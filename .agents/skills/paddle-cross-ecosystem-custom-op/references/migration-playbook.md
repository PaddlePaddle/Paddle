# 迁移手册

本手册默认目标是：在最大限度保留上游代码形状的前提下，把一个原生 PyTorch 自定义算子仓库接到 Paddle。

## 步骤 0：先让上游版本在 PyTorch 下可复现

迁移前先确认：

- 原仓库在自己的推荐环境下能成功 build / import / run。
- 至少有一条最小测试路径可以复现正确行为。
- 你知道哪几个文件是真正的 build 入口、调用入口和测试入口。

不要在“原仓库本来就跑不通”的状态下直接做 Paddle 迁移。

## 步骤 1：先分层，不要一上来就改代码

把仓库分成下面四层：

| 层 | 通常怎么处理 |
|---|---|
| 框架无关内核 / 算法 | 默认不动 |
| 构建与打包 | 往往是第一批要改的文件 |
| C++ compat API / 注册 | 先尽量保持原样，让 compat 层接住 |
| Python 包装 / runtime glue / tests | 第二批修改点，通常加 `enable_compat(scope={...})` |

先把“哪些文件不该动”说清楚，再动手。

## 步骤 2：优先改 build，而不是改 kernel

最常见的第一步不是改掉 `torch.utils.cpp_extension`，而是在 build script 顶部启用 Paddle 的 PyTorch proxy，让原始 import 形状尽量保持不变。

### 典型改法

```diff
+import paddle
+paddle.enable_compat()
 from torch.utils import cpp_extension
```

这样 `from torch.utils import cpp_extension` 会通过 proxy 走到 Paddle 的扩展构建实现，patch 面通常比直接改 import 更小，也更利于后续 rebase。

如果当前仓库的 import 顺序、构建工具或代理边界导致上述方式不能工作，再局部切到 Paddle 入口：

```diff
-from torch.utils import cpp_extension
+# TODO(<issue>): remove direct Paddle entry after torch.utils.cpp_extension proxy covers this build path.
+from paddle.utils import cpp_extension
```

或者在已经确认需要直接入口时使用：

```python
import paddle

paddle.enable_compat()

from paddle.utils import cpp_extension
```

### 为什么这里先加 `paddle.enable_compat()`

- 某些仓库的构建脚本除了 `cpp_extension`，还会顺带 import 其他 `torch` 模块。
- 提前启用 compat，可以让 build script 自己的 PyTorch 依赖尽量先被代理层接住。
- build script 通常是短生命周期入口，必要时使用全局 proxy 可以接受；库的运行时入口和测试仍应优先使用 `scope={...}`。

### build 层的最小修改原则

- 保留 package 名称和目录布局。
- 保留 `setup.py` / `pyproject.toml` 的主体结构。
- 先只加 compat 前置准备；只有实测失败时，才替换编译入口、include/lib 来源，以及少量必要的 flags。
- 不主动改版本号策略、打包布局、wheel 命名，除非迁移本身要求。

## 步骤 3：C++ 侧先“让 compat 头接住”，再局部桥接

很多库的 C++ 部分可以原样保留：

- `#include <ATen/Functions.h>`
- `#include <torch/library.h>`
- `TORCH_LIBRARY(...)`
- `TORCH_LIBRARY_IMPL(...)`
- pybind11 module 定义

先编译，确认真正缺的是哪一个 API，再局部修。

### 典型的单点桥接：`torch::empty` 缺口

原始代码：

```cpp
at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
```

如果 compat 层没有这个 API，可以只桥接这一点：

```cpp
auto paddle_size = a_contig.sizes()._PD_ToPaddleIntArray();
auto paddle_dtype = compat::_PD_AtenScalarTypeToPhiDataType(a_contig.dtype());
auto paddle_place = a_contig.options()._PD_GetPlace();
auto paddle_result = paddle::experimental::empty(
    paddle_size, paddle_dtype, paddle_place);
at::Tensor result(paddle_result);
```

这里的原则是：

- 保留原函数签名、调用路径和 surrounding logic。
- 只替换缺的那个 API。
- 不顺手重写整段 tensor logic。

## 步骤 4：注册代码默认先不动

通常先保持以下代码不变：

```cpp
TORCH_LIBRARY(extension_cpp, m) {
  m.def("muladd_cpp(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("muladd_cpp", &muladd_cpu);
}
```

以及 pybind11 入口：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ...
}
```

只有在以下场景才改：

- compat 的 `torch/library.h` 没覆盖到该注册模式。
- 库依赖了 PyTorch 特有的 dispatch / class registration / private registry 语义。
- 运行时已经明确落在 compat gap 上。

## 步骤 5：Python 入口和测试优先 scoped proxy

对库的实际入口、最小示例、测试脚本，优先使用 scoped compat：

```python
import paddle

paddle.enable_compat(scope={"extension"})

import extension
```

不要一开始就全局代理整个进程，除非：

- 仓库结构很散，import 路径很多，难以限定；
- 或者构建脚本 / 运行脚本本身就必须在全局代理下才能成功导入。

## 步骤 6：针对不同生态库，关注点不同

### A. 普通 Torch extension / custom op repo

优先看：

- `setup.py`
- `csrc/*.cc` / `*.cu`
- `extension/__init__.py`
- `test.py` 或最小示例

通常只需要：

- 在 build 入口加 compat，必要时才切编译入口
- 加 scoped compat
- 修少量缺失的 C++ API

### B. 运行时 glue 很重的生态库

例如 FlashInfer、DeepEP、TorchCodec、SonicMoE。

优先看：

- `torch.ops` / `torch.library` / `torch._dynamo` / `torch.profiler`
- distributed group / stream / event / device helpers
- 自定义 wrapper、monkey patch、private API 依赖

这类库不要一开始动 kernel。先找 glue layer。

### C. Kernel DSL / compiler 生态

例如 Triton、TileLang、TVM FFI。

优先看：

- DLPack 转换
- 当前 device / current stream 获取
- JIT compile cache
- profiler / runtime hooks
- 某些框架在导入阶段就假设了 PyTorch 已初始化 CUDA runtime

这类库常见改动点不是 kernel 本身，而是 runtime adapter。

## 步骤 7：验证顺序

推荐顺序：

1. `pip install . --no-build-isolation` 或等价 build 命令
2. 最小 import 测试
3. 单个最小功能测试
4. 再看更完整的 test suite

先做最便宜的 falsifiable check，不要一上来全量跑。

## 步骤 8：运行时不一致时，沿单测逐段对比

如果已经满足下面三个条件：

- 能编译
- 能导入
- 但结果、device、stream、分布式行为或性能路径与原始 PyTorch 不一致

不要立刻扩大 patch 面，先沿着一个最小单测做逐段对比。

推荐做法：

1. 选一个 upstream 已存在的最小测试，或者自己抽一个最小脚本。
2. 保证 PyTorch 版和 Paddle 版输入一致，包括随机种子、dtype、device、shape、环境变量。
3. 在 Python wrapper、custom op 调用前后、关键张量变换点、必要的 C++ 入口处加观测点。
4. 找到第一次出现差异的位置，再判断它更像落在四层机制中的哪一层。

详细做法见 [运行时调试](runtime-debugging.md)。

## 迁移时不要做的事

- 不要全仓机械地把 `torch` 替换成 `paddle`。
- 不要为了“顺眼”主动重排 import、格式化大文件、清理上游风格差异。
- 不要在没有 issue / TODO / 注释边界的情况下加大范围 workaround。
- 不要用宽泛的 `try/except/pass` 把 compat 问题吞掉。

## 官方最小示例

官方文档中的示例仓库是 `PFCCLab/cross-ecosystem-custom-op-example`。

它说明了一件很重要的事：

- 很多时候第一步只改 build script 和测试入口；
- build script 的首选改法通常是保留原有 `torch.utils.cpp_extension` import，只在前面加 `paddle.enable_compat()`；
- 真正需要改的 C++ 代码，往往只是一两个 compat 尚未覆盖到的 API 点；
- `TORCH_LIBRARY` 本身并不一定需要改。
