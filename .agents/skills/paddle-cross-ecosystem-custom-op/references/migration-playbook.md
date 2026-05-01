# 迁移手册

本手册的目标很直接：在最大限度保留上游代码形状的前提下，把一个原生 PyTorch 自定义算子仓库接到 Paddle 上跑起来。

## 步骤 0：先确认上游基线

迁移前先确认三件事：

- 原仓库在推荐环境下能成功 build / import / run。
- 至少有一条最小测试路径可以复现正确行为。
- build 入口、调用入口、测试入口已经找全。

## 步骤 1：先分层，再动代码

按控制面把仓库分成四层：

| 层 | 处理原则 |
|---|---|
| 框架无关的内核 / 算法 | 默认不动 |
| 构建与打包 | 往往是第一批要改的地方 |
| C++ compat API / 注册 | 先让 compat 层接住 |
| Python 包装 / runtime glue / tests | 第二批要改的地方 |

这一步的输出应该包含两张清单：

- 当前不需要动的文件
- 当前最可能需要先改的文件

## 步骤 2：先让 build 跑通

很多仓库的第一处修改只需要落在 build script 顶部，让原始 import 语句继续生效。

### 典型改法

```diff
+import paddle
+paddle.enable_compat()
 from torch.utils import cpp_extension
```

这样 `from torch.utils import cpp_extension` 会通过 proxy 走到 Paddle 的扩展构建实现，改动面最小，也最利于后续 rebase。

如果当前仓库的 import 顺序、构建工具或代理边界需要直接入口，再局部切到 Paddle：

```diff
-from torch.utils import cpp_extension
+from paddle.utils import cpp_extension
```

直接切到 `paddle.utils.cpp_extension` 不一定就是 compat gap。只有当它是在绕过一个明确的 proxy / compat 缺口时，才需要在代码或结果里记录 TODO、删除条件和 issue MRE；如果它只是当前构建系统下更小的入口选择，把原因写清楚即可。

### build 层的控制原则

- 保留 package 名称和目录布局。
- 保留 `setup.py` / `pyproject.toml` 主体结构。
- 首轮只加 compat 前置准备；编译入口、include / lib 来源、flags 只在实测失败后再调整。
- 版本号策略、打包布局、wheel 命名保持与上游一致，除非迁移本身明确要求变更。

## 步骤 3：让 compat 头先接住 C++ API

很多库的 C++ 部分可以先按原状编译：

- `#include <ATen/Functions.h>`
- `#include <torch/library.h>`
- `TORCH_LIBRARY(...)`
- `TORCH_LIBRARY_IMPL(...)`
- pybind11 module 定义

先编译，确认真实缺口落在哪个 API，再做单点桥接。

### 典型的单点桥接：`torch::empty`

原始代码：

```cpp
at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
```

如果 compat 层当前没有这个入口，可以只桥接这一点：

```cpp
auto paddle_size = a_contig.sizes()._PD_ToPaddleIntArray();
auto paddle_dtype = compat::_PD_AtenScalarTypeToPhiDataType(a_contig.dtype());
auto paddle_place = a_contig.options()._PD_GetPlace();
auto paddle_result = paddle::experimental::empty(
    paddle_size, paddle_dtype, paddle_place);
at::Tensor result(paddle_result);
```

这里需要维持三条边界：

- 原函数签名保持不变
- 调用路径保持不变
- surrounding logic 保持不变

## 步骤 4：保持注册路径稳定

注册层默认先按上游原样继续工作，例如：

```cpp
TORCH_LIBRARY(extension_cpp, m) {
  m.def("muladd_cpp(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("muladd_cpp", &muladd_cpu);
}
```

以及：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ...
}
```

只有在 schema、dispatch、class registration 或 private registry 语义确实落到 compat gap 上时，才需要修改注册代码。

## 步骤 5：运行时入口优先用 scoped compat

对实际入口、最小示例、测试脚本，优先使用 scoped compat：

```python
import paddle

paddle.enable_compat(scope={"extension"})

import extension
```

build script 适合全局 compat；运行时入口更适合 scoped compat，这样更容易收敛问题边界。

## 步骤 6：按生态类型选择第一落点

### A. 普通 Torch extension / custom op 仓库

优先检查：

- `setup.py`
- `csrc/*.cc` / `*.cu`
- `extension/__init__.py`
- `test.py` 或最小示例

首轮改动通常集中在 build 入口、scoped compat，以及少量缺失的 C++ API。

### B. runtime glue 较重的生态库

例如 FlashInfer、DeepEP、TorchCodec、SonicMoE。

优先检查：

- `torch.ops` / `torch.library` / `torch._dynamo` / `torch.profiler`
- distributed group / stream / event / device helpers
- 自定义 wrapper、monkey patch、private API 依赖

这类库的第一落点通常是运行时上下文边界。

### C. Kernel DSL / compiler 生态

例如 Triton、TileLang、TVM FFI。

优先检查：

- DLPack 转换
- current device / current stream 获取
- JIT compile cache
- profiler / runtime hooks
- import 阶段的 CUDA runtime 初始化

这类库常见的首轮补丁集中在 runtime adapter。

## 步骤 7：按最小成本验证

推荐顺序：

1. `pip install . --no-build-isolation` 或等价 build 命令
2. 最小 import 测试
3. 单个最小功能测试
4. 再跑更完整的 test suite

每一步都要先做最便宜的验证，确认没问题了再扩大范围。

## 步骤 8：运行时不一致时，沿最小样本逐段对照

当 build 和 import 都跑通了，但运行结果、`place`、stream、分布式行为或性能路径开始出现偏差，下一步应切换到逐段对照模式。

推荐做法：

1. 选一个上游已有的最小测试，或者自己抽一个最小脚本。
2. 保证 PyTorch 与 Paddle 输入一致，包括随机种子、dtype、device / place、shape、环境变量。
3. 在 Python wrapper、custom op 调用点、关键张量变换点、必要的 C++ 入口处加观测点。
4. 记录第一次差异出现在哪一行、哪个调用点、属于哪一层。

详细做法见 [运行时调试](runtime-debugging.md)。

## 迁移边界

迁移过程中始终保持这些边界：

- `torch` 的写法优先保留，让 compat 层承担映射职责。
- import、目录布局、主要 API 形状优先保留。
- workaround 需要带 TODO、删除条件；只有确认是 Paddle compat 公共缺口时才准备 issue 信息。
- 无关的格式化、风格清理、重命名、顺手重构都不在迁移范围内。

## 官方最小示例

官方文档中的示例仓库是 `PFCCLab/cross-ecosystem-custom-op-example`。

它清楚展示了一个典型顺序：

- build script 先接入 compat
- 测试入口再接入 scoped compat
- C++ 侧只桥接少量 compat 尚未覆盖到的 API 点
- `TORCH_LIBRARY` 通常可以保持不变
