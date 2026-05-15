# 生态库差异模式

这份总结按控制面分类整理了 PFCCLab 现有适配仓库的迁移经验，重点关注三个问题：

- 这类库的主要控制面在哪里
- 第一轮补丁应该落在哪一层
- 哪些部分适合保持不变

参考对照如下：

- PFCCLab/DeepEP 对照 deepseek-ai/DeepEP
- PFCCLab/tilelang-paddle 对照 tile-ai/tilelang
- PFCCLab/paddlecodec 对照 meta-pytorch/torchcodec
- PFCCLab/flashinfer 对照 flashinfer-ai/flashinfer
- PFCCLab/FlashMLA 对照 deepseek-ai/FlashMLA
- PFCCLab/sonic-moe 对照 Dao-AILab/sonic-moe
- PFCCLab/DeepGEMM 对照 deepseek-ai/DeepGEMM

实际做 diff 前先确认 `parent` 和默认分支。PFCCLab 适配仓库的迁移分支通常是 `paddle`，而上游多为 `main`。

推荐先查元数据，再 compare：

```bash
gh repo view PFCCLab/flashinfer --json parent,defaultBranchRef
gh api repos/flashinfer-ai/flashinfer/compare/main...PFCCLab:paddle --jq '.files[].filename'
```

不同仓库的 base/head 可能不一样，以上命令只是形状示例；不要在未确认分支前套用。

## 先做控制面分类

| 仓库 | 主控制面 | 第一落点 | 通常不动的部分 |
|---|---|---|---|
| DeepEP | distributed / communicator / stream | runtime glue | collective kernel 与算法主体 |
| tilelang-paddle | adapter / device / stream / DLPack | runtime adapter | lowering 与 DSL 主体 |
| paddlecodec | Python glue / private API shim | wrapper 与薄 shim | C++ custom op 主体 |
| flashinfer | runtime feature gate / device 语义 / workaround 边界 | wrapper 与创建路径 | kernel 主体 |
| FlashMLA | benchmark / profiler / harness | 验证层隔离 | 主算子路径 |
| sonic-moe | import-time patch / Triton runtime wrapper | import 与 runtime 边界 | Triton kernel 主体 |
| DeepGEMM | build / runtime header / macro 前提 | build 与 header | GEMM 内核与算法主体 |

选择第一落点时，可以直接参考这张表：

- diff 主要集中在 `setup.py`、编译标志、runtime header → 第一落点通常是 build / header
- diff 主要集中在 device、stream、group、communicator helper → 第一落点通常是 runtime glue
- diff 主要集中在 wrapper、private API shim、`torch._dynamo` / `torch.profiler` helper → 第一落点通常是 Python glue
- diff 主要集中在 `paddle_test/`、benchmark、profiler harness → 第一落点通常是验证层隔离

## DeepEP：控制面在分布式上下文

这类库最关键的对象是 ProcessGroup、communicator、stream、event 以及相关上下文。通信 kernel 通常不需要动，迁移工作主要集中在这些上下文如何获取、如何传入底层。

### 第一落点

- runtime glue
- distributed context bridge
- communicator / stream 初始化

### 一个具体例子

如果上游代码默认"拿到一个 PyTorch group 对象后，后续通信都建立在这套语义上"，迁移时应该先把 group 到 communicator/context 的桥接接好，再跑最小单测。

### 优先查看的文件

- `setup.py`：确认 build 入口如何声明分布式相关能力
- `csrc/deep_ep.hpp`：看 runtime context、communicator、stream 成员如何组织
- `csrc/deep_ep.cpp`：看 communicator/context 初始化落点
- `deep_ep/buffer.py`：看 Python 侧 group、event、stream 如何传到底层
- `tests/utils.py`：看最小分布式测试如何起环境和 group

### 可复用结论

- 先把 distributed glue 和上下文边界对齐
- stream / event / communicator 初始化是一等公民，不能绕过
- 依赖 Paddle 分布式内部接口时，要把依赖收敛在最小入口

## tilelang-paddle：控制面在 adapter 与 runtime

这类库的主体是编译器或 DSL，跨框架迁移时最常遇到的控制点是 current device、current stream、DLPack、JIT runtime 初始化。

### 第一落点

- runtime adapter
- device helper
- stream helper
- DLPack bridge

### 一个具体例子

如果某个 backend 在导入库时默认依赖已经初始化好的 CUDA runtime，迁移时通常需要补 runtime preload 或 adapter，让环境准备在导入阶段就稳定下来。

### 优先查看的文件

- `tilelang/__init__.py`：看导入阶段的 runtime preload 与环境准备
- `tilelang/jit/adapter/base.py`：看 current device/current stream 的共用入口
- `tilelang/jit/adapter/tvm_ffi.py`：看 backend 如何把框架张量送进 FFI
- `tilelang/contrib/dlpack.py`：看跨框架张量协议边界
- `tests_paddle/`：看当前已验证的 backend 路径

### 可复用结论

- adapter 层通常就是第一轮补丁的位置
- DLPack、device、stream 是最稳定的观察点
- backend 较多时，先把主路径跑通，再逐步扩展

## paddlecodec：控制面在 Python glue

这类库表面上是 C++ custom op，但实际迁移中更关键的是 Python 层对 `torch.ops`、`torch._dynamo`、buffer 创建、metadata 管理的依赖。

### 第一落点

- wrapper
- 薄 shim
- 私有 API 依赖边界

### 一个具体例子

如果上游只要求存在一个类似 `torch._dynamo` 的对象来屏蔽某段图优化，迁移时可以先放一个最小 shim，让运行路径继续前进，同时把边界记录清楚。

### 优先查看的文件

- `src/torchcodec/_core/ops.py`：看最厚的 Python glue
- `src/torchcodec/_core/CMakeLists.txt`：看 C++ 扩展最终如何链接到 Paddle 侧库
- `setup.py`：看打包入口的最小切换点
- `src/torchcodec/__init__.py`：看对外 API 形状
- `test_paddle/`：看当前验证覆盖了哪些用户路径

### 可复用结论

- Python glue 很厚时，薄 shim 是性价比最高的入口
- shim 要带边界与删除条件
- shim 数量开始扩散时，应回查 compat gap

## flashinfer：控制面在 runtime feature gate 与 workaround 边界

这类高性能推理库的 kernel 主体通常比较稳定，跨框架迁移更常见的工作集中在 device 语义、custom op registration、通信路径以及 runtime feature gate。

### 第一落点

- feature gate
- 张量创建路径
- wrapper 与 registration fallback

### 一个具体例子

如果某条路径只在创建张量时触发 Paddle 当前行为差异，最小补丁通常应落在这个创建点，并同步整理 issue MRE。

### 优先查看的文件

- `flashinfer/utils.py`：看 feature gate、device 判断、registration fallback
- `flashinfer/fused_moe/core.py`：看高频运行路径里的框架分支
- `flashinfer/comm/trtllm_ar.py`：看通信和张量创建路径
- `flashinfer/decode.py`：看用户入口如何把 device/place 一路传下去
- `tests/conftest.py`：看测试入口的 compat 范围与环境准备

### 可复用结论

- feature gate 是定位 runtime 分支的重要线索
- workaround 最适合停留在最小创建路径或最小 wrapper 层
- 同类 workaround 开始扩散时，要转向 compat gap 处理

## FlashMLA：控制面在验证层隔离

这类仓库的常见情况是主算子路径已经基本跑通，主要差异集中在测试、benchmark、profiler 和验证脚本。

### 第一落点

- Paddle 专用验证层
- benchmark / profiler harness

### 一个具体例子

如果 profiler 路径依赖 PyTorch 的私有上下文管理，而算子本身已经能正确计算，迁移工作更适合停留在 benchmark/test harness 层面。

### 优先查看的文件

- `setup.py`：确认主构建入口的修改规模
- `paddle_test/`：区分主实现问题与 Paddle 专用验证问题
- `paddle_test/kernelkit/bench.py`：看 benchmark/profiler 适配如何组织
- `flash_mla/flash_mla_interface.py`：看主算子入口是否已经稳定

### 可复用结论

- 主实现和验证层要分开判断
- 主路径已通时，外围验证体系适合独立收敛
- profiler 差异本身不能直接推导出算子主体有问题

## sonic-moe：控制面在 import-time patch 与 Triton runtime wrapper

这类库的高频工作点通常是 Triton runtime wrapper、import 阶段的框架假设、stream 与 DLPack 边界。

### 第一落点

- import-time patch
- runtime wrapper
- stream / DLPack helper

### 一个具体例子

如果某个 Triton helper 只要求在编译阶段看到一组熟悉的 `torch` 命名空间语义，最小补丁往往适合停在 wrapper 层或 import 边界。

### 优先查看的文件

- `sonicmoe/__init__.py`：看 import-time patch 是否集中在入口
- `sonicmoe/triton_utils.py`：看 Triton runtime 隔离层
- `sonicmoe/utils.py`：看 stream、DLPack、wrapper 共用逻辑
- `sonicmoe/moe.py`：看业务入口如何串起 runtime 假设
- `sonicmoe/functional/`：看 patch 是否已经扩散过深

### 可复用结论

- import-time patch 的重量直接反映 compat 边界压力
- Triton 生态常从 wrapper 与 runtime 边界切入
- monkey patch 需要配合删除条件和 compat gap 判断

## DeepGEMM：控制面在 build 与 runtime header

这类库的 kernel 与算法主体通常高度框架无关，迁移价值主要来自把补丁收敛在 build、宏和 runtime header 边界。

### 第一落点

- `setup.py`
- 编译标志
- runtime header

### 一个具体例子

如果上游只在编译入口和少量 runtime header 里假设了 PyTorch 扩展环境，迁移时最自然的做法就是把补丁收在 build 入口和必要宏上。

### 优先查看的文件

- `setup.py`：看 build 入口如何最小切换
- `csrc/jit/device_runtime.hpp`：看 device runtime、stream、环境前提
- `csrc/jit/compiler.hpp`：看 JIT 编译边界需要哪些宏与运行时条件
- `csrc/python_api.cpp`：看 Python 到 C++ 的入口形状
- `tests/`：看最小验证是否覆盖真实用户路径

### 可复用结论

- 框架无关核心越多，patch 面越要克制
- build / header 经常就是足够的迁移边界
- capability 假定要通过最小验证逐条确认

## 新库如何复用这些案例

### 第一步：先判断主控制面

新库通常会更接近以下四类之一：

- 普通 extension：build、compat 头、最小测试
- distributed / stream glue 较重：运行时上下文和通信桥接
- Python glue 较重：wrapper、私有 API 依赖、薄 shim
- DSL / compiler：adapter、DLPack、current device/current stream

### 第二步：复用判断顺序

真正可复用的是判断顺序：

1. 先定位主控制面
2. 再确定第一落点
3. 最后圈出适合保持不变的部分

### 第三步：用 rebase 能力做最终校验

如果迁移方案开始显著破坏 upstream rebase 能力，或者开始系统性改写上游 API 形状，说明当前补丁边界需要重新收缩。
