# 生态库差异模式

这份总结按控制面分类整理了 PFCCLab 现有适配仓库的迁移经验，重点关注三个问题：

- 这类库的主要控制面在哪里
- 第一轮补丁应该落在哪一层
- 哪些部分适合保持不变

## 快速导航

- [这些 case 是什么](#先说清楚这些-case-是什么)
- [生态库清单与案例索引](#当前生态库清单与案例索引)
- [控制面分类](#先做控制面分类)
- [按需读取案例](#按需读取案例)
- [新库如何复用案例](#新库如何复用这些案例)

## 先说清楚这些 case 是什么

每个 case 都是特定时间点、特定分支状态的快照，不是固定 pattern。随着 Paddle compat 层持续演进，case 里的部分改动会被吸纳进 Paddle（proxy override、PyLayerContext shim 这类兼容性补丁最典型），届时对应 patch 应该删掉而不是照抄。所以复用这些 case 时：

- 要学的是**主要改动方式**：怎么定位控制面、补丁往哪层收敛、哪些部分坚决不动，而不是具体某一行 diff。
- 动手前先验证当前 compat 覆盖：case 里的某个 workaround 在最新 Paddle 下可能已经不需要了。带 `hasattr` / `try` 守卫的 shim 会自动短路，硬编码的桥接则需要人工确认后移除。
- case 中标注了「可能被 compat 吸纳」的改动，属于将来可删除的类别；API 面裁剪、lazy import 消除、梯度返回契约改写这类上游形状/语义调整则会长期保留。

## 当前生态库清单与案例索引

| PFCCLab fork | 上游 | 迁移分支 | 案例 |
|---|---|---|---|
| DeepEP | deepseek-ai/DeepEP | `paddle` | [DeepEP / HybridEP](ecosystem-cases/deep-ep.md) |
| DeepGEMM | deepseek-ai/DeepGEMM | `paddle` | [DeepGEMM](ecosystem-cases/deepgemm.md) |
| FlashMLA | deepseek-ai/FlashMLA | `paddle` | [FlashMLA / paddle-train](ecosystem-cases/flash-mla.md) |
| flashinfer | flashinfer-ai/flashinfer | `paddle` | [flashinfer](ecosystem-cases/flashinfer.md) |
| sonic-moe | Dao-AILab/sonic-moe | `paddle` | [sonic-moe / supersonic-moe](ecosystem-cases/sonic-moe.md) |
| supersonic-moe | Dao-AILab/sonic-moe | `paddle` | [sonic-moe / supersonic-moe](ecosystem-cases/sonic-moe.md) |
| quack | Dao-AILab/quack | `paddle/v0.3.7` | [quack](ecosystem-cases/quack.md) |
| flash-linear-attention | fla-org/flash-linear-attention | `paddle/kda-compact` | [flash-linear-attention](ecosystem-cases/flash-linear-attention.md) |
| MoonEP | MoonshotAI/MoonEP | `paddle` | [MoonEP](ecosystem-cases/moonep.md) |
| cudnn-frontend | NVIDIA/cudnn-frontend | `paddle/v1.26.0` | [cudnn-frontend](ecosystem-cases/cudnn-frontend.md) |
| fast-hadamard-transform | Dao-AILab/fast-hadamard-transform | `paddle-migrate-fast-hadamard-transform` | [fast-hadamard-transform](ecosystem-cases/fast-hadamard-transform.md) |
| tilelang | tile-ai/tilelang | `paddle` | [tilelang-paddle](ecosystem-cases/tilelang.md) |
| PaddleCodec | meta-pytorch/torchcodec | `paddle` | [paddlecodec](ecosystem-cases/paddlecodec.md) |

实际做 diff 前先确认 `parent` 和默认分支。PFCCLab 适配仓库的迁移分支通常是 `paddle`，而上游多为 `main`；也有按上游版本号命名的迁移分支（如 `paddle/v0.3.7`），base 是上游对应 release tag，升级时在新 tag 上重放补丁栈开新分支。

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
| HybridEP（DeepEP 分支） | 通信引导 / 拓扑探测 / JIT 构建链 | 双 setup 拆分与 pybind distributed bridge | HybridEP 后端 kernel 与 JIT 生成代码 |
| tilelang-paddle | adapter / device / stream / DLPack | runtime adapter | lowering 与 DSL 主体 |
| paddlecodec | Python glue / private API shim | wrapper 与薄 shim | C++ custom op 主体 |
| flashinfer | runtime feature gate / device 语义 / workaround 边界 | wrapper 与创建路径 | kernel 主体 |
| FlashMLA | benchmark / profiler / harness | 验证层隔离 | 主算子路径 |
| FlashMLA（paddle-train） | build 入口的 compat 开关 | setup.py 一处 enable_compat | 其余全部（含 autograd 训练路径与上游测试） |
| sonic-moe | import-time patch / Triton runtime wrapper | import 与 runtime 边界 | Triton kernel 主体 |
| supersonic-moe | proxy override / PyLayer 梯度契约 / 作用域化 torch swap | import 入口与 autograd.Function 边界 | CuTe DSL 与 Triton kernel 本体 |
| quack | PyLayer ctx shim / CuTe DSL 导入与 runtime 边界 | `quack/__init__.py` 的 import-time shim 集合 | CuTe kernel 本体与 DLPack 张量入径 |
| flash-linear-attention | Python 包装层（PyLayer 契约 + 设备/分布式 glue） | 各级 `__init__.py` 裁剪出业务子集 + compat 入口 | `@triton.jit` 内核本体、构建打包 |
| MoonEP | 分布式引导（handle 交换）/ stream 句柄 / 测试启动器 glue | setup.py compat 开关 + tests/conftest.py 分布式入口 | CuTe DSL kernel 与通信算法主体 |
| cudnn-frontend | Python bindings 层（dtype 映射 / stream / 张量元数据） | python/cudnn 的 feature-detect 与 runtime helper | header-only C++ 与 pybind 主体 |
| fast-hadamard-transform | 直接改写式迁移（compat 成熟前的历史做法） | setup.py 与 Python 接口整体切换 | CUDA kernel（csrc/*.cu） |
| DeepGEMM | build / runtime header / macro 前提 | build 与 header | GEMM 内核与算法主体 |

选择第一落点时，可以直接参考这张表：

- diff 主要集中在 `setup.py`、编译标志、runtime header → 第一落点通常是 build / header
- diff 主要集中在 device、stream、group、communicator helper → 第一落点通常是 runtime glue
- diff 主要集中在 wrapper、private API shim、`torch._dynamo` / `torch.profiler` helper → 第一落点通常是 Python glue
- diff 主要集中在 `paddle_test/`、benchmark、profiler harness → 第一落点通常是验证层隔离

## 按需读取案例

- 分析清单中的现有 fork：读取表中对应的案例；变体分支与其主仓库按同一案例分析。
- 为新仓库复用迁移经验：先用控制面分类确定类型，再最多读取一到两个最接近的案例文件。
- 横向比较多个仓库时，按用户指定的范围读取相应案例。
- 案例是特定时间点的快照。应用任何 workaround 前都要重新验证当前 Paddle compat 是否已经覆盖。

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
