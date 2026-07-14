# 运行时调试

本章的前提：仓库已经能 build、能 import，但运行时开始出现报错、结果偏差、`device` / `place` / `stream` 语义不一致，或者只在分布式和高性能路径上出问题。

这里的任务是把控制路径收缩到一个最小样本，找出**第一次语义偏离**发生在哪一层。

## 调试目标

一次有效的运行时排查，最后应该回答四个问题：

1. PyTorch 原始路径的关键状态是在哪一行建立的。
2. Paddle 迁移路径第一次偏离是在哪一行出现的。
3. 偏离发生时，张量的 `shape`、`dtype`、`place`、layout、stream、group 等关键上下文分别是什么。
4. 这个偏离更可能落在 Python 代理层、Python wrapper、注册调度层、C++ compat 层，还是库自身逻辑改动。

## 调试入口

### 步骤 1：只保留一个最小样本

优先顺序保持简单：

1. 上游已有的最小单测
2. 当前 fork 中最稳定的最小单测
3. 自己抽出来的最小脚本

这个样本应满足四个条件：

- 输入小，便于重复运行
- 路径短，调用链清楚
- 能固定随机种子
- 不依赖太多 benchmark、profiler、训练框架状态

### 步骤 2：先把两边的输入对齐

在开始插桩前，先把下面这些量打印出来并固定：

- 随机种子
- 输入 `shape`
- 输入 `dtype`
- 输入 `device` / `place`
- 环境变量
- 是否开启额外优化路径

如果这一步没有对齐，后面的逐行对照就没有结论价值。

## 阶段 1：先看 `place`

`place` 是跨生态迁移里最容易被忽略、又最容易把调试结论带偏的一层。很多段错误、非法访问、结果漂移，根因其实是张量根本不在你以为的那块内存上。

### 先确认三种内存语义

在 Paddle 路径里，至少要区分下面三类位置：

| 观测值 | 含义 | 调试时的理解 |
|---|---|---|
| `Place(cpu)` | 普通 host 内存 | 只能按 CPU 张量处理 |
| `Place(gpu_pinned)` | host pinned memory | 仍是 host 内存，只是便于 DMA / 异步拷贝 |
| `Place(gpu:x)` | device memory | 才能直接当作 GPU tensor 进入 CUDA 路径 |

`Place(gpu_pinned)` 和 `Place(gpu:x)` 要严格区分。前者的名字里虽然带 `gpu`，语义仍然是 pinned host memory。

### 为什么这一步在 Paddle 尤其重要

Paddle 的 Python 创建 API 在 `device` / `place` 没有显式传入时，会走当前 expected place。也就是说，`paddle.tensor(...)`、`paddle.to_tensor(...)`、某些 helper 内部创建张量时，实际落点可能跟当前 dygraph guard 或全局 expected place 绑定；在 GPU 环境下，这些调用完全可能直接落到 GPU。

这和很多 PyTorch 生态库的默认假设不同。上游 helper 如果默认"未指定 device 时先创建 CPU tensor"，迁到 Paddle 后，完全相同的调用方式也可能因为 expected place 在 GPU 上而直接得到 `Place(gpu:0)`。

因此，运行时对比里必须记录真实 `place`，不能只看调用代码长得像不像。

### expected place 的来源

调试时至少要分清 expected place 的两个来源：

- 当前 dygraph guard 或显式设备上下文
- Paddle 的全局 expected place

如果当前没有额外 guard，Paddle 会回到全局 expected place。GPU 版本且有可见设备时，这个默认值可以直接是 `CUDAPlace(0)`；设备不可用时才会退回 CPU。

这也是为什么同一段 helper 代码在 PyTorch 和 Paddle 下可能出现不同落点：

| 调用方式 | PyTorch 生态库的常见假设 | Paddle 迁移时需要实际确认的 |
|---|---|---|
| 创建张量时省略 `device` | helper 会先得到 CPU tensor | helper 会落到当前 expected place；GPU 环境下可能直接得到 `Place(gpu:0)` |
| 需要 host staging 时开启 pinned memory | helper 仍把它视作 host tensor | 结果可能是 `Place(gpu_pinned)`，后续 copy / stream / `data_ptr` 逻辑都要按 pinned host 处理 |

因此，建议在 Python wrapper 入口和 custom op 调用前各打一次：

- `paddle.framework._current_expected_place_()`
- 关键输入张量的 `tensor.place`

这些 underscored API 只适合作为调试观测点，不要把它们沉淀成生态库的长期 runtime workaround。

### pinned memory 也要单独确认

Paddle 的创建与拷贝路径显式支持 `gpu_pinned` / `CUDAPinnedPlace`，而且 `pin_memory=True` 会把结果收敛到 pinned memory 路径。数据管线和 reader 代码里也常见 CPU → CUDAPinned → CUDA 的 staging。

这意味着迁移库里的中间张量可能落在 `Place(gpu_pinned)`。如果上游代码把这类张量当作 device tensor 继续传给 CUDA runtime 或自定义 kernel，就很容易出现段错误、非法地址访问，或者更隐蔽的异步崩溃。

### Python 侧第一轮日志建议

第一轮只打最必要的信息：

```python
def dump_tensor(tag, tensor):
   print(
      tag,
      {
         "shape": list(tensor.shape),
         "dtype": str(tensor.dtype),
         "place": str(tensor.place),
         "stop_gradient": tensor.stop_gradient,
      },
   )


print("expected_place", paddle.framework._current_expected_place_())
dump_tensor("input", x)
```

先在 PyTorch 版和 Paddle 版的同一调用点各打一轮，再决定下一跳去 wrapper、注册层还是 C++ 入口。

## 阶段 2：沿调用链逐层收缩

推荐观测点如下：

| 位置 | 第一轮先看什么 | 更可能对应的层 |
|---|---|---|
| 测试入口 | seed、shape、dtype、place 是否一致 | 排除环境差异 |
| Python wrapper 前后 | 参数是否被改写、重排、cast、split | Python 接口兼容层 |
| `torch.ops` / custom op 调用点 | 调用名、namespace、参数顺序是否一致 | 算子注册兼容层 |
| C++ 入口 | sizes、dtype、device、layout、pointer 来源 | C++ API 兼容层 |
| kernel 前后 | 哪一步开始产生结果偏差 | kernel 或上游输入已错 |
| 返回 Python 后 | post-process、pack/unpack、gather、profiler glue | Python 接口兼容层 |

一次只扩一层。Python wrapper 没对齐前，不要先去 kernel 里铺满日志。

## 阶段 3：`data_ptr` 只能在确认 `place` 之后看

### 为什么 `data_ptr` 是高风险点

Paddle compat 里的 `at::Tensor::data_ptr()` 直接返回底层 `tensor_.data()` 指针。这个指针只表达"当前地址"，不表达这块地址对应的是 CPU、pinned host 还是 CUDA device。

所以只要有下面这类情况，`data_ptr` 就是高风险点：

- Python wrapper 或 helper 先把张量建在 `Place(cpu)`
- 创建路径带了 `pin_memory=True`
- 中间 staging 张量落在 `Place(gpu_pinned)`
- 上游代码默认"这里已经是 CUDA tensor"，直接把 `data_ptr` 交给 CUDA kernel、CUDA runtime、Triton runtime 或自定义 C API

这类问题的表现常常不会先变成稳定的 Python 异常，常见现象包括：

- 段错误
- `illegal memory access`
- 某一行之后才报出的异步 CUDA 错误
- 偶发崩溃或结果随机漂移

### 排查顺序

1. 先在 Python 侧确认输入张量真实 `place`。
2. 进入 C++ 后先记下 `tensor.device()`、`tensor.is_cuda()`、sizes、dtype。
3. 确认张量确实在目标设备上之后，再看 `data_ptr()` 和后续 kernel 调用。
4. 如果实际是 CPU 或 `gpu_pinned`，先回到 wrapper / 创建路径找谁改变了 `place`。

### 对 `gpu_pinned` 的额外判断

如果张量是 `gpu_pinned`，下一步要确认它属于代码设计里的 staging buffer，还是本应继续进入 device memory 的张量却停在了 pinned host 上。

前者通常需要继续沿 copy / stream / async 边界排查；后者通常说明 wrapper、helper、copy 路径或 TensorOptions 语义已经偏了。

## 阶段 4：把"第一次偏离"写成对照表

建议维护一张最小表格，不靠记忆推进：

| PyTorch 路径 | Paddle 路径 | 观察值 | 结论 |
|---|---|---|---|
| 测试入口 | 对应测试入口 | 输入一致 | 继续向下 |
| wrapper 某一行 | 对应迁移行 | `place` 开始不同 | 回查 Python wrapper / 创建路径 |
| `torch.ops` 调用 | 对应迁移行 | operator 名称或 schema 不一致 | 转查注册层 |
| C++ 入口 | 对应迁移行 | `device` 一致但 dtype 偏了 | 转查 compat API 或 wrapper |

这张表的目的很单纯：固定"第一次偏离"发生的位置，避免被最后一处崩溃带偏。

## 常见问题分型

### 结果不对，但不崩

先查：

- Python wrapper 有没有先改写输入
- `dtype` / layout / `place` 有没有中途变化
- 返回 Python 后有没有额外 post-process

### `torch.ops` 找不到算子，或者调错实现

先查：

- namespace
- schema
- 注册顺序
- dispatch key

### 只在 GPU、stream、distributed 路径出问题

先查：

- current device / current stream 获取点
- event / communicator / group 初始化点
- 是否有异步错误被后面一行才观察到
- copy 路径里是否经过 CPU 或 `gpu_pinned` staging

### 只在 benchmark、profiler、compile 路径出问题

先查：

- 这些路径是否依赖 PyTorch 私有 API
- 主算子路径是否已经正常
- 问题是否只存在于外围 harness

## 如何判断更像哪一层

### 更像 Python API 代理层

信号通常是：`import torch` 本身异常、`scope` 表现不稳定、代理模块没有进入目标命名空间。

### 更像 Python 接口兼容层

信号通常是：参数在进入 C++ 之前就已经被改写，或者 `shape`、`dtype`、`place` 在 wrapper 中途就偏了。

### 更像算子注册兼容层

信号通常是：调用名一致，但 operator 找不到、schema 不匹配，或者 dispatch 落到错误实现。

### 更像 C++ API 兼容层

信号通常是：已经进入同一个 C++ 函数，但 sizes、dtype、device、layout、pointer 语义和 PyTorch 不一致。

## 什么时候该抽成 Paddle issue MRE

出现下面任一情况，就应该开始整理最小复现：

- PyTorch 与 Paddle 在同一调用点上行为已经明确分叉
- 问题来自 compat 公共行为，不属于当前库自己的特例
- workaround 开始在多个文件扩散
- 继续推进已经需要依赖 Paddle 内部私有接口

## 与 `paddle-debug` skill 的关系

本章只覆盖跨生态迁移场景里的最小样本逐层对照。

如果问题已经落到 Paddle 核心实现、复杂 CUDA 崩溃、sticky error、分布式 runtime 深层异常等更底层范围，继续交给 `paddle-debug` skill 做专项调试和报告。
