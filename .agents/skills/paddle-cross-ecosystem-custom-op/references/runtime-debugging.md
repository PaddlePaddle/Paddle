# 运行时调试

本章处理的是这样一类问题：

- 已经能编译
- 已经能导入
- 但运行时报错、结果不对、device/stream 语义不对、分布式行为不对，或者性能路径明显偏离原始 PyTorch

这时不要继续盲改代码。目标应当变成：沿着一个最小单测，找到第一次出现差异的是哪一行、哪个调用点、以及它更像是四层机制中的哪一层出了问题。

## 调试目标

你要回答的不是“哪里看起来怪”，而是下面三个更具体的问题：

1. PyTorch 原始路径在哪一行开始产生某个关键状态？
2. Paddle 迁移路径在哪一行第一次偏离？
3. 这个偏离更像是：
   - C++ API 兼容层
   - 算子注册兼容层
   - Python 接口兼容层
   - Python API 代理层
   - 或者根本不是 compat 问题，而是库自己的逻辑改动带来的回归

## 推荐工作流

### 步骤 1：只选一个最小单测

优先顺序：

1. 上游已有的最小单测
2. 当前 fork 中最小且最稳定的单测
3. 自己抽出来的最小脚本

单测要满足：

- 输入小
- 路径短
- 容易固定随机种子
- 不依赖太多外围框架状态

不要一开始就跑整个 test suite。你需要的是一个能被反复插桩、快速复现的最小切口。

### 步骤 2：先让 PyTorch 与 Paddle 版可比较

在开始插桩前，先保证以下条件尽可能一致：

- 相同随机种子
- 相同输入 shape / dtype / device
- 相同环境变量
- 相同测试逻辑
- 非必要优化先关掉

如果连输入都没对齐，后面的逐行对比没有意义。

### 步骤 3：沿调用链分段插桩

推荐把观测点放在下面这些位置：

| 位置 | 你要看什么 | 对应更可能的问题层 |
|---|---|---|
| 测试入口 | 输入 shape、dtype、device、seed 是否一致 | 先排除环境差异 |
| Python wrapper 进入前后 | 参数有没有被改写、重排、转换 | Python 接口兼容层 |
| `torch.ops` / custom op 调用前后 | 是否真正调到了预期 operator | 算子注册兼容层 |
| C++ 入口函数 | 看到的 tensor metadata 是否与 PyTorch 一致 | C++ API 兼容层 |
| kernel 前后 | 算法结果何时开始偏 | 可能是内核或前面几层已经传错 |
| 返回到 Python 后 | 后处理、layout、cast、split、gather 是否偏 | Python 接口兼容层 |

### 步骤 4：做一张“逐行对照表”

不要只靠脑子记。建议在调试记录里维护一张表：

| 原始 PyTorch 路径 | Paddle 迁移路径 | 观察值 | 结论 |
|---|---|---|---|
| 测试里的某行调用 | 对应迁移行 | 输入一致 | 继续向下 |
| wrapper 某行处理 | 对应迁移行 | device 语义开始不同 | 先锁定 Python 接口层 |
| `torch.ops` 调用 | 对应迁移行 | operator 名称不一致 / 未注册 | 转查注册层 |

调试的本质是找“第一次偏离”，不是最后一次爆炸。

## 一个可复用的调试范式

假设你已经有一个最小测试 `test_case_xxx`，推荐按下面顺序推进：

1. 在 PyTorch 原始仓库跑一次，记下关键中间值。
2. 在 Paddle 迁移仓库跑同一个测试，确认最初输入完全一致。
3. 只在 Python wrapper 上加第一层观测点。
4. 如果 Python wrapper 一致，再去 operator 调用点加第二层观测点。
5. 如果 operator 也一致，再去 C++ 入口打印 tensor metadata。
6. 如果 C++ 入口一致，再看 kernel 前后或者返回 Python 后的后处理。

不要同时在十几个地方加日志。那会让你丢掉“第一次出现差异”的顺序信息。

## 如何判断更像是哪一层出了问题

### 更像 Python API 代理层

常见信号：

- `import torch` 的行为就不对
- 同一个模块在不同 `scope` 下表现不同
- 某个被代理模块没有按预期进入 Paddle 命名空间

这时优先看 `paddle.enable_compat(scope={...})`、blocked modules、导入顺序。

反过来说，如果 `import torch`、模块作用域、代理范围都正常，那就不要在这一层打转，继续往 Python wrapper 或注册层收缩。

### 更像 Python 接口兼容层

常见信号：

- wrapper 已经把参数改歪了
- cast、reshape、split、pack/unpack、metadata 处理逻辑在 Python 侧先偏了
- 结果进 C++ 前就已经和 PyTorch 不一样

这时优先比对 Python wrapper，而不是怀疑 kernel。

一个常见误区是：结果不对就立刻怀疑底层算子。很多情况下，问题其实是在 wrapper 里某个 cast、split、reshape 或 metadata 处理先偏了。

### 更像算子注册兼容层

常见信号：

- `torch.ops.xxx` 找不到算子
- 找到了算子，但 dispatch 到了错误实现
- schema、命名空间、注册顺序与预期不同

这时应沿着注册与 dispatch 路径查，而不是直接改测试。

如果你已经能证明 wrapper 调用名和上游一致，但运行时仍然找不到 operator，那基本上就该优先沿注册层往下查。

### 更像 C++ API 兼容层

常见信号：

- 已经进入同一个 C++ 函数，但看到的 sizes、dtype、device、layout 和 PyTorch 不同
- 某个 `at::*` / `torch::*` / `c10::*` 操作在语义上不等价
- data pointer、options、place、stream 语义不一致

这时优先判断是不是 compat 头覆盖不完整，或某个具体 API 需要最小桥接。

## 运行时问题的常见分型

### 结果不对，但不崩

优先看：

- Python wrapper 有没有先改写输入
- dtype / layout / device 有没有在中途悄悄变化
- 返回 Python 后有没有额外 post-process

### 找不到算子，或者调到了错误实现

优先看：

- namespace
- schema
- 注册顺序
- dispatch key 选择

### 只在 GPU、stream、distributed 路径出问题

优先看：

- current device / current stream 获取点
- event / communicator / group 初始化点
- 是否存在异步错误被后面一行才观察到

### 只在 benchmark / profiler / compile 路径出问题

优先看：

- 这些路径是不是依赖了 PyTorch 私有 API
- 主算子路径是否其实已经正常
- 问题是不是只在外围测试和 profiling harness

## 什么时候该抽成 Paddle issue MRE

出现下面任一情况，就应该开始准备最小复现，而不是继续扩大 patch：

- 你已经能证明 PyTorch 与 Paddle 在同一调用点上行为不同
- 偏差来自 compat 层公共行为，而不是当前库自己的特例
- workaround 开始在多个文件重复出现
- 需要依赖 Paddle 内部私有接口才能继续绕过去

## 与 `paddle-debug` skill 的关系

本章只覆盖跨生态迁移场景下的“最小单测逐段对比”。

如果问题已经明显落到 Paddle 核心实现、CUDA sticky error、分布式 runtime、复杂内核崩溃等更底层问题，请继续使用 `paddle-debug` skill 做更系统的调试和报告。
