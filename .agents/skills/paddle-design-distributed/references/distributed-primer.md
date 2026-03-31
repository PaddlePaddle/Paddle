# 分布式训练策略详解

## 核心原则：等价性

分布式训练的基本目标：**多卡训练的数学结果与单卡训练完全等价**。所有并行策略的设计都围绕这一等价性展开——通过切分计算和通信来还原单卡的语义。

## 集合通信原语

分布式训练依赖 NCCL 提供的集合通信原语：

| 原语 | 语义 | 典型用途 |
|------|------|---------|
| **Broadcast** | 一个进程的数据广播给所有进程 | 参数初始化同步 |
| **AllGather** | 每个进程收集所有进程的数据 | ZeRO Stage3 前向前收集完整权重 |
| **AllReduce** | 所有进程归约后广播结果 | DP 梯度同步 |
| **Reduce** | 所有进程归约到一个进程 | 聚合 loss |
| **ReduceScatter** | 归约后将结果切分分发 | ZeRO Stage2 梯度同步 |

**关键恒等式**：`AllReduce = ReduceScatter + AllGather`。理解这一点对分析 ZeRO 和 Sequence Parallel 的通信量至关重要。

## 三种编程范式

### 1. 手动并行 (`fleet.meta_parallel`)

开发者显式指定切分方式，手动插入通信算子。灵活但代码量大、易出错。

```python
import paddle.distributed.fleet as fleet
fleet.init(is_collective=True)
strategy = fleet.DistributedStrategy()
strategy.tensor_parallel = True
strategy.tensor_parallel_configs = {"tensor_parallel_degree": 2}
```

### 2. 半自动动态图 (`ProcessMesh` + `shard_tensor`)

用户标注 Tensor 的切分方式，框架自动推导通信。兼具易用性和灵活性。

```python
import paddle.distributed as dist
mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=["x"])
x = dist.shard_tensor(x, mesh, [dist.Shard(0)])  # 沿 dim 0 切分
```

### 3. 半自动静态图 (`auto_parallel.Engine`)

基于静态图 IR，框架做全局优化（算子切分、通信插入、调度优化）。适合追求极致性能的场景。

```python
from paddle.distributed.auto_parallel import Engine
engine = Engine(model, loss, optimizer, strategy=strategy)
engine.fit(train_dataset)
```

## Data Parallel（数据并行）

**思路**：每张卡持有完整模型副本，训练数据按 batch 维度切分。

**流程**：
1. 每张卡用不同的 mini-batch 做前向 + 反向
2. 梯度通过 **AllReduce** 同步（求平均）
3. 每张卡用相同的平均梯度更新参数

**通信量**：每次迭代 AllReduce 全部梯度，通信量 = 2 * model_size（Ring AllReduce 下）。

**局限**：模型必须完整放入单卡显存。

## Group Sharded（ZeRO 系列）

Group Sharded 是 ZeRO（Zero Redundancy Optimizer）在 Paddle 中的实现，渐进式减少显存冗余。

### Stage 1：切分优化器状态

- 每张卡只维护 1/N 的优化器状态（如 Adam 的 m, v）
- 梯度仍然 AllReduce
- 参数更新后 **Broadcast** 更新的参数分片

**显存节省**：优化器状态占总显存的大头（Adam 为参数量的 2 倍），Stage 1 将其降为 1/N。

### Stage 2：+ 切分梯度

- 梯度不再 AllReduce，而是 **ReduceScatter**（每张卡只保留自己负责的那部分梯度）
- 进一步减少梯度的显存占用

### Stage 3：+ 切分权重

- 参数也按卡切分，每张卡只存 1/N 的参数
- 前向/反向计算前通过 **AllGather** 收集完整参数
- 计算完成后释放非本卡参数

**额外通信**：相比 DP，Stage 3 增加了前向 + 反向各一次 AllGather，通信量约增加 50%，但显存占用可降至接近 1/N。

## Model Parallel（模型并行 / 张量并行）

将单个算子的权重矩阵切分到多卡。以线性层 `Y = XW` 为例：

### Column Parallel（列切分）

将权重 W 按列切分为 [W1, W2]，分布在 2 张卡上：
- 卡 0 计算 Y1 = X @ W1
- 卡 1 计算 Y2 = X @ W2
- 输出 Y = [Y1, Y2]（无需通信，或后续做 AllGather）

### Row Parallel（行切分）

将权重 W 按行切分，输入 X 也相应切分：
- 卡 0 计算 Y1 = X1 @ W1
- 卡 1 计算 Y2 = X2 @ W2
- 输出 Y = Y1 + Y2（需要 **AllReduce**）

**Transformer 中的典型组合**：MLP 的第一个线性层用 Column Parallel，第二个用 Row Parallel，首尾各一次 AllReduce（或 f/g 共轭算子对消前向/反向各一次 AllReduce）。

## Pipeline Parallel（流水线并行）

将模型按层分为多个 stage，分配到不同卡上。

### Naive 方式

一次只有一张卡在计算，其余空闲。GPU 利用率 = 1/N，不实用。

### F-then-B（先前向后反向）

将 mini-batch 拆分为多个 micro-batch：
1. 先依次执行所有 micro-batch 的前向
2. 再依次执行所有 micro-batch 的反向

**bubble 比例**：(num_stages - 1) / num_microbatches。需要同时保存所有 micro-batch 的激活值，**显存占用大**。

### 1F1B（One Forward One Backward）

交错执行前向和反向：
1. warm-up 阶段：依次填充流水线（只做前向）
2. steady 阶段：每做一次前向，紧接着做一次反向
3. cool-down 阶段：依次排空流水线（只做反向）

**优势**：稳态阶段只需保存 num_stages 个 micro-batch 的激活值，相比 F-then-B **显存减少约 37.5%**（4 stages 时）。bubble 比例与 F-then-B 相同。

## Sequence Parallel（序列并行）

Tensor Parallel 的扩展，专门针对 Transformer 架构中 **不在 Tensor Parallel 范围内** 的算子（LayerNorm、Dropout）。

**思路**：
- Tensor Parallel 区域：权重按列/行切分
- 非 Tensor Parallel 区域（LayerNorm, Dropout）：沿 **sequence 维度** 切分激活值

**通信转换**：
- 原 Tensor Parallel 需要 AllReduce → 替换为 ReduceScatter（进入 SP 区域）+ AllGather（离开 SP 区域）
- 通信量不变（AllReduce = ReduceScatter + AllGather），但 SP 区域的激活值显存降为 1/N

**收益**：在不增加通信量的前提下，减少了 LayerNorm/Dropout 区域的激活值显存占用。
