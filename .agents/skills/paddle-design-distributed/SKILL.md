---
name: paddle-design-distributed
description: "Use when working with Paddle's distributed training system: understanding parallelism strategies (DP, ZeRO, TP, PP, SP), semi-automatic parallel with ProcessMesh + shard_tensor, SPMD inference rules, pipeline scheduling, or auto_parallel Engine."
---

# Paddle 分布式训练

## 分布式范式速查

| 范式 | 核心思想 | 通信原语 |
|------|---------|---------|
| **Data Parallel** | 复制模型，切分数据，AllReduce 梯度 | AllReduce |
| **Group Sharded (ZeRO)** | Stage1 切 optimizer / Stage2 + 切 grad / Stage3 + 切 weight | Broadcast, ReduceScatter, AllGather |
| **Model Parallel (Tensor)** | Column Parallel 切权重列 / Row Parallel 切权重行 | AllReduce / AllGather |
| **Pipeline Parallel** | F-then-B / 1F1B 交错前反向 | Send / Recv (P2P) |
| **Sequence Parallel** | 沿 sequence 维度切分 LayerNorm/Dropout | AllGather / ReduceScatter |

## 三种编程范式

| 范式 | 入口 | 适用场景 |
|------|------|---------|
| **手动并行** | `fleet.meta_parallel` | 灵活但代码量大，适合深度定制 |
| **半自动动态图** | `ProcessMesh` + `shard_tensor` | 用户标注切分方式，框架自动推导通信，兼具易用性和灵活性 |
| **半自动静态图** | `auto_parallel.Engine` | 基于 PIR 做全局优化，追求极致性能 |

### 半自动动态图示例

```python
import paddle.distributed as dist
mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=["x"])
x = dist.shard_tensor(x, mesh, [dist.Shard(0)])  # 沿 dim 0 切分
```

## SPMD 推导规则

半自动并行的核心：用户只标注部分 Tensor 的切分方式（Placement），框架自动推导每个算子的输入输出应如何切分，必要时插入通信算子。

### 基于 Einsum 的计算类规则

1. **推导 Einsum 表示**：`matmul(X[M,K], Y[K,N]) -> Z[M,N]` → `mk,kn->mn`
2. **合并输入 dims_mapping**：同一 Einsum 轴的切分必须一致，冲突时 reshard
3. **推导输出 dims_mapping**：根据合并后的轴切分信息映射到输出

### 形状变换类规则

使用 DimTrans 系统描述维度映射：InputDim / Flatten / Split / Singleton。

## Pipeline 调度

| 调度策略 | 特点 |
|---------|------|
| **F-then-B** | 所有前向完毕再反向，简单但显存占用大 |
| **1F1B** | warm-up → steady（前反交替）→ cool-down，显存减少约 37.5% |

## 什么场景看什么文件

| 场景 | 参考文档 |
|------|---------|
| 分布式策略原理（DP/Sharded/MP/PP/SP） | [references/distributed-primer.md](references/distributed-primer.md) |
| SPMD 推导规则与 Pipeline 调度 | [references/spmd-rules.md](references/spmd-rules.md) |

## 源码入口

| 模块 | 路径 |
|------|------|
| 分布式总目录 | `python/paddle/distributed/` |
| ProcessMesh / shard_tensor API | `python/paddle/distributed/auto_parallel/` |
| auto_parallel Engine | `python/paddle/distributed/auto_parallel/high_level_api.py` |
| fully_shard（ZeRO-like） | `python/paddle/distributed/auto_parallel/fully_shard.py` |
| SPMD 推导规则 C++ | `paddle/phi/infermeta/spmd_rules/` |
| SPMD 规则注册 | `paddle/phi/infermeta/spmd_rules/rules.h` |
| Pipeline 调度 Pass | `python/paddle/distributed/passes/pipeline_scheduler_pass/` |
| 分布式策略配置 | `python/paddle/distributed/auto_parallel/strategy.py` |
| SPMD 规则测试 | `test/auto_parallel/spmd_rules/` |
