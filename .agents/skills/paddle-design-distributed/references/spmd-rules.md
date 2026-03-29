# SPMD 推导规则与 Pipeline 调度

## 概述

半自动并行的核心问题：用户只标注部分 Tensor 的切分方式（Placement），框架需要 **自动推导** 每个算子的输入输出应如何切分，并在必要时插入通信算子。这套推导机制称为 SPMD（Single Program Multiple Data）Inference Rules。

## 基本概念

### ProcessMesh

进程拓扑，描述参与并行计算的进程组织方式：

```python
# 4 个进程组成一维 mesh
mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=["x"])
# 4 个进程组成 2x2 mesh
mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
```

### dims_mapping

描述 Tensor 各维度与 ProcessMesh 维度的对应关系：

- `dims_mapping[i] = j` 表示 Tensor 的第 i 维沿 ProcessMesh 的第 j 维切分
- `dims_mapping[i] = -1` 表示 Tensor 的第 i 维 **不切分（Replicated）**

示例：shape=[B, S, H] 的 Tensor，dims_mapping=[0, -1, 1] 表示 batch 维沿 mesh 的 x 轴切分，hidden 维沿 mesh 的 y 轴切分，sequence 维不切分。

## 计算类规则（Matmul, Elementwise 等）

基于 **Einsum Notation** 推导，分三步：

### 步骤 1：推导 Einsum 表示

根据算子的计算语义写出 Einsum notation。例如：

- `matmul(X[M,K], Y[K,N]) -> Z[M,N]`：Einsum = `mk,kn->mn`
- `elementwise_add(X[M,N], Y[M,N]) -> Z[M,N]`：Einsum = `mn,mn->mn`
- `matmul(X[B,M,K], Y[B,K,N]) -> Z[B,M,N]`：Einsum = `bmk,bkn->bmn`

### 步骤 2：合并输入 dims_mapping

调用 `ShardingMergeForTensors()` 将所有输入 Tensor 的 dims_mapping 按 Einsum 轴合并：

- 同一 Einsum 轴对应的 dims_mapping 值必须一致（或其中一个为 -1）
- 冲突时：若两个输入对同一轴有不同的切分维度，需要 **reshard**（插入通信算子使切分一致）

### 步骤 3：推导输出 dims_mapping

调用 `GetDimsMappingForAxes()` 根据合并后的轴切分信息，映射到输出 Tensor 的各维度。

**示例**：`matmul(X[M,K], Y[K,N])` 在 2D mesh 上

- X 的 dims_mapping = [0, -1]（M 轴沿 mesh dim 0 切分）
- Y 的 dims_mapping = [-1, 1]（N 轴沿 mesh dim 1 切分）
- 合并后：m→0, k→-1, n→1
- 输出 Z 的 dims_mapping = [0, 1]（M 沿 mesh dim 0，N 沿 mesh dim 1）

## 形状变换类规则（Reshape, Squeeze 等）

形状变换不涉及计算，但输入输出的 shape 不同，需要用 **DimTrans** 系统描述维度映射关系。

### DimTrans 类型

| 类型 | 含义 | 示例 |
|------|------|------|
| **InputDim(i)** | 输出维度直接对应输入的第 i 维 | reshape [6,12] → [6,12] |
| **Flatten(dims)** | 输出维度由输入的多个维度合并而来 | reshape [2,3,4] → [6,4]：dim0 = Flatten([InputDim(0), InputDim(1)]) |
| **Split(input_dim, sizes, idx)** | 输出维度由输入某维度拆分而来 | reshape [6,4] → [2,3,4]：dim0 = Split(InputDim(0), [2,3], 0) |
| **Singleton** | 输出维度大小为 1，不对应任何输入维度 | unsqueeze |

**切分规则**：Flatten 时，只有最内层维度可以保留切分；Split 时，切分维度可以映射到对应的拆分片段。

## 开发与注册

### 接口文件

每个算子的 SPMD 规则实现在：
```
paddle/phi/infermeta/spmd_rules/{op_name}.h
paddle/phi/infermeta/spmd_rules/{op_name}.cc
```

需要实现两个函数：
- `SpmdInfo {Op}InferSpmd(const DistMetaTensor& x, ...)`：前向推导
- `SpmdInfo {Op}InferSpmdReverse(const DistMetaTensor& x, ..., const DistMetaTensor& out)`：反向推导（从输出推导输入）

### 注册

在 `paddle/phi/infermeta/spmd_rules/rules.h` 中使用宏注册：

```cpp
PD_REGISTER_SPMD_RULE(
    matmul,
    PD_INFER_SPMD(phi::distributed::MatmulInferSpmd),
    PD_INFER_SPMD(phi::distributed::MatmulInferSpmdReverse));
```

### 测试

测试位于 `test/auto_parallel/spmd_rules/`，使用 Python 调用验证推导结果。

## Pipeline 调度

Pipeline Parallel 将模型按层切分为多个 stage，每个 stage 分配到不同设备。调度策略决定 micro-batch 的执行顺序。

### F-then-B 调度

Job list 形如：`[F0, F1, F2, F3, B3, B2, B1, B0]`

所有前向执行完毕后再执行反向。简单但显存占用大（需保存所有 micro-batch 的激活值）。

### 1F1B 调度

Job list 形如：`[F0, F1, F2, F3, B0, F4, B1, F5, B2, ..., B_last]`

warm-up 阶段填充流水线，steady 阶段前向/反向交替执行。

### 程序切分

Pipeline Parallel 需要将完整程序切分为子程序：

| 子程序 | 内容 |
|--------|------|
| **LR** | 学习率调度 |
| **FORWARD** | 前向计算（按 stage 切分） |
| **BACKWARD** | 反向计算（按 stage 切分） |
| **OPT** | 参数更新（优化器 step） |

调度器根据 job list 按顺序执行子程序，通过 Send/Recv 实现 stage 间的激活值和梯度传递。

相关代码路径：
- 调度 pass：`python/paddle/distributed/passes/pipeline_scheduler_pass/`
- Job 定义：`python/paddle/distributed/auto_parallel/strategy.py`
