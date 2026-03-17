# 执行器（Executor）执行流程

## 概览

Paddle 静态图的 Executor 负责将 ProgramDesc 翻译为实际的算子执行序列。当前默认使用**新执行器**（StandaloneExecutor / InterpreterCore），替代了早期的逐算子串行执行方式，通过依赖分析和异步调度实现更高效的执行。

## Python 侧入口

```
Executor.run(program, feed, fetch_list)
  │
  ├─ _ExecutorCache._get_program_and_executor(program, feed, fetch_list)
  │     ├─ 对 program 做 IR 优化（若启用）
  │     └─ 创建 / 缓存 StandaloneExecutor (C++ 对象)
  │
  └─ StandaloneExecutor.run(feed, fetch_list)
```

`_ExecutorCache` 以 program 的 cache key 为索引缓存已创建的 StandaloneExecutor，避免每次 `run()` 都重新构建。首次调用时创建，后续调用直接复用。

## C++ 侧：StandaloneExecutor

StandaloneExecutor 是 C++ 端的执行入口：

```cpp
StandaloneExecutor::StandaloneExecutor(place, plan)
  │
  ├─ 遍历 plan 中的每个 Job
  │     └─ 为每个 Job 创建一个 InterpreterCore (ProgramInterpreter)
  │           └─ ProgramInterpreter 持有 Block 中的 OpDesc 列表
  │
  └─ Run(feed_names, feed_tensors)
        └─ 按 Job 顺序依次调用对应 InterpreterCore 的 Run()
```

每个 Job 对应一段需要执行的 Program（或 Program 的一个 sub-block）。多数简单场景只有一个 Job。

## Pre-analysis 阶段（Build）

InterpreterCore 首次 `Run()` 时执行 Build 阶段，结果被缓存，后续迭代直接进入 Scheduling 阶段。Build 分三步：

### Step 1: BuildVariableScope

```
遍历 BlockDesc 中的所有 VarDesc:
  ├─ 在 Scope 中创建对应的 Variable 对象
  ├─ 设置 Variable 的 place / dtype 等属性
  └─ 记录 VarDesc.name → Variable* 的映射 (var_scope)
```

此步骤将静态描述（VarDesc）实例化为运行时对象（Variable），放入 Scope 的对应层级中。persistable 变量放入 root scope，非 persistable 变量放入 local scope。

### Step 2: BuildOpFuncList

为每个 OpDesc 构建一个 **OpFuncNode**，包含执行算子所需的全部信息：

```
遍历 BlockDesc 中的所有 OpDesc:
  │
  ├─ 1) 创建 OperatorBase 对象
  │     └─ OpRegistry 根据 OpDesc.type 查找并创建算子实例
  │
  ├─ 2) Kernel 选择
  │     ├─ 根据输入 Tensor 的 place / dtype 确定 KernelKey
  │     └─ 从 KernelFactory 中选择匹配的 Kernel 函数
  │
  ├─ 3) 数据传输算子插入（Data Transfer）
  │     ├─ 若输入 Tensor 的 place 与 Kernel 期望的 place 不一致
  │     │   → 插入 memcpy 算子（如 CPU→GPU）
  │     ├─ 若输入 dtype 不匹配 → 插入 cast 算子
  │     └─ 若输入 layout 不匹配 → 插入 transfer_layout 算子
  │
  └─ 4) 构建 OpFuncNode
        ├─ operator_base_: OperatorBase*     ← 算子实例
        ├─ input_index: vector<int>          ← 输入 Variable 在 Scope 中的索引
        ├─ output_index: vector<int>         ← 输出 Variable 的索引
        ├─ kernel_func_: KernelFunction      ← 实际执行函数
        ├─ dev_ctx_: DeviceContext*           ← 设备上下文（CUDA stream 等）
        └─ type_: OpFuncType                 ← kQueueSync / kQueueAsync / kQueueDataTransfer
```

**OpFuncType** 决定该算子将被派发到哪个执行队列：同步队列（CPU 算子）、异步队列（GPU 计算算子）或数据传输队列。

### Step 3: Convert（依赖分析与流调度）

Convert 阶段在 OpFuncNode 列表上进行全局分析，建立算子间的执行顺序约束。

#### 3.1 BuildOperatorDependences — 构建依赖 DAG

基于数据依赖分析，构建算子间的有向无环图（DAG）：

```
对于每对算子 (op_i, op_j)，其中 i < j，检查：
  ├─ RAW (Read-After-Write): op_i 写变量 V，op_j 读变量 V → 依赖
  ├─ WAR (Write-After-Read): op_i 读变量 V，op_j 写变量 V → 依赖
  └─ WAW (Write-After-Write): op_i 写变量 V，op_j 写变量 V → 依赖

构建完成后，shrink 传递性边：
  若 A→B→C 且 A→C，则移除 A→C 的直接边（减少冗余同步）
```

传递性边消除是一个关键优化：它减少了不必要的同步点，使更多算子能够并行执行。

#### 3.2 StreamAnalyzer::Schedule — 流调度分类

```
对 DAG 中的每条边 (op_i → op_j)，根据两端算子的 stream 属性分类：

  ├─ direct_run:       同一 stream 上的连续算子
  │                    → 无需额外同步（stream 内天然有序）
  │
  ├─ synchronize_run:  不同 stream 且无法用 event 精确同步
  │                    → 使用 cudaStreamSynchronize 全局等待
  │
  └─ event_run:        不同 stream 之间的数据依赖
                       → 使用 cudaEventRecord + cudaEventWait 精确同步
```

#### 3.3 Variable 生命周期分析

```
遍历所有 OpFuncNode，计算每个非 persistable Variable 的引用计数：
  ├─ 初始引用计数 = 该变量被后续算子使用的次数
  └─ 用于 Scheduling 阶段的垃圾回收 (GC)
```

引用计数为 0 时，Variable 的内存可被回收和复用，减少峰值内存占用。

## Scheduling / Execution 阶段

Build 完成后，每次 `Run()` 进入调度执行阶段，基于依赖 DAG 异步执行算子：

```
RunInstructionAsync 主循环:
  │
  ├─ 初始化：将所有 dep_count = 0 的算子推入 work queue
  │
  └─ while work queue 非空:
       │
       ├─ 1) Pop: 从 work queue 取出一个 OpFuncNode
       │
       ├─ 2) RunInstruction: 执行算子
       │     ├─ 准备输入输出 Tensor
       │     └─ 调用 kernel_func_(dev_ctx, inputs, outputs, attrs)
       │
       ├─ 3) Event Record: 若算子在 GPU stream 上执行
       │     └─ cudaEventRecord(event, stream)
       │
       ├─ 4) Decrement Deps: 遍历该算子的所有下游算子
       │     ├─ downstream.dep_count -= 1
       │     ├─ 若 edge 类型为 event_run:
       │     │   └─ cudaEventWait(upstream_event)
       │     └─ 若 downstream.dep_count == 0:
       │           └─ push downstream 到 work queue
       │
       └─ 5) GC Check: 遍历该算子的输出变量
             ├─ ref_count -= 1
             └─ 若 ref_count == 0:
                   └─ 回收 Variable 内存
```

work queue 的执行由线程池驱动，多个无依赖的算子可被并行派发到不同线程执行。对于 GPU 算子，实际计算在 CUDA stream 上异步进行，线程池仅负责 kernel launch。

## 跨 stream 同步机制

当两个有依赖关系的算子在不同 CUDA stream 上执行时：

1. **上游算子**执行完毕后调用 `cudaEventRecord(event, stream_A)`，在 stream_A 上记录一个 event。
2. **下游算子**执行前调用 `cudaEventWait(event)`（在 stream_B 上），确保 stream_A 上的计算已完成。
3. 这种机制比 `cudaStreamSynchronize` 更精细，只阻塞必要的依赖点而非整个 stream。

## 关键源码路径

| 模块 | 路径 |
|------|------|
| Python Executor 入口 | `python/paddle/base/executor.py` |
| StandaloneExecutor | `paddle/fluid/framework/new_executor/standalone_executor.cc` |
| ProgramInterpreter (InterpreterCore) | `paddle/fluid/framework/new_executor/program_interpreter.cc` |
| StreamAnalyzer | `paddle/fluid/framework/new_executor/stream_analyzer.cc` |
| OpFuncNode 定义 | `paddle/fluid/framework/new_executor/instruction/instruction_base.h` |
| Scope | `paddle/fluid/framework/scope.cc` |
| OperatorBase / OpRegistry | `paddle/fluid/framework/operator.cc`, `op_registry.cc` |
