# PIR 执行器（PirInterpreter）

## 概览

PIR 执行器负责将编译好的 `pir::Program`（经过 Kernel 选择、Pass 优化后的 kernel-level IR）翻译为可执行的 Instruction 序列，通过依赖分析和多 stream 异步调度实现高效执行。

核心流程：`StandaloneExecutor` → `InterpreterCore` → `PirInterpreter`

## 入口：StandaloneExecutor

```
StandaloneExecutor(place, plan)
  │
  ├─ 遍历 Plan 中的每个 Job
  │   ├─ PdOpLowerToKernelPass（pd_op → pd_kernel/onednn_kernel）
  │   ├─ InplacePass（可选）
  │   └─ 创建 InterpreterCore → 内部持有 PirInterpreter
  │
  └─ Run(feed_names, feed_tensors)
        └─ 按 Job 顺序依次调用 InterpreterCore::Run()
```

每个 Job 对应一段需要执行的 Program。多数简单场景只有一个 Job，pipeline 并行等场景会有多个。

## PirInterpreter 执行流程

PirInterpreter 首次 `Run()` 时执行 Build 阶段（结果缓存，后续 Run 跳过），然后进入 Scheduling 阶段。

### Build 阶段

#### Step 1: BuildInstruction — 构建 Instruction 列表

遍历 `pir::Block` 中的每个 Operation，根据其所属 Dialect 创建对应的 Instruction：

| Dialect | Operation | Instruction |
|---------|-----------|-------------|
| `builtin` | `CombineOp` | `BuiltinCombineInstruction` |
| `cf` | `TuplePushOp` / `TuplePopOp` / `YieldOp` | 对应的控制流 Instruction |
| `pd_op` | `IfOp` | `IfInstruction`（含 true/false 两个子 PirInterpreter） |
| `pd_op` | `WhileOp` | `WhileInstruction`（含 body 子 PirInterpreter） |
| `pd_op` | `PyLayerOp` | `PyLayerInstruction` |
| `pd_op` | `HasElementsOp` / `AssertOp` / `SelectInput/OutputOp` | 各自对应的 Instruction |
| `pd_kernel` | 普通算子 | `PhiKernelInstruction` |
| `pd_kernel` | 旧算子 | `LegacyKernelInstruction` |
| `onednn_kernel` | OneDNN 算子 | `OneDNNPhiKernelInstruction` 等 |
| `cinn_runtime` | `jit_kernel` | `CinnJitInstruction` |
| `custom_kernel` | 自定义算子 | `CustomKernelInstruction` |
| `py_func` | Python 函数 | `PythonFunctionInstruction` |

控制流 Op（IfOp / WhileOp / PyLayerOp）会递归创建子 PirInterpreter 处理内部 Block。

#### Step 2: BuildInstructionDependences — 构建依赖 DAG

基于数据依赖（RAW/WAR/WAW）分析 Instruction 间的执行顺序，构建有向无环图（DAG）。传递性边被消除以减少冗余同步。

每条依赖边根据两端 Instruction 的 KernelType 分为：
- **SameThread**：同类算子（如 GPU→GPU），放入同一线程的 ready queue
- **DifferentThread**：异类算子（如 GPU→CPU），通过线程池 AddTask 跨线程调度

每个 Instruction 记录 `dependency_count`（入度），为 0 时可被调度执行。

#### Step 3: Stream 调度分析

`PirStreamAnalyzer::AnalyseAllEventInfo` 对 DAG 中的每条跨 stream 边分类：

| 类型 | 含义 | 同步方式 |
|------|------|---------|
| `kDirectRun` | 同一 stream 上的连续算子 | 无需同步（stream 内天然有序） |
| `kEventRun` | 不同 stream 之间的数据依赖 | `cudaEventRecord` + `cudaEventWait` |

每个 Instruction 在执行前会 `WaitEvent`（等待上游的跨 stream 事件），执行后 `RecordEvent`（通知下游）。

#### Step 4: Variable 引用计数

计算每个非 persistable Variable 的引用计数（被后续 Instruction 使用的次数）。引用计数归零时，GC 回收其内存。

### Scheduling 阶段

每次 `Run()` 进入异步调度循环：

```
RunInstructionBaseAsync:
  │
  ├─ 初始化：dep_count=0 的 Instruction 推入 SchedulingQueue（优先级队列）
  │
  └─ while queue 非空:
       ├─ Pop 优先级最高的 Instruction
       ├─ RunInstructionBase：WaitEvent → 执行 kernel → RecordEvent
       ├─ RunNextInstructions：
       │   ├─ 遍历 DifferentThread 下游 → dep_count-1，为 0 则 AddTask 到线程池
       │   └─ 遍历 SameThread 下游 → dep_count-1，为 0 则推入本线程 queue
       └─ GC：ref_count-1，为 0 则回收 Variable 内存
```

线程池驱动多个无依赖的 Instruction 并行派发。GPU 算子在 CUDA stream 上异步执行，线程池仅负责 kernel launch。

## GC（垃圾回收）

| GC 实现 | 使用场景 |
|---------|---------|
| `EventGarbageCollector` | GPU 模式默认，基于 CUDA event 判断何时安全释放 |
| `FastGarbageCollector` | CPU 模式，直接释放 |
| `AsyncFastGarbageCollector` | 异步快速释放 |
| `NoEventGarbageCollector` | 无 event 模式 |

WhileOp 还支持 **early GC**：在循环体内部，如果一个 Variable 的动态引用计数降为 1 且不被循环体后续使用，可以提前回收。

## CINN Kernel 执行

CINN 编译产物通过 `CinnJitInstruction` 执行：
1. 从 `CINNKernelInfo` 获取 `fn_ptr`（编译好的 CUDA kernel 函数指针）
2. 通过 `symbol_args_map` 收集输入输出 device pointer
3. 调用 `cuLaunchKernel` 执行

## 关键源码路径

| 模块 | 路径 |
|------|------|
| Python Executor 入口 | `python/paddle/base/executor.py` |
| StandaloneExecutor | `paddle/fluid/framework/new_executor/standalone_executor.cc` |
| InterpreterCore 统一入口 | `paddle/fluid/framework/new_executor/interpretercore.cc` |
| PirInterpreter | `paddle/fluid/framework/new_executor/pir_interpreter.cc` |
| Instruction 基类 | `paddle/fluid/framework/new_executor/instruction/instruction_base.h` |
| PhiKernelInstruction | `paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.cc` |
| CinnJitInstruction | `paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc` |
| 控制流 Instruction | `paddle/fluid/framework/new_executor/instruction/control_flow/` |
| PirStreamAnalyzer | `paddle/fluid/framework/new_executor/interpreter/stream_analyzer.cc` |
| 依赖分析 | `paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc` |
| GC 实现 | `paddle/fluid/framework/new_executor/garbage_collector/` |
| Scope（变量容器） | `paddle/fluid/framework/scope.cc` |
