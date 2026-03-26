---
name: paddle-design-compiler
description: "Use when working with Paddle 3.0 compiler full pipeline: SOT (Symbolic Opcode Translator) for bytecode-level dy2st graph capture, PIR (Paddle IR) for SSA-based intermediate representation, CINN for fused CUDA kernel generation, operator decomposition (Prim), or the end-to-end flow from Python eager code to optimized GPU execution."
---

# Paddle 3.0 编译器全链路

Paddle 3.0 的编译器体系通过 **SOT → PIR → CINN** 三阶段流水线，将用户的动态图 Python 代码编译为高性能 GPU Kernel，实现「动态图编写、编译器加速」的开发体验。

## 全链路概览

```
用户 Python 代码（动态图 eager mode）
  │
  ▼  Stage 0: SOT 图捕获
  PEP 523 eval_frame 拦截 → OpcodeExecutor 字节码模拟
  → FunctionGraph / StatementIR
  → paddle.jit.to_static(full_graph=True) 编译子图
  → pir::Program
  │
  ▼  Stage 1: PIR Pass 优化
  pir::Program（SSA 形式的 pd_op.* 算子图）
  │  ├── ShapeOptimizationPass（InferSymbolicShape 动态 shape 符号推导）
  │  ├── 组合算子分解 (DecompInterface → primitive operators)
  │  └── 通用 Pass 优化（常量折叠、死代码消除等）
  │
  ▼  Stage 2: CINN 编译
  ├── PdOpToCinnOpPass / PdOpToDynamicShapeCinnOpPass（算子映射）
  ├── add_cinn_pass → cinn_op.group（算子融合）
  ├── OpLower（Compute + Schedule）→ LoweredFunc
  ├── CodeGenCUDA_Dev → CUDA source → NVRTC → CUfunction
  ├── CompilationCache（编译缓存，相同子图复用已编译 Kernel）
  │
  ▼  Stage 3: 执行
  PirInterpreter 调度 → CinnJitInstruction → cuLaunchKernel
```

**SOT → PIR 的衔接**：SOT 捕获的 StatementIR 被包装为 Python 函数后，通过 `paddle.jit.to_static(full_graph=True)` 再次走 AST Transformer 路径编译为 `pir::Program`（参见 `python/paddle/jit/sot/symbolic/compile_cache.py`）。这意味着 SOT 负责"图捕获"，而 `to_static` 负责"图编译"。

## SOT（Symbolic Opcode Translator）

SOT 是 Paddle 3.0 的动转静前端，在 Python VM 字节码层面拦截和模拟执行用户代码，精确捕获 Tensor 计算子图。相比旧的 AST Transformer 方案，SOT 能处理 numpy/Tensor 互操作、动态控制流、第三方库调用等复杂场景。

### 核心机制

```
Python Frame
  │
  ▼  PEP 523 eval_frame 拦截
PyInterpreterState.eval_frame
  │
  ▼
OpcodeExecutor（模拟 Python VM 字节码执行）
  │  ├─ Variable 体系 (TensorVariable, ConstantVariable, ContainerVariable, ...)
  │  ├─ Tracker 追踪来源 → 生成 Guard（缓存有效性校验）
  │  └─ SideEffect 记录副作用（全局变量修改、可变对象修改）
  ▼
FunctionGraph / StatementIR（记录算子调用）
  │
  ├─ 无 fallback: 完整子图 → to_static(full_graph=True) → pir::Program
  └─ fallback: 子图切分 → 可静态化部分编译 + 不可静态化部分 Python 执行
```

| 组件 | 说明 |
|------|------|
| **OpcodeExecutor** | 模拟 Python VM 执行字节码，不真正计算，而是追踪 Tensor 操作 |
| **Variable 体系** | 将 Python 对象包装为 Variable（TensorVariable / ConstantVariable / ContainerVariable / CallableVariable） |
| **Tracker** | 记录 Variable 来源（provenance），形成 DAG，用于生成 Guard |
| **Guard** | `Callable[[FrameType], bool]`，判断当前帧输入是否满足编译假设，用于缓存命中判断 |
| **FunctionGraph** | 收集 Tensor 相关操作，输出 StatementIR |
| **StatementIR** | 4 种语句类型（call_api / call_method / call_sir / call_layer），最终经 `to_static(full_graph=True)` 编译为 Program |
| **SideEffect** | 记录并回放模拟执行中对全局变量和可变对象的修改，保证语义等价 |
| **OpcodeInlineExecutor** | 跨函数边界模拟执行，实现子图跨函数融合 |

### Fallback 场景

| 缩写 | 全称 | 场景 |
|------|------|------|
| **DDCF** | Data-Dependent Control Flow | 控制流条件依赖 Tensor 值（如 `if x.sum() > 0`） |
| **UNSPS** | Unsupported Simulation | 无法模拟的 Python 操作（如某些 C 扩展、`.numpy()`） |
| **CDBL** | Custom Blacklist | 用户或框架标记的不转换函数（如产生 -1 shape 的算子） |
| **UNIMP** | Unimplemented Opcode | 尚未实现模拟的字节码指令 |

Fallback 是安全兜底：任何无法处理的情况退化为部分子图编译 + 部分 Python 执行，不会导致报错。

### 使用方式

```python
# full_graph=False（默认）：启用 SOT 模式（字节码级别捕获 + 自动 fallback）
# full_graph=True：使用传统 AST Transformer（要求整图可转）
net = paddle.jit.to_static(net)  # 默认 full_graph=False，即 SOT 模式
output = net(x)
```

## PIR（Paddle Intermediate Representation）

PIR 是 Paddle 3.0 的统一中间表示，采用 MLIR 风格的 SSA 设计，替代旧的 ProgramDesc/OpDesc 体系。

### 核心概念

| 概念 | 关键类 | 说明 |
|------|--------|------|
| **Type** | `TypeID` / `AbstractType` / `TypeStorage` / `Type` | 统一类型系统：TypeID 用 static 变量地址做唯一标识，Type 本质是指向 TypeStorage 的指针，相等性通过指针比较 O(1) |
| **Value** | `ValueImpl` / `OpResultImpl` / `OpOperandImpl` | SSA 值系统：OpResult 是算子输出（inline 0-5 / out-of-line），OpOperand 通过侵入式双向链表管理 use-chain |
| **Operation** | `Operation`（连续内存布局） | 核心执行单元：`[OutOfLineResults | InlineResults | Operation | Operands]` 连续分配 |
| **Block/Region** | `Block` / `Region` | Block 持有 Operation 列表 + BlockArgument + terminator；Region 是 Block 的容器，约束 Value 作用域 |
| **Dialect** | `BuiltinDialect` / `PaddleDialect` / `CinnDialect` | 模块化容器：聚合一组 Type、Attribute、Op 定义，支持独立注册与扩展 |
| **Trait/Interface** | `OpTraitBase` / concept-model 多态 | Trait 是静态标记，Interface 通过 concept-model 实现多态分派，替代 C++ 虚函数 |

### 核心 Dialect

| Dialect | 职责 | 典型内容 |
|---------|------|---------|
| `BuiltinDialect` | PIR 内置基础类型 | `Float32Type`, `Int64Type`, `VectorType`, `DenseTensorType` |
| `PaddleDialect` | Paddle 算子定义 | `pd_op.matmul`, `pd_op.relu`, `pd_op.conv2d` |
| `CinnDialect` | CINN 编译器专用 | `cinn_op.group`, `cinn_op.yield`, `cinn_op.generate_shape` |
| `ControlFlowDialect` | 控制流辅助 | `cf.yield`, `cf.stack_create`, `cf.tuple_push`, `cf.tuple_pop` |
| `PaddleDialect`（控制流部分）| 控制流算子 | `pd_op.if`, `pd_op.while` |

### PIR Program 结构

```
Program
├── weights: unordered_map<string, shared_ptr<Parameter>>
└── ModuleOp (顶层 Operation)
    └── Region[0]
        └── Block[0]
            ├── builtin.parameter("w")      → %0  (从权重表读取参数)
            ├── pd_op.matmul(%input, %0)    → %1
            ├── pd_op.if(%cond)              → %2
            │   ├── Region[0] (then)
            │   │   └── Block[0]: pd_op.relu(%1) → cf.yield
            │   └── Region[1] (else)
            │       └── Block[0]: pd_op.tanh(%1) → cf.yield
            └── builtin.set_parameter(%2, "out")
```

### 组合算子分解（Prim）

将高层算子分解为基础算子（primitive operators），降低编译器 / 分布式 / 新硬件适配成本：

- **前向分解**：`DecompInterface` → `call_decomp_rule()` → `composite.h`
- **反向分解**（VJP）：两条路径——`VjpInterface` 经 `call_vjp()` 处理前向 op 的反向；`DecompVjpInterface` 经 `call_decomp_vjp()` 分解反向 op。规则实现均在 `details.h`
- **CustomVJP**：为 sigmoid、log_softmax 等数值敏感算子提供手写反向

### PIR Pass 框架

PIR 提供 MLIR 风格的 Pass 基础设施，用于图优化：

- `Pass`：单个优化 Pass 基类，通过 `Run(Operation*)` 执行
- `PassManager`：管理 Pass 执行顺序，支持嵌套 Pipeline
- `PatternRewritePass`：基于 Pattern Matching 的重写 Pass，通过 `RewritePattern` 定义匹配和替换规则

## CINN 编译与执行

CINN（Compiler Infrastructure for Neural Networks）将 PIR Program 中的算子子图编译为高性能 CUDA Kernel，由 PirInterpreter 调度执行。当前默认走**动态 shape**主线。

### 编译流水线（含动态 shape）

```
PIR Program (pd_op.*)
  │
  ▼  Stage 1: Frontend（前端）
  ├── ShapeOptimizationPass（InferSymbolicShape 符号推导）
  ├── PdOpToCinnOpPass / PdOpToDynamicShapeCinnOpPass（算子映射）
  ├── add_broadcast_to_elementwise_pass（显式 broadcast 插入）
  └── add_cinn_pass → cinn_op.group（按 OpPatternKind 融合）
  │
  ▼  Stage 2: Lowering（后端下降）
  ├── PirCompiler → CompilationTask（per GroupOp）
  │   ├── CompilationCache 查询（命中则跳过编译）
  │   ├── OpLower：Compute → AST IR → Schedule
  │   ├── DynamicShapeGroupScheduler（动态 shape 调度）
  │   └── LowerToAstVec → LoweredFunc
  │
  ▼  Stage 3: CodeGen（代码生成）
  ├── ir::Module → CodeGenCUDA_Dev → CUDA __global__ source
  └── nvrtc::Compiler → PTX → cubin → CUfunction
  │
  ▼  Stage 4: Execution（执行）
  └── cinn_runtime.jit_kernel (CINNKernelInfo: fn_ptr + symbol_args_map)
      └── CinnJitInstruction → cuLaunchKernel
```

### OpPatternKind 融合规则

| Kind | 含义 | 典型算子 |
|------|------|---------|
| `kElementWise` | 逐元素计算 | relu, add, multiply |
| `kBroadcast` | 含广播语义 | broadcast_to |
| `kInjective` | 单射映射 | reshape, transpose, slice |
| `kReduction` | 规约操作 | reduce_sum, reduce_max |
| `kOutFusible` | 规约但输出可继续融合 | softmax 中间步骤 |
| `kNonFusible` | 不可融合 | custom_call, sort |

### Group-level Schedule（DynamicShapeGroupScheduler）

| 步骤 | 说明 |
|------|------|
| `DoLoopAlignment` | 对齐各算子的循环范围 |
| `DoComputeInline` | 将简单计算内联到消费者 |
| `OptimizeReduction` | 优化规约算子的并行策略 |
| `DoHorizontalLoopFusion` | 水平融合：合并独立的并行循环 |
| `DoVerticalLoopFusion` | 垂直融合：合并生产者-消费者循环 |
| `BindCudaAxis` | 绑定循环到 CUDA threadIdx/blockIdx |
| `AllocateStorage` | 分配 shared memory 和 local buffer |

### 编译缓存（CompilationCache）

CINN 对已编译的 GroupOp 结果进行缓存（基于 FusionInfo hash），相同结构的子图可直接复用已编译的 Kernel，避免重复编译开销。

### 执行（PirInterpreter）

编译完成的 Kernel 最终由 PirInterpreter 调度执行：

```
StandaloneExecutor
  └─ PirInterpreter (per Job)
       │
       ├─ Build（首次 Run，结果缓存）
       │   ├── 为每个 Op 构建 Instruction（Kernel 选择 + 数据传输插入）
       │   ├── 构建算子依赖 DAG → 传递性边消除
       │   ├── PirStreamAnalyzer 流调度分类（direct / event / sync）
       │   └── Variable 引用计数 → GC 生命周期管理
       │
       └─ Scheduling（每次 Run）
            ├── dep_count=0 的 Instruction 推入 work queue
            ├── 线程池并行派发 → kernel launch
            ├── 跨 stream 同步：cudaEventRecord + cudaEventWait
            └── ref_count=0 时回收 Variable 内存
```

CINN 编译产物通过 `CinnJitInstruction` 执行：从 `CINNKernelInfo` 获取 `fn_ptr`，收集输入输出 device pointer，调用 `cuLaunchKernel`。非 CINN 算子则通过 PHI Kernel 常规路径执行。

## 调试速查

| 场景 | 应关注的文件 |
|------|------------|
| SOT 捕获失败 / fallback 过多 | `python/paddle/jit/sot/opcode_translator/executor/opcode_executor.py` — 检查未支持的 opcode |
| SOT SIR 到 Program 编译失败 | `python/paddle/jit/sot/symbolic/compile_cache.py` — `to_static(full_graph=True)` 环节 |
| PIR 动态 shape 推导错误 | `paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc` |
| CINN 融合策略问题 | `paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc` |
| CINN 动态 shape 算子映射 | `paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.cc` — `PdOpToDynamicShapeCinnOpPass` |
| CINN 编译缓存命中 / 未命中 | `paddle/cinn/hlir/framework/pir/compilation_cache.cc` |
| CINN Schedule 调试 | `paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.cc` |
| CINN CodeGen CUDA 源码 | `paddle/cinn/backends/codegen_cuda_dev.cc` |
| 执行器 Kernel 启动 | `paddle/fluid/framework/new_executor/pir_interpreter.cc` |
| 执行器依赖分析 / 调度 | `paddle/fluid/framework/new_executor/interpreter/stream_analyzer.cc` |
| 执行器 Variable 内存泄漏 | `paddle/fluid/framework/new_executor/garbage_collector/` |

## 什么场景看什么文件

| 场景 | 参考文档 |
|------|---------|
| SOT 架构设计（eval_frame / OpcodeExecutor / Guard / Fallback） | [references/sot-design.md](references/sot-design.md) |
| PIR 类型系统、Dialect、Trait/Interface 设计 | [references/pir-basics.md](references/pir-basics.md) |
| PIR Program/Value/Operation 内存结构、ProgramTranslator | [references/pir-program.md](references/pir-program.md) |
| CINN 从 GroupOp 到 CUDA Kernel 的完整编译流程 | [references/cinn-pipeline.md](references/cinn-pipeline.md) |
| PIR 控制流（IfOp/WhileOp）、反向 Stack 机制 | [references/control-flow.md](references/control-flow.md) |
| PIR 执行器（PirInterpreter）、Instruction 调度、Stream 分析、GC | [references/executor.md](references/executor.md) |

## 源码入口

### SOT

| 模块 | 路径 |
|------|------|
| to_static 入口（full_graph 分发） | `python/paddle/jit/api.py` |
| eval_frame 入口 | `python/paddle/jit/sot/opcode_translator/eval_frame_callback.py` |
| OpcodeExecutor | `python/paddle/jit/sot/opcode_translator/executor/opcode_executor.py` |
| OpcodeInlineExecutor | `python/paddle/jit/sot/opcode_translator/executor/opcode_inline_executor.py` |
| Variable 体系 | `python/paddle/jit/sot/opcode_translator/executor/variables/` |
| Tracker | `python/paddle/jit/sot/opcode_translator/executor/tracker.py` |
| Guard | `python/paddle/jit/sot/opcode_translator/executor/guard.py` |
| FunctionGraph | `python/paddle/jit/sot/opcode_translator/executor/function_graph.py` |
| StatementIR | `python/paddle/jit/sot/symbolic/statement_ir.py` |
| SIR 编译缓存 | `python/paddle/jit/sot/symbolic/compile_cache.py` |
| SideEffect | `python/paddle/jit/sot/opcode_translator/executor/side_effects.py` |
| 符号 Shape 推导 | `python/paddle/jit/sot/symbolic_shape/` |

### PIR

| 模块 | 路径 |
|------|------|
| PIR 核心 | `paddle/pir/include/core/` — `type.h`, `value.h`, `operation.h`, `block.h`, `program.h` |
| IRContext / StorageManager | `paddle/pir/src/core/ir_context.cc`, `storage_manager.cc` |
| Dialect 基类 | `paddle/pir/include/core/dialect.h` |
| PaddleDialect | `paddle/fluid/pir/dialect/operator/ir/op_dialect.h` |
| 控制流 Dialect | `paddle/pir/include/dialect/control_flow/ir/cf_op.h`, `cf_type.h` |
| 控制流 Op 实现 | `paddle/fluid/pir/dialect/operator/ir/control_flow_op.h` |
| Shape Dialect | `paddle/pir/include/dialect/shape/` |
| ShapeOptimizationPass | `paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc` |
| InferSymbolicShape 接口 | `paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/` |
| Pass 框架 | `paddle/pir/include/pass/pass.h`, `pass_manager.h` |
| Pattern Rewrite | `paddle/pir/include/pattern_rewrite/pattern_match.h` |
| DecompInterface（Prim 前向分解接口）| `paddle/fluid/pir/dialect/operator/interface/decomp.h` |

### 组合算子（Prim）

| 模块 | 路径 |
|------|------|
| 前向分解规则 | `paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h` |
| 反向分解规则（VJP） | `paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h` |
| 分解调度入口 | `paddle/fluid/primitive/base/decomp_trans.cc` |
| Primitive 基础算子 | `paddle/fluid/primitive/primitive/primitive.h` |
| VJP 接口 | `paddle/fluid/primitive/vjp_interface/vjp.h` |
| Backend 适配 | `paddle/fluid/primitive/backend/backend.h` |

### CINN

| 模块 | 路径 |
|------|------|
| CINN 总入口 Pass | `paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc` |
| 算子映射（含动态 shape）| `paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.cc` |
| 算子融合 | `paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.cc` |
| PirCompiler | `paddle/cinn/hlir/framework/pir_compiler.cc` |
| OpLower 实现 | `paddle/cinn/hlir/framework/pir/op_lowering_impl.cc` |
| 编译任务 | `paddle/cinn/hlir/framework/pir/compilation_task.cc` |
| 编译缓存 | `paddle/cinn/hlir/framework/pir/compilation_cache.cc` |
| DynamicShapeGroupScheduler | `paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.cc` |
| CodeGen | `paddle/cinn/backends/codegen_cuda_dev.cc` |
| NVRTC 编译 | `paddle/cinn/backends/nvrtc/nvrtc_util.cc` |
| CINNKernelInfo 定义 | `paddle/cinn/hlir/framework/pir/utils.h` |
| JitKernelOp 定义 | `paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h` |
| AST IR 节点 | `paddle/cinn/ir/` |
| Schedule 原语 | `paddle/cinn/ir/schedule/` |

### 执行器（PIR-based）

| 模块 | 路径 |
|------|------|
| Python Executor 入口 | `python/paddle/base/executor.py` |
| StandaloneExecutor | `paddle/fluid/framework/new_executor/standalone_executor.cc` |
| InterpreterCore 统一入口 | `paddle/fluid/framework/new_executor/interpretercore.cc` |
| PirInterpreter | `paddle/fluid/framework/new_executor/pir_interpreter.cc` |
| ProgramInterpreter（旧 IR 兼容） | `paddle/fluid/framework/new_executor/program_interpreter.cc` |
| PirStreamAnalyzer | `paddle/fluid/framework/new_executor/interpreter/stream_analyzer.cc` |
| Instruction 定义 | `paddle/fluid/framework/new_executor/instruction/` |
| CinnJitInstruction | `paddle/fluid/framework/new_executor/instruction/` |
| Scope（变量容器） | `paddle/fluid/framework/scope.cc` |
| GC 实现 | `paddle/fluid/framework/new_executor/garbage_collector/` |
