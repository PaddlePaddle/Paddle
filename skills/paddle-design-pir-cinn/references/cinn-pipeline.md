# CINN 编译流水线

CINN (Compiler Infrastructure for Neural Networks) 将 PIR Program 中的算子子图编译为高性能 CUDA Kernel。整个流程分为 4 个阶段。

## Stage 1: Frontend（前端）

前端负责将 PIR 中的 Paddle 算子映射为 CINN 算子，并进行算子融合。

### 1.1 PdOp2CinnOpConverter

将 `pd_op.*` 算子转换为 `cinn_op.*` 算子。大部分映射是一对一的（如 `pd_op.relu` → `cinn_op.relu`），少数需要拆分或重组。

### 1.2 add_broadcast_to_elementwise_pass

为 elementwise 类算子显式插入 broadcast 算子。Paddle 算子隐式支持 broadcast 语义，但 CINN 后端要求 shape 严格匹配，因此前端需要补齐。

### 1.3 build_cinn_pass：算子融合

核心 Pass，将可融合的算子聚合为 `cinn_op.group`（GroupOp）：

```
pd_op.relu → pd_op.add → pd_op.sigmoid
            ↓ build_cinn_pass
cinn_op.group {
  cinn_op.relu → cinn_op.add → cinn_op.sigmoid
  cinn_op.yield(...)
}
```

### OpPatternKind 融合规则

每个算子被标记一个 `OpPatternKind`，决定融合策略：

| Kind | 含义 | 典型算子 |
|------|------|---------|
| `kElementWise` | 逐元素计算 | relu, add, multiply |
| `kBroadcast` | 含广播语义 | broadcast_to |
| `kInjective` | 单射映射（reshape/transpose） | reshape, transpose, slice |
| `kReduction` | 规约操作 | reduce_sum, reduce_max |
| `kOutFusible` | 规约但输出可继续融合 | softmax 中间步骤 |
| `kNonFusible` | 不可融合 | custom_call, sort |

融合决策的基本原则：
- `kElementWise` 可以与任何非 `kNonFusible` 的算子融合
- `kReduction` 作为消费者时，生产者必须是 `kElementWise` 或 `kBroadcast`
- `kNonFusible` 不参与融合，单独成组

## Stage 2: Lowering（后端下降）

将 GroupOp 中的 CINN 高层算子下降为 AST IR，并进行调度优化。

### 2.1 PirCompiler → OpLower

`PirCompiler` 为每个 GroupOp 创建 `CompilationTask`，内部通过 `OpLower` 执行 4 步流程：

```
LowerOps → DoOpSchedule → DoGroupSchedule → PostProcess
```

### 2.2 Compute：三层抽象

每个 CINN 算子的计算语义通过三层函数描述：

```
pe::Relu(input, output_name)           // 第1层：算子语义入口
  → lang::Relu(input)                  // 第2层：数学表达式
    → lang::Compute(domain, lambda)    // 第3层：通用计算原语
      → ComputeOp::Make(name, lambda, shape, reduce_axis)
        → ir::Tensor (包含 ComputeOp 节点)
```

- **pe 层**（Paddle Expressions）：对外接口，处理算子特殊逻辑（如 reduce 轴处理）
- **lang 层**：纯数学表达式描述
- **Compute 层**：生成 AST IR 节点（`ComputeOp`），产出 `ir::Tensor`

### 2.3 AST IR 类型系统

CINN 内部使用自己的 AST IR（不同于 PIR）：

```
IrNode (基类)
├── ExprNode<T> → Expr (表达式节点)
│   ├── IntImm, FloatImm          // 立即数
│   ├── Add, Sub, Mul, Div, Mod   // 算术运算
│   ├── _Var_, _Tensor_           // 变量/张量引用
│   ├── Call, Cast                // 函数调用/类型转换
│   ├── For, IfThenElse           // 控制流
│   ├── ScheduleBlock, ScheduleBlockRealize  // 调度块
│   ├── Load, Store, BufferOp     // 内存操作
│   └── Block                     // 语句块
└── _Module_, _LoweredFunc_       // 顶层容器
```

`Expr` 与 `IrNodeRef` 的关系：`Expr` 是 `IrNodeRef` 的子类，`IrNodeRef` 内部持有 `shared_ptr<IrNode>`。

### 2.4 LowerToAstVec：从 Compute 到 LoweredFunc

```
GenerateFunctionBody
  → ScheduleBlockRealize(ScheduleBlock(compute_body))
    → 嵌套 For 循环包裹
→ AllocateBuffers (分配中间 buffer)
→ GenerateFunctionArgumentList (参数列表)
→ _LoweredFunc_::Make(name, args, body)
```

### 2.5 Schedule：调度优化

#### Op-level Schedule

针对单个算子的调度策略，主要处理 Reduce 类算子：

- Block Reduce / Warp Reduce / Discrete Reduce
- 根据 reduce 轴大小和数据量选择策略

#### Group-level Schedule（StaticShapeGroupScheduler）

针对整个融合组的全局调度，按顺序执行：

| 步骤 | 说明 |
|------|------|
| `DoLoopAlignment` | 对齐各算子的循环范围 |
| `DoComputeInline` | 将简单计算内联到消费者 |
| `OptimizeReduction` | 优化规约算子的并行策略 |
| `DoHorizontalLoopFusion` | 水平融合：合并独立的并行循环 |
| `DoVerticalLoopFusion` | 垂直融合：合并生产者-消费者循环 |
| `BindCudaAxis` | 绑定循环到 CUDA threadIdx/blockIdx |
| `AllocateStorage` | 分配 shared memory 和 local buffer |

## Stage 3: CodeGen（代码生成）

将调度优化后的 AST IR 编译为 CUDA 可执行代码：

```
ir::Module (包含多个 LoweredFunc)
  │
  ├─ CodeGenCUDA_Dev::Compile(module)
  │   → 遍历每个 LoweredFunc，生成 CUDA __global__ 函数源码
  │   → 处理 shared memory 声明、thread 同步等
  │
  ├─ nvrtc::Compiler::operator()(cuda_source)
  │   → 调用 NVIDIA NVRTC 运行时编译 API
  │   → 生成 PTX → cubin
  │
  └─ CUDAModule::GetFunction(func_name)
      → cuModuleLoadData + cuModuleGetFunction
      → 返回 CUfunction 句柄
```

`CodeGenCUDA_Dev` 继承自 `CodeGenC`，重写了 CUDA 特有的语法生成（如 `__global__`、`__shared__`、`threadIdx.x`、`__syncthreads()`）。

## Stage 4: Execution（执行）

### JitKernelOp

编译产物通过 `cinn_op.jit_kernel` 在 PIR 中表示：

```cpp
struct CINNKernelInfo {
  void *fn_ptr;                    // CUfunction 指针
  std::map<int, int> int_args_map; // 动态 shape 参数的位置映射
};
```

### PdOpLowerToKernelPass

将整个 PIR Program 中的 GroupOp 替换为 JitKernelOp：

```
cinn_op.group { ... }  →  cinn_op.jit_kernel (携带 CINNKernelInfo)
```

### CinnJitInstruction

执行器层面，`CinnJitInstruction` 负责实际的 Kernel 启动：

1. 从 `CINNKernelInfo` 获取 `fn_ptr`
2. 收集输入/输出 Tensor 的 device pointer
3. 处理动态 int 参数（shape 维度等）
4. 调用 `cuLaunchKernel` 执行

整个流水线的端到端路径：`pd_op.relu + pd_op.add` → GroupOp → Compute → AST IR → Schedule → LoweredFunc → CUDA source → PTX → CUfunction → `cuLaunchKernel`。
