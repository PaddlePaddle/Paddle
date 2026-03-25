---
name: paddle-pir-cinn
description: "Use when working with Paddle's new IR system (PIR) or CINN compiler: understanding SSA-based Program structure, Dialect/Type/Attribute design, writing or debugging Passes, tracing the CINN compilation pipeline from GroupOp to CUDA kernel, or translating legacy ProgramDesc to PIR."
---

# PIR & CINN 编译器

PIR (Paddle Intermediate Representation) 是 Paddle 的新一代中间表示，采用 MLIR 风格的 SSA 设计；CINN 是基于 PIR 的算子编译器，将高层算子编译为高性能 CUDA Kernel。

## PIR 核心概念速查

| 概念 | 关键类 | 说明 |
|------|--------|------|
| **Type** | `TypeID` / `AbstractType` / `TypeStorage` / `Type` | 类型系统：TypeID 用 static 变量地址做唯一标识，Type 本质是指向 TypeStorage 的指针，相等性通过指针比较 |
| **Value** | `ValueImpl` / `OpResultImpl` / `OpOperandImpl` | SSA 值系统：OpResult 是算子输出(inline 0-5 / out-of-line)，OpOperand 通过侵入式双向链表管理 use-chain |
| **Operation** | `Operation` (连续内存布局) | 核心执行单元：`[OutOfLineResults | InlineResults | Operation | Operands]` 连续分配 |
| **Block/Region** | `Block` / `Region` | Block 持有 Operation 列表 + BlockArgument + terminator；Region 是 Block 的容器，约束 Value 作用域 |
| **Dialect** | `BuiltinDialect` / `PaddleDialect` / `CinnDialect` | 模块化容器：聚合一组 Type、Attribute、Op 定义，支持独立注册与扩展 |

## CINN 4 阶段编译流水线

```
PIR Program (pd_op.*)
  │
  ▼  Stage 1: Frontend
  ├── PdOp2CinnOpConverter (算子映射)
  ├── add_broadcast_to_elementwise_pass
  └── build_cinn_pass → cinn_op.group (按 OpPatternKind 融合)
  │
  ▼  Stage 2: Lowering (Backend)
  ├── PirCompiler → OpLower
  │   ├── Compute: pe::Relu → lang::Relu → ComputeOp::Make → ir::Tensor
  │   ├── Schedule: Op-level + Group-level (LoopAlignment/Inline/Reduce/Fusion/BindCuda)
  │   └── LowerToAstVec → LoweredFunc
  │
  ▼  Stage 3: CodeGen
  ├── ir::Module → CodeGenCUDA_Dev → CUDA source
  └── nvrtc::Compiler → CUDAModule → CUfunction
  │
  ▼  Stage 4: Execution
  └── JitKernelOp (CINNKernelInfo: fn_ptr + int_args_map)
      └── CinnJitInstruction → launch kernel
```

**OpPatternKind 融合优先级**：`kElementWise < kBroadcast < kInjective < kReduction < kOutFusible < kNonFusible`

## 什么场景看什么文件

| 场景 | 参考文档 |
|------|---------|
| 理解 PIR 类型系统、Dialect、Trait/Interface 设计 | [references/pir-basics.md](references/pir-basics.md) |
| 理解 Program/Value/Operation 内存结构、ProgramTranslator | [references/pir-program.md](references/pir-program.md) |
| 追踪 CINN 从 GroupOp 到 CUDA Kernel 的完整编译流程 | [references/cinn-pipeline.md](references/cinn-pipeline.md) |
| 理解 PIR 控制流 (IfOp/WhileOp)、反向 Stack 机制 | [references/control-flow.md](references/control-flow.md) |

## 源码入口 (L3)

- PIR 核心：`paddle/pir/include/pir/core/` — `type.h`, `value.h`, `operation.h`, `block.h`, `program.h`
- Dialect 定义：`paddle/fluid/pir/dialect/operator/ir/` — `pd_op.h`, `cinn_op.h`
- CINN 前端：`paddle/cinn/hlir/dialect/operator/transforms/` — `build_cinn_pass.cc`, `pd_to_cinn_pass.cc`
- CINN 后端：`paddle/cinn/hlir/framework/pir/` — `op_lowering_impl.cc`, `compilation_task.cc`
- CodeGen：`paddle/cinn/backends/` — `codegen_cuda_dev.cc`, `nvrtc/nvrtc_util.cc`
- 控制流：`paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc`
