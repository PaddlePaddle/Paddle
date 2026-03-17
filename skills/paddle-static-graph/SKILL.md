---
name: paddle-static-graph
description: "Use when working with Paddle's static graph mode: understanding Program/Block/Op/Var data structures, tracing the executor lifecycle from graph construction to scheduling, debugging InterpreterCore issues, or analyzing operator dependency and variable lifetime management."
---

# Paddle 静态图模式

Paddle 静态图采用**先编译、后执行**的范式：Python 端构建 Program 描述计算逻辑，再由 Executor 将 Program 翻译为可调度的算子序列并执行。

## Program 结构速查

```
ProgramDesc
 └─ blocks_: vector<BlockDesc>
      ├─ BlockDesc [0] (global block)
      │   ├─ ops_: vector<OpDesc>     ← 前向 + 反向 + 优化器算子
      │   └─ vars_: map<string, VarDesc>  ← 参数、中间变量、梯度
      └─ BlockDesc [1..N] (sub-blocks, 用于 while_op / cond 等控制流)
```

- **ProgramDesc**：顶层容器，持有 protobuf 描述和 BlockDesc 列表。
- **BlockDesc**：逻辑作用域，包含 OpDesc 序列和 VarDesc 字典。
- **OpDesc**：一个算子描述，记录类型、输入输出名称、属性。
- **VarDesc**：一个变量描述，记录名称、dtype、shape、持久化标记等。

## 执行器生命周期概览

```
[Graph Construction]         [Build Phase (once)]              [Scheduling / Execution]
                             ┌─────────────────────┐
 Python API ──► ProgramDesc  │ 1. BuildVariableScope│           work queue
   nn.Linear / optimizer     │    VarDesc → Variable │           ┌───────────┐
   append_op → OpDesc        │ 2. BuildOpFuncList    │──────────►│ async      │
                             │    OpDesc → OpFuncNode│           │ dispatch   │
                             │    (kernel select,    │           │ (thread    │
                             │     transfer insert)  │           │  pool)     │
                             │ 3. Convert            │           └───────────┘
                             │    dependency DAG      │           dep_count=0
                             │    stream schedule     │           → push queue
                             │    var lifetime / GC   │           → execute
                             └─────────────────────┘           → decrement deps
```

## 什么场景看什么文件

| 调试场景 | 应关注的文件 / 模块 |
|---------|-------------------|
| Program 构建逻辑、append_op 流程 | `python/paddle/base/framework.py` (Block, Program) |
| 参数初始化、startup_program | `python/paddle/base/framework.py` + `python/paddle/nn/initializer/` |
| Executor Python 入口 | `python/paddle/base/executor.py` |
| StandaloneExecutor / InterpreterCore | `paddle/fluid/framework/new_executor/standalone_executor.cc` |
| Build 阶段（变量创建、算子构建、依赖分析） | `paddle/fluid/framework/new_executor/program_interpreter.cc` |
| 算子依赖 DAG 与流调度 | `paddle/fluid/framework/new_executor/stream_analyzer.cc` |
| Scope 与变量生命周期管理 | `paddle/fluid/framework/scope.cc`, `paddle/fluid/framework/variable.h` |
| OpDesc / VarDesc protobuf 定义 | `paddle/fluid/framework/framework.proto` |
| 反向算子自动生成 | `python/paddle/autograd/backward_utils.py`, `paddle/fluid/operators/` |

## 社区源码参考 (L3)

- Program / Block / Op / Var 描述：`paddle/fluid/framework/program_desc.cc`, `block_desc.cc`, `op_desc.cc`, `var_desc.cc`
- 新执行器入口：`paddle/fluid/framework/new_executor/standalone_executor.cc`
- InterpreterCore 实现：`paddle/fluid/framework/new_executor/program_interpreter.cc`
- StreamAnalyzer 流调度：`paddle/fluid/framework/new_executor/stream_analyzer.cc`

更多实现细节请参考 `references/` 目录下的专题文档。
