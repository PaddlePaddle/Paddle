---
name: paddle-distributed
description: "Use when working with Paddle's distributed training, dynamic-to-static conversion (SOT/dy2st), or Python-C++ interoperability: understanding parallelism strategies, SPMD sharding rules, pipeline scheduling, PaddleSOT bytecode-level graph capture, or how C++ core is exposed to Python via pybind11 and Python/C API."
---

# Paddle 分布式训练、SOT 动转静与 Python-C++ 互操作

## 分布式范式速查

| 范式 | 核心思想 | 通信原语 |
|------|---------|---------|
| **Data Parallel** | 复制模型，切分数据，AllReduce 梯度 | AllReduce |
| **Group Sharded (ZeRO)** | Stage1 切 optimizer / Stage2 + 切 grad / Stage3 + 切 weight | Broadcast, ReduceScatter, AllGather |
| **Model Parallel (Tensor)** | Column Parallel 切权重列 / Row Parallel 切权重行 | AllReduce / AllGather |
| **Pipeline Parallel** | F-then-B / 1F1B 交错前反向 | Send / Recv (P2P) |
| **Sequence Parallel** | 沿 sequence 维度切分 LayerNorm/Dropout | AllGather / ReduceScatter |

三种编程范式：**手动** (`fleet.meta_parallel`)、**半自动动态图** (`ProcessMesh` + `shard_tensor`)、**半自动静态图** (`auto_parallel.Engine`)。

## SOT 架构速查

```
Python Frame
  │
  ▼  PEP 523 eval_frame 拦截
PyInterpreterState.eval_frame
  │
  ▼
OpcodeExecutor（模拟 Python VM 字节码执行）
  │  ├─ Variable 体系 (TensorVariable, ConstantVariable, ...)
  │  ├─ Tracker 追踪来源 → 生成 Guard
  │  └─ SideEffect 记录副作用
  ▼
FunctionGraph / StatementIR（记录算子调用）
  │
  ▼  fallback 场景: DDCF / UNSPS / CDBL / UNIMP
Sub-graph 切分 → StaticFunction 编译执行
```

## 什么场景看什么文件

| 场景 | 文件 / 参考 |
|------|------------|
| 分布式策略原理（DP/Sharded/MP/PP/SP） | `references/distributed-primer.md` |
| SPMD 推导规则与 Pipeline 调度 | `references/spmd-rules.md` |
| SOT 动转静（字节码级别图捕获） | `references/sot-design.md` |
| Python-C++ 桥接（pybind11 / Python/C API） | `references/python-cpp-bridge.md` |
| ProcessMesh / shard_tensor API | `python/paddle/distributed/auto_parallel/` |
| SPMD 推导规则 C++ 实现 | `paddle/phi/infermeta/spmd_rules/` |
| Pipeline 调度器 | `python/paddle/distributed/passes/pipeline_scheduler_pass/` |
| SOT 核心实现 | `python/paddle/jit/sot/opcode_translator/` |
| Python-C 绑定源码 | `paddle/fluid/pybind/eager*.cc` |
| 自动代码生成（Op 绑定） | `paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py` |

## 社区源码入口 (L3)

- 分布式总目录：`python/paddle/distributed/`
- PHI SPMD 规则：`paddle/phi/infermeta/spmd_rules/`
- SOT 总目录：`python/paddle/jit/sot/`
- pybind 目录：`paddle/fluid/pybind/`
- 自动生成脚本：`paddle/fluid/eager/auto_code_generator/`
