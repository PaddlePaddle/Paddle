---
name: paddle-eager-graph
description: "Use when navigating Paddle eager-mode (dynamic graph) source code, tracing forward/backward execution, debugging autograd issues, understanding PyLayer, or investigating complex-valued gradient computation. Covers Python API to C++ kernel call chain, backward graph topology sort, and inplace version tracking."
---

# Paddle 动态图（Eager Mode）导航索引

Paddle 动态图**边执行边建图**：前向执行时构建反向图，调用 `backward()` 时按拓扑序执行反向图。

## 前向调用链路

```
Python paddle.add(x, y)
  │
  ▼
① ops_api.cc          ─ Python-C 映射，GetTensorFromArgs 提取 Tensor
  ▼
② eager_op_function.cc ─ 参数解析 / Dist Tensor / 释放 GIL / backend 选择
  ▼
③ dygraph_functions.cc ─ AMP / Type Promotion / 创建 GradNode / 构建反向图
  ▼
④ api.cc              ─ KernelKey 构造 / Kernel 选择 / PrepareData / InferMeta
  ▼
⑤ PHI Kernel 执行
```

## 关键文件表

| 层级 | 代码路径 | 代码生成器 |
|------|---------|-----------|
| ① Python-C 映射 | `paddle/fluid/pybind/ops_api.cc` | `ops_api_gen.py` |
| ② 动态图 C++ 接口 | `paddle/fluid/pybind/eager_op_function.cc` | `python_c_gen.py` |
| ③ 自动微分函数 | `paddle/fluid/eager/api/generated/.../dygraph_functions.cc` | `eager_gen.py` |
| ④ PHI 算子库接口 | `paddle/phi/api/lib/api.cc` | `tensor_operants_gen.py` |

## 反向关键数据结构

| 数据结构 | 一句话描述 |
|---------|-----------|
| **AutogradMeta** | Tensor 持有的反向元信息：梯度、来源 GradNode、slot/rank 位置 |
| **GradNodeBase** | 反向节点基类，纯虚 `operator()` 执行反向计算 |
| **GradSlotMeta** | 描述 GradNode 某个 slot 的 meta 信息与出边 Edge |
| **Edge** | 指向后继 GradNode 的边，含 `in_slot_id_` 和 `in_rank_` |
| **TensorWrapper** | GradNode 中保存前向 Tensor 的包装器，含 inplace 版本快照 |
| **GradTensorHolder** | 反向执行期间临时存放与聚合梯度的二维 buffer |

## 调试场景速查

| 调试场景 | 阅读哪个参考文档 |
|---------|----------------|
| 前向调用链路 / Kernel 选择问题 | [forward-call-chain.md](references/forward-call-chain.md) |
| 反向梯度不正确 / 拓扑排序问题 | [backward-execution.md](references/backward-execution.md) |
| 数据结构成员 / 内存泄漏排查 | [autograd-data-structs.md](references/autograd-data-structs.md) |
| 自定义反向 PyLayer | [pylayer.md](references/pylayer.md) |
| 复数梯度 / Wirtinger 导数 | [complex-autograd.md](references/complex-autograd.md) |
| Inplace 操作 / 版本追踪 | [inplace.md](references/inplace.md) |

## 社区资料（L3 层）

- [PFCC 动态图源码阅读](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/Dygraph)
