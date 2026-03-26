# 自动微分数据结构详解

本文档完整列出 Paddle 动态图自动微分系统中所有核心数据结构的成员和作用，并描述它们之间的数据流关系。

## 数据流总览

```
phi::Tensor
  └─ autograd_meta_ (AutogradMeta)
       ├─ grad_           ← 存放该 Tensor 的梯度
       └─ grad_node_      → GradNodeBase (如 AddGradNode)
            ├─ bwd_in_meta_  [vector<GradSlotMeta>]  ← 输入 slot 元信息
            ├─ bwd_out_meta_ [vector<GradSlotMeta>]  ← 输出 slot 元信息
            │     └─ adj_edge_ (Edge)
            │          └─ grad_node_ → 后继 GradNodeBase
            ├─ TensorWrapper (保存前向 Tensor)
            └─ operator()    ← 执行反向计算

执行阶段：
  GradTensorHolder.buffer_[slot][rank] → 聚合梯度 → 传入 operator()
```

## phi::Tensor

**文件**：`paddle/phi/api/include/tensor.h`

Paddle 的统一 Tensor 接口，所有 Python 端 `paddle.Tensor` 底层对应一个 `phi::Tensor`。

| 成员 | 类型 | 说明 |
|------|------|------|
| `impl_` | `std::shared_ptr<phi::TensorBase>` | 底层存储实现（`DenseTensor`、`SparseCsrTensor` 等） |
| `autograd_meta_` | `std::unique_ptr<AbstractAutogradMeta>` | 自动微分元信息，动态图模式下为 `AutogradMeta` |
| `name_` | `std::string` | Tensor 名称，调试用 |

**关键方法**：
- `set_autograd_meta()` / `mutable_autograd_meta()` — 设置/获取自动微分信息
- `defined()` — 检查 `impl_` 是否有效
- `initialized()` — 检查底层存储是否已分配

## AbstractAutogradMeta

**文件**：`paddle/phi/api/include/tensor.h`

纯虚基类，仅定义接口，使 `phi::Tensor` 不依赖具体的自动微分实现。

```cpp
class AbstractAutogradMeta {
public:
    virtual ~AbstractAutogradMeta() = default;
};
```

## AutogradMeta

**文件**：`paddle/fluid/eager/autograd_meta.h`

继承自 `AbstractAutogradMeta`，包含 Tensor 参与自动微分所需的全部信息。

| 成员 | 类型 | 说明 |
|------|------|------|
| `grad_` | `paddle::Tensor` | 该 Tensor 的梯度，`backward()` 后通过 `.grad` 访问 |
| `grad_node_` | `std::shared_ptr<GradNodeBase>` | 产生该 Tensor 的反向节点 |
| `out_slot_id_` | `size_t` | 该 Tensor 在 `grad_node_` 输出中的 slot 索引 |
| `out_rank_` | `size_t` | 该 Tensor 在 slot 内的 rank 索引 |
| `stop_gradient_` | `bool` | 为 true 时不参与梯度计算（对应 Python `Tensor.stop_gradient`） |
| `persistable_` | `bool` | 是否为持久化 Tensor（模型参数等） |
| `retain_grads_` | `bool` | 为 true 时即使是非叶节点也保留梯度 |

**slot 和 rank 的含义**：一个算子可能有多个输出（slot），每个输出可能是 Tensor 列表（rank 区分列表内位置）。`out_slot_id_` 和 `out_rank_` 记录该 Tensor 是 GradNode 第几个输出 slot 的第几个 Tensor。

## GradNodeBase

**文件**：`paddle/fluid/eager/grad_node_info.h`

反向图中的节点基类，每个前向算子对应一个具体的 `GradNode` 子类（如 `AddGradNode`、`MatmulGradNode`）。

| 成员 | 类型 | 说明 |
|------|------|------|
| `bwd_in_meta_` | `std::vector<std::vector<GradSlotMeta>>` | 反向输入的 meta 信息（对应前向输出） |
| `bwd_out_meta_` | `std::vector<std::vector<GradSlotMeta>>` | 反向输出的 meta 信息（对应前向输入），含 adj_edge_ |
| `gradient_hooks_` | `std::map<int, GradientHookFunc>` | 注册的梯度 hook 函数 |
| `default_attr_map_` | `paddle::framework::AttributeMap` | 前向算子的属性（如 axis、keepdim 等） |

**关键虚方法**：

| 方法 | 说明 |
|------|------|
| `operator()(vector<vector<Tensor>>&)` | **纯虚**。执行反向计算，输入为上游梯度，输出为下游梯度 |
| `ClearTensorWrappers()` | 清理保存的前向 Tensor（`retain_graph=false` 时调用） |
| `Copy()` | 复制节点（`GeneralGrad` 中使用） |

**注意 bwd_in/out 的命名方向**：
- `bwd_in_meta_` 对应反向计算的**输入**（即前向算子的**输出**的梯度）
- `bwd_out_meta_` 对应反向计算的**输出**（即前向算子的**输入**的梯度），其中的 `adj_edge_` 连接后继节点

## GradSlotMeta

**文件**：`paddle/fluid/eager/grad_node_info.h`

描述 GradNode 某个 slot 的元信息，同时通过 `adj_edge_` 建立反向图的边。

| 成员 | 类型 | 说明 |
|------|------|------|
| `stop_gradient_` | `bool` | 该 slot 是否停止梯度 |
| `place_` | `phi::Place` | Tensor 所在设备 |
| `meta_` | `phi::DenseTensorMeta` | 前向 Tensor 的 shape/dtype/layout 等 meta |
| `adj_edge_` | `Edge` | 连接到后继 GradNode 的边 |

## Edge

**文件**：`paddle/fluid/eager/grad_node_info.h`

反向图中的有向边，从当前 GradNode 的某个输出 slot 指向后继 GradNode 的某个输入 slot。

| 成员 | 类型 | 说明 |
|------|------|------|
| `grad_node_` | `std::shared_ptr<GradNodeBase>` | 后继 GradNode |
| `in_slot_id_` | `size_t` | 在后继 GradNode 中对应的输入 slot 索引 |
| `in_rank_` | `size_t` | 在该 slot 内的 rank 索引 |

**Edge 的方向理解**：反向图中，梯度从 loss 流向叶节点。Edge 描述的是梯度的流向——从当前节点的输出连接到下一个节点的输入。

## TensorWrapper

**文件**：`paddle/fluid/eager/tensor_wrapper.h`

GradNode 中用于保存前向 Tensor 的包装器。反向计算通常需要前向的输入或输出值（如 `y = relu(x)` 的反向需要 `x`）。

| 成员 | 类型 | 说明 |
|------|------|------|
| `intermediate_tensor_` | `paddle::Tensor` | 保存的前向 Tensor（拼写沿用代码中的 intermediate） |
| `no_need_buffer_` | `bool` | 为 true 时只保存 meta 信息，不保存数据（节省内存） |
| `weak_grad_node_` | `std::weak_ptr<GradNodeBase>` | 弱引用前向 Tensor 的 GradNode，**防止循环引用** |
| `inplace_version_snapshot_` | `uint32_t` | 保存时的 inplace 版本号，`recover()` 时校验是否被修改 |
| `packed_value_` | `std::shared_ptr<void>` | pack hook 打包后的值（用于内存优化，如 offload 到 CPU） |
| `unpack_hook_` | `std::shared_ptr<UnpackHookBase>` | 与 `packed_value_` 配合的 unpack 回调 |

**关键方法**：
- `recover()` — 从 `intermediate_tensor_` 恢复出 `paddle::Tensor`，同时检查 inplace 版本一致性
- 若 `packed_value_` 非空，通过 `unpack_hook_` 解包而非直接使用 `intermediate_tensor_`

**weak_ptr 防循环引用**：若 TensorWrapper 直接持有 `shared_ptr<GradNodeBase>`，会形成 `GradNode → TensorWrapper → Tensor → AutogradMeta → GradNode` 的循环引用，导致内存泄漏。使用 `weak_ptr` 打破循环。

## GradTensorHolder

**文件**：`paddle/fluid/eager/grad_tensor_holder.h`

反向执行阶段，每个待执行的 GradNode 都有一个对应的 `GradTensorHolder`，用于接收和聚合来自多条路径的上游梯度。

| 成员 | 类型 | 说明 |
|------|------|------|
| `buffer_` | `std::vector<std::vector<paddle::Tensor>>` | 二维 Tensor 容器，`buffer_[slot_id][rank]` |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `add(slot, rank, tensor)` | 梯度聚合：若 `buffer_[slot][rank]` 为空则赋值，否则执行 Tensor 加法累加 |
| `Buffers()` | 返回 `buffer_` 引用，传入 `GradNode::operator()` |
| `SetBufferSlotRankZeros(slot, rank)` | 将指定位置设为零 Tensor |

**聚合场景**：当一个 Tensor 被多个算子使用时（如 `y = x + x`），反向时两条路径的梯度需要累加到同一个 `GradTensorHolder` 中。

## 数据结构间的关系图

```
phi::Tensor ──owns──→ AutogradMeta
                         │
                    ┌────┘ grad_node_ (shared_ptr)
                    ▼
              GradNodeBase (如 AddGradNode)
                    │
         ┌─────────┼──────────┐
         ▼         ▼          ▼
   bwd_in_meta_  bwd_out_meta_  TensorWrapper
   (输入meta)   (输出meta)       │
                    │            └─ intermediate_tensor_
                    ▼            └─ weak_grad_node_ ──weak──→ GradNodeBase
              GradSlotMeta
                    │
                    └─ adj_edge_ (Edge)
                         │
                         └─ grad_node_ ──shared_ptr──→ 后继 GradNodeBase
```

构建阶段创建上述结构，执行阶段 `GradTensorHolder` 沿边传递和聚合梯度。
