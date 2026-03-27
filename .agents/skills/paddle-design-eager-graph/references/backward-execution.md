# 反向执行流程详解

本文档详述 Paddle 动态图中 `loss.backward()` 的完整反向执行流程，包括准备阶段、BFS 拓扑排序执行阶段，以及 `paddle.grad()` 使用的 `GeneralGrad` 机制。

## 入口调用链

```
Python: loss.backward()
  → C++: Backward()                    # paddle/fluid/eager/backward.cc
    → RunBackward(tensors, grad_tensors, retain_graph)
```

`Backward()` 是对 `RunBackward()` 的简单封装，负责处理默认梯度（若未指定 `grad_tensors`，则初始化为全 1 Tensor）。

## 准备阶段

### 1. 提取起始 GradNode

```cpp
// 从 loss tensor 中取出 AutogradMeta
auto* autograd_meta = EagerUtils::nullable_autograd_meta(loss);
// 获取起始 GradNode
auto start_node = autograd_meta->GradNode();
```

每个需要求梯度的 Tensor 的 `AutogradMeta` 中都持有一个 `grad_node_` 指针，指向产生该 Tensor 的反向节点。`loss` 的 `grad_node_` 就是反向图的起始节点。

### 2. 初始化 GradTensorHolder

```cpp
// 为起始节点创建 GradTensorHolder
node_input_buffers_dict[start_node.get()] =
    std::make_unique<GradTensorHolder>(start_node->InputMeta());

// 将初始梯度（默认 1.0）写入 buffer
node_input_buffers_dict[start_node.get()]->add(
    slot_id, rank, grad_tensor);
```

`GradTensorHolder` 的 `buffer_` 是一个二维向量 `vector<vector<Tensor>>`，第一维对应 slot，第二维对应该 slot 内的 rank。`add()` 方法实现梯度聚合——当多条路径汇聚到同一节点时，梯度会累加。

### 3. BFS 计算入度

```cpp
std::unordered_map<GradNodeBase*, int> node_in_degree_map;
std::deque<GradNodeBase*> queue;
queue.push_back(start_node.get());

while (!queue.empty()) {
    auto* node = queue.front();
    queue.pop_front();
    for (auto& [slot_id, meta] : node->OutputMeta()) {
        auto& edge = meta.GetEdge();
        auto* next_node = edge.GetGradNode();
        if (next_node) {
            node_in_degree_map[next_node]++;
            if (node_in_degree_map[next_node] == 1) {
                queue.push_back(next_node);
            }
        }
    }
}
```

通过 BFS 遍历整个反向图，统计每个节点的入度（有多少条边指向该节点）。这是后续拓扑排序的基础。

## 执行阶段（BFS 拓扑排序）

### 核心循环伪代码

```
queue = deque()
queue.push_back(start_node)    // 起始节点入度为 0

while queue 非空:
    node = queue.pop_front()

    // 1. 取出该节点的输入梯度 buffer
    node_input_buffer = node_input_buffers_dict[node]

    // 2. 执行反向计算
    //    调用 GradNode 的 operator()，传入上游梯度，返回下游梯度
    grad_output_tensors = node->operator()(node_input_buffer.Buffers())

    // 3. 释放已用完的 input buffer，节省内存
    delete node_input_buffers_dict[node]

    // 4. 将输出梯度传递给后继节点
    for each (slot_id, output_meta) in node->OutputMeta():
        edge = output_meta.adj_edge_
        next_node = edge.grad_node_

        if next_node is nullptr:
            continue

        // 为后继节点创建 GradTensorHolder（如果还没有的话）
        if next_node not in node_input_buffers_dict:
            node_input_buffers_dict[next_node] =
                new GradTensorHolder(next_node->InputMeta())

        // 梯度聚合：将当前输出梯度加到后继节点的 buffer 中
        node_input_buffers_dict[next_node].add(
            edge.in_slot_id_,
            edge.in_rank_,
            grad_output_tensors[slot_id])

        // 5. 入度减一
        node_in_degree_map[next_node] -= 1

        // 6. 入度为 0 时入队
        if node_in_degree_map[next_node] == 0:
            if next_node is AccumulationNode:
                queue.push_front(next_node)   // 叶节点优先
            else:
                queue.push_back(next_node)
```

### 关键细节

- **拓扑序保证**：只有入度减为 0 的节点才入队，确保节点在所有上游梯度到齐后才执行。
- **AccumulationNode 优先**：叶节点（参数 Tensor）对应的 `AccumulationNode` 使用 `push_front` 优先入队。这是因为叶节点执行后可以立即释放梯度 buffer，节省内存。
- **梯度聚合**：`GradTensorHolder::add()` 内部检查 buffer 是否已有 Tensor，若有则执行 `paddle::experimental::add` 累加；若无则直接赋值。
- **retain_graph**：若 `retain_graph=false`（默认），执行完反向后会清理 `TensorWrapper`，释放前向 Tensor 引用，防止内存泄漏。调用 `node->ClearTensorWrappers()`。
- **Hooks**：执行节点前后可触发 gradient hooks（`node->ApplyGradientHooks()`），用于梯度裁剪、监控等。

## AccumulationNode（叶节点）

叶节点（如模型参数 `w`）不是由某个算子产生的，其 `GradNode` 是特殊的 `AccumulationNode`。

```cpp
class AccumulationNode : public GradNodeBase {
    // operator() 实现：将传入的梯度累加到 tensor.grad 中
    paddle::Tensor* tensor_;  // 指向原始叶 Tensor
};
```

`AccumulationNode::operator()` 将梯度写入叶 Tensor 的 `AutogradMeta::grad_` 中，这就是训练时 `w.grad` 的来源。

## GeneralGrad：paddle.grad() 的实现

`paddle.grad(outputs, inputs, grad_outputs)` 不同于 `backward()`——它只计算指定 `inputs` 的梯度，不累加到 `.grad` 属性中。

### 核心类：GeneralGrad

```cpp
class GeneralGrad {
    // 单例模式
    static GeneralGrad& Instance();

    void PreparedForGeneralGrad(...);
    void GetGraphInfoBetweenTargets(...);
    std::vector<paddle::Tensor> GetResults(...);
};
```

### 执行流程

#### 1. CopyGradNode

对起始节点创建一个 `CopyGradNode` 副本，避免修改原始反向图。

#### 2. PreparedForGeneralGrad

标记目标输入 Tensor 对应的 `GradNode`，以便在执行过程中识别。

#### 3. GetGraphInfoBetweenTargets

从 outputs 对应的 GradNode 开始 BFS，找到 outputs 到 inputs 之间的所有节点，裁剪不需要的分支：
- 只保留从 outputs 可达且最终能到达 inputs 的节点
- 对不在路径上的节点，将其入度设为 0（跳过执行）

#### 4. RegisterFetchGradHook

在目标 inputs 对应的 `AccumulationNode` 上注册 hook，在梯度到达时捕获梯度值，而不是写入 `.grad` 属性。

#### 5. 执行

复用与 `backward()` 相同的 BFS 拓扑排序循环，但只遍历裁剪后的子图。

#### 6. GetResults

从注册的 hook 中收集捕获的梯度，按 `inputs` 的顺序返回。

## 调试提示

- **梯度为 None**：检查对应 Tensor 的 `stop_gradient` 是否为 True，或该 Tensor 是否在反向图路径上。
- **梯度数值错误**：在 `GradNode::operator()` 中打断点，检查输入梯度和输出梯度。
- **内存泄漏**：检查 `retain_graph` 是否被误设为 True，或 `TensorWrapper` 是否正确释放。
- **拓扑序异常**：打印 `node_in_degree_map`，检查入度计算是否正确。
