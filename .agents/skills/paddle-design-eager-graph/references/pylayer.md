# PyLayer：Python 端自定义反向

本文档介绍 Paddle 的 `PyLayer` 机制，允许用户在 Python 端自定义前向和反向计算，同时无缝接入动态图的自动微分系统。

## 基本用法

```python
import paddle
from paddle.autograd import PyLayer

class SquaredReLU(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.nn.functional.relu(x)
        ctx.save_for_backward(y)
        return y * y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensor()
        return dy * 2 * y

x = paddle.randn([3, 4], dtype='float32')
x.stop_gradient = False
y = SquaredReLU.apply(x)
y.backward()
print(x.grad)
```

## PyLayer 规则

1. **`forward` 和 `backward` 必须是 `@staticmethod`**
2. **第一个参数是 `PyLayerContext`（即 `ctx`）**：提供 `save_for_backward()` 和 `saved_tensor()` 方法
3. **`backward` 的非 ctx 参数 = `forward` 的输出个数**：每个 `forward` 输出对应一个上游梯度
4. **`backward` 的返回值个数 = `forward` 的输入 Tensor 个数**：每个前向输入 Tensor 需要一个对应的下游梯度
5. **通过 `ctx.save_for_backward()` 传递 Tensor 给反向**：不要用闭包捕获 Tensor，否则可能导致内存泄漏
6. **通过 `ClassName.apply(args...)` 调用**：不要直接调用 `forward`
7. **`backward` 中不需要梯度的返回值可以返回 `None`**

## 多输入多输出示例

```python
class MultiIO(PyLayer):
    @staticmethod
    def forward(ctx, x, y, alpha):
        # alpha 是非 Tensor 参数，不需要返回梯度
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha
        out1 = alpha * x + y
        out2 = x * y
        return out1, out2

    @staticmethod
    def backward(ctx, d_out1, d_out2):
        x, y = ctx.saved_tensor()
        # 返回顺序对应 forward 的 (x, y, alpha)
        # alpha 是非 Tensor，返回 None
        dx = ctx.alpha * d_out1 + d_out2 * y
        dy = d_out1 + d_out2 * x
        return dx, dy, None
```

## 前向实现详解

**入口**：`SquaredReLU.apply(x)` 在 C++ 层最终调用 `PyLayerApply`。

### 步骤 1：解析参数，收集 AutogradMeta

```cpp
// 遍历所有输入参数，识别其中的 Tensor
// 获取每个 Tensor 的 AutogradMeta
auto autograd_metas = EagerUtils::nullable_autograd_meta(input_tensors);
bool require_any_grad = EagerUtils::ComputeRequireGrad(
    trace_backward, autograd_metas...);
```

与普通算子的 `ad_func` 类似，首先判断是否需要构建反向图。

### 步骤 2：关闭自动建图，执行 Python forward

```cpp
{
    AutoGradGuard guard(false);  // 临时关闭自动微分
    // 调用用户定义的 Python forward 函数
    py_outputs = PyObject_Call(forward_fn, py_args, nullptr);
}
// guard 析构时自动恢复
```

**为什么要关闭自动建图？** `forward` 内部可能调用 `paddle.relu()` 等算子，这些算子会自动构建反向图。但 PyLayer 有自己的 `backward`，不需要框架再为内部算子建图。关闭后，内部算子只做前向计算。

### 步骤 3：创建 GradNodePyLayer

```cpp
auto grad_node = std::make_shared<GradNodePyLayer>(
    py_layer_ctx, backward_fn, output_count, input_count);
```

`GradNodePyLayer` 是 `GradNodeBase` 的子类，持有：
- `py_layer_ctx_`：Python 的 `PyLayerContext` 对象（包含 `save_for_backward` 保存的 Tensor）
- `backward_fn_`：用户定义的 Python `backward` 函数

### 步骤 4：建立反向图连接

```cpp
// 连接到后继 GradNode（输入 Tensor 的 GradNode）
grad_node->SetGradOutMeta(input_tensors, slot_ids);

// 将 GradNodePyLayer 写入输出 Tensor 的 AutogradMeta
EagerUtils::SetHistory(output_autograd_metas, grad_node);

// 设置输入 meta
grad_node->SetGradInMeta(output_tensors, slot_ids);
```

这与普通算子的反向图构建完全一致——区别仅在于 `GradNode` 的 `operator()` 实现不同。

## 反向实现详解

当反向执行到 `GradNodePyLayer` 时，调用其 `operator()`：

### 步骤 1：将 C++ 梯度转为 Python 对象

```cpp
// grad_inputs: vector<vector<Tensor>> — 来自上游的梯度
// 逐个转换为 Python paddle.Tensor
PyObject* py_grad_inputs = ToPyObject(grad_inputs);
```

### 步骤 2：调用 Python backward

```cpp
// 调用用户定义的 backward 函数
PyObject* py_grad_outputs = PyObject_Call(
    backward_fn_, py_args_with_ctx, nullptr);
// py_args_with_ctx = (ctx, *py_grad_inputs)
```

此时用户的 `backward` 方法被执行，在 Python 端完成反向计算。

### 步骤 3：将 Python 结果转回 C++

```cpp
// 将 Python 返回的梯度 Tensor 转换回 C++ paddle::Tensor
auto grad_outputs = ToTensors(py_grad_outputs);
// 返回给反向图执行逻辑继续传播
return grad_outputs;
```

## 常见问题与调试

| 问题 | 原因与解决 |
|------|-----------|
| `backward` 返回值个数不匹配 | `backward` 返回的 Tensor 数量必须等于 `forward` 的输入 Tensor 数量（非 Tensor 参数用 None） |
| `forward` 内部算子的梯度丢失 | 正常行为——`forward` 内部自动建图被关闭，梯度由用户的 `backward` 负责 |
| `ctx.saved_tensor()` 返回空 | 检查是否在 `forward` 中调用了 `ctx.save_for_backward()` |
| 内存泄漏 | 避免在 `backward` 中引用外部大 Tensor；使用 `ctx.save_for_backward` 而非闭包 |
| `apply()` 未被调用 | 直接调用 `forward()` 不会构建反向图，必须通过 `apply()` |

## 与普通算子的对比

| 维度 | 普通算子 | PyLayer |
|------|---------|---------|
| 反向实现 | C++ `GradNode::operator()` | Python `backward()` |
| 反向图节点 | 自动生成的 `XxxGradNode` | `GradNodePyLayer` |
| 前向建图 | 自动 | 关闭（由用户 backward 接管） |
| 性能 | 高（纯 C++） | 较低（Python 回调开销） |
| 灵活性 | 需修改 C++ 代码 | 纯 Python 实现 |
