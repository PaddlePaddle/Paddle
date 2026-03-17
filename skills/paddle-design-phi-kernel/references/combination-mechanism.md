# 组合算子（Operator Combination / Decomposition）机制

## 问题背景

Paddle 原生算子库包含约 1061 个算子。每当需要适配新场景（分布式自动并行、编译器优化、新硬件接入），都需要对这些算子逐一适配，成本极高：

- **分布式**：每个算子需要编写切分推导规则（SPMDRule）
- **编译器（CINN）**：每个算子需要编写 lowering 实现
- **新硬件**：每个算子需要编写 kernel

## 解决方案：基础算子集

定义约 200 个基础算子（primitive operators），将其余原生算子分解（decompose）为基础算子的组合。适配工作只需覆盖基础算子集即可。

基础算子集的选取原则：
- 语义原子性：不能再分解为更简单的操作
- 计算完备性：能组合表达所有原生算子
- 性能可接受：组合后的性能损失在可控范围内

## 前向分解：DecompInterface

### 注册

每个可分解的算子需要在 `paddle/fluid/primitive/decomp_rule/register.h` 中注册：

```cpp
// register.h
namespace paddle {
namespace primitive {
static std::unordered_map<std::string, DecompFunction> decomp_rule_map = {
    {"softmax", softmax_decomp},
    {"layer_norm", layer_norm_decomp},
    {"gelu", gelu_decomp},
    // ...
};
}  // namespace primitive
}  // namespace paddle
```

### 实现

分解规则的实现位于 `paddle/fluid/primitive/composite/composite.h`：

```cpp
template <typename T>
std::tuple<Tensor, Tensor, Tensor> layer_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    float epsilon,
    int begin_norm_axis) {
  // 使用基础算子表达：
  auto mean = paddle::mean(x, reduce_axes, true);
  auto diff = x - mean;
  auto variance = paddle::mean(diff * diff, reduce_axes, true);
  auto rsqrt_var = paddle::rsqrt(variance + epsilon);
  auto out = diff * rsqrt_var;
  if (scale) out = out * scale.get();
  if (bias) out = out + bias.get();
  return {out, mean, variance};
}
```

### 调用入口

`call_decomp_rule()` 位于 `paddle/fluid/primitive/decomp_rule/decomp_rule.cc`，作为统一分发入口：

```cpp
std::vector<Tensor> call_decomp_rule(const framework::OpDesc& op_desc) {
  auto it = decomp_rule_map.find(op_desc.Type());
  if (it != decomp_rule_map.end()) {
    return it->second(op_desc);
  }
  PADDLE_THROW("No decomp rule for op: " + op_desc.Type());
}
```

## 反向分解：VjpInterface

### 概述

VJP（Vector-Jacobian Product）是反向传播的数学本质。组合算子体系为反向传播提供独立的分解机制，使得反向也可用基础算子表达，无需编写原生反向 kernel。

### 注册

VJP 规则注册在 `paddle/fluid/primitive/rule/vjp/generated/generated_vjp.cc`（自动生成）和手写文件中：

```cpp
// 自动生成的 VJP 注册
REGISTER_VJP_INTERFACE(add, add_vjp);
REGISTER_VJP_INTERFACE(matmul, matmul_vjp);
```

### 实现

VJP 规则的实现位于 `paddle/fluid/primitive/rule/vjp/details.h`：

```cpp
template <typename T>
std::vector<std::vector<Tensor>> add_vjp(
    const Tensor& x,
    const Tensor& y,
    const Tensor& out_grad,
    int axis) {
  // add 的反向：grad_x = out_grad, grad_y = out_grad
  // 需要处理广播情况
  auto grad_x = reduce_as(out_grad, x);
  auto grad_y = reduce_as(out_grad, y);
  return {{grad_x}, {grad_y}};
}
```

### 调用入口

`call_vjp()` 位于 `paddle/fluid/primitive/rule/vjp/vjp_dispatch.cc`，在反向图构建时被调用。

## CustomVJP：数值稳定性特殊处理

某些算子的数学分解虽然正确，但在数值上不稳定。例如：

- **sigmoid 反向**：数学上 `grad = out_grad * sigmoid(x) * (1 - sigmoid(x))`，但直接用基础算子组合会丢失精度。CustomVJP 直接使用前向输出 `out`，计算 `grad = out_grad * out * (1 - out)`，避免重复计算 sigmoid。
- **log_softmax 反向**：类似地，利用前向已计算的中间结果提升数值稳定性。

CustomVJP 的注册方式与普通 VJP 相同，但实现中会利用前向输出作为中间量，而非重新从输入计算。

## 开发工作流

### 新增前向分解

1. 在 `paddle/fluid/primitive/composite/composite.h` 中实现分解模板函数
2. 在 `paddle/fluid/primitive/decomp_rule/register.h` 中注册算子名到分解函数的映射
3. 编写单元测试验证精度

### 新增反向分解（VJP）

1. 在 `paddle/fluid/primitive/rule/vjp/details.h` 中实现 VJP 模板函数
2. 如果是自动生成的算子，确保 YAML 中配置了 `composite` 字段
3. 手写 VJP 需要在注册文件中添加 `REGISTER_VJP_INTERFACE`
4. 编写测试验证梯度正确性

### 测试

```bash
# 单算子精度测试
python test/legacy_test/test_activation_op.py TestSigmoid

# 组合算子专项测试
python test/prim/prim/vjp/test_comp_vjp_sigmoid.py
```

## 动态 Shape 支持

组合算子在编译器场景下可能遇到动态 shape（编译期 shape 未知）。关键函数：

### has_dynamic_shape

```cpp
bool has_dynamic_shape(const std::vector<int64_t>& shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int64_t s) { return s < 0; });
}
```

检查 shape 中是否包含负数维度（-1 表示动态维度）。

### backend::reshape 的 Tensor 重载

当 shape 是动态的，不能用 `std::vector<int64_t>` 传递 shape，而是用 `Tensor` 类型：

```cpp
// 静态 shape
auto out = paddle::reshape(x, {batch_size, seq_len, hidden_size});

// 动态 shape
auto shape_tensor = paddle::shape(x);  // 返回 Tensor
auto out = paddle::backend::reshape(x, shape_tensor);
```

开发组合算子时，需要检查输入是否有动态 shape，并选择合适的 API 版本。

## 调试方法

### 前向分解调试

```bash
GLOG_vmodule=op_decomp=4 python test.py
```

输出信息包含：被分解的算子名、分解产生的基础算子序列、中间 Tensor shape。

### 反向分解（VJP）调试

```bash
GLOG_vmodule=generated_vjp=4 python test.py
```

输出信息包含：VJP 调用链、梯度 Tensor 的 shape 和 dtype。

### 常见问题

1. **分解后精度下降**：检查是否需要 CustomVJP，避免数值不稳定的组合
2. **动态 shape 报错**：检查分解实现中是否使用了 `has_dynamic_shape` 分支
3. **未注册的分解规则**：确认 register.h 中已添加映射

## 关键文件路径汇总

| 文件 | 说明 |
|------|------|
| `paddle/fluid/primitive/composite/composite.h` | 前向分解规则实现 |
| `paddle/fluid/primitive/decomp_rule/register.h` | 前向分解注册表 |
| `paddle/fluid/primitive/decomp_rule/decomp_rule.cc` | `call_decomp_rule` 入口 |
| `paddle/fluid/primitive/rule/vjp/details.h` | VJP 反向分解实现 |
| `paddle/fluid/primitive/rule/vjp/generated/generated_vjp.cc` | 自动生成的 VJP 注册 |
| `paddle/fluid/primitive/rule/vjp/vjp_dispatch.cc` | `call_vjp` 入口 |
| `paddle/fluid/primitive/primitive.h` | 基础算子集声明 |
| `test/prim/` | 组合算子测试目录 |
