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

### 接口定义

前向分解通过 PIR Interface 机制实现。每个可分解的算子实现 `DecompInterface`：

```cpp
// paddle/fluid/pir/dialect/operator/interface/decomp.h
class DecompInterface : public pir::OpInterfaceBase<DecompInterface> {
  // concept-model 多态，由 Op 注册时自动绑定
};
```

### 分解规则实现

分解规则的实现位于 `paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h`：

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

`call_decomp_rule()` 位于 `paddle/fluid/primitive/base/decomp_trans.cc`，作为统一分发入口：

```cpp
std::vector<std::vector<pir::Value>> call_decomp_rule(pir::Operation* op) {
  paddle::dialect::DecompInterface decomp_interface =
      op->dyn_cast<paddle::dialect::DecompInterface>();
  // 通过 concept-model 多态调用对应 Op 的分解实现
  return decomp_interface.Decomp(op);
}
```

分解判断函数 `has_decomp_rule()` 检查 Op 是否注册了 `DecompInterface`。

## 反向分解：VjpInterface / DecompVjpInterface

### 概述

VJP（Vector-Jacobian Product）是反向传播的数学本质。组合算子体系为反向传播提供两层分解机制：

- **VjpInterface**：定义在 `paddle/fluid/pir/dialect/operator/interface/vjp.h`，提供反向计算规则
- **DecompVjpInterface**：定义在 `paddle/fluid/pir/dialect/operator/interface/decomp_vjp.h`，提供反向的组合算子分解

### 实现

VJP 规则分为自动生成和手写两部分：

- **自动生成**：`${PADDLE_BINARY_DIR}/paddle/fluid/primitive/vjp_interface/generated/generated_vjp.cc`（构建时由 `codegen/decomp_vjp_gen.py` 生成）
- **手写**：`paddle/fluid/primitive/vjp_interface/manual/manual_vjp.cc`

反向分解规则实现位于 `paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h`：

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

`call_decomp_vjp()` 同样位于 `paddle/fluid/primitive/base/decomp_trans.cc`，通过 `DecompVjpInterface` 分派。

统一入口头文件：`paddle/fluid/primitive/vjp_interface/vjp.h`。

## CustomVJP：数值稳定性特殊处理

某些算子的数学分解虽然正确，但在数值上不稳定。例如：

- **sigmoid 反向**：数学上 `grad = out_grad * sigmoid(x) * (1 - sigmoid(x))`，但直接用基础算子组合会丢失精度。CustomVJP 直接使用前向输出 `out`，计算 `grad = out_grad * out * (1 - out)`，避免重复计算 sigmoid。
- **log_softmax 反向**：类似地，利用前向已计算的中间结果提升数值稳定性。

CustomVJP 的注册方式与普通 VJP 相同，但实现中会利用前向输出作为中间量，而非重新从输入计算。

## 开发工作流

### 新增前向分解

1. 在 `paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h` 中实现分解模板函数
2. 确保对应 Op 注册了 `DecompInterface`（通过 YAML 配置 `composite` 字段或手写接口注册）
3. 编写单元测试验证精度

### 新增反向分解（VJP）

1. 在 `paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h` 中实现 VJP 模板函数
2. 如果是自动生成的算子，确保 YAML 中配置了 `composite` 字段
3. 手写 VJP 需要在 `paddle/fluid/primitive/vjp_interface/manual/manual_vjp.cc` 中添加
4. 编写测试验证梯度正确性

### 测试

```bash
# 单算子精度测试
python test/legacy_test/test_activation_op.py TestSigmoid

# 组合算子 VJP 专项测试
python test/prim/prim/vjp/eager/test_comp_eager_sigmoid_grad.py
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
3. **未注册的分解规则**：确认对应 Op 已注册 `DecompInterface`

## 关键文件路径汇总

| 文件 | 说明 |
|------|------|
| `paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h` | 前向分解规则实现 |
| `paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h` | VJP 反向分解实现 |
| `paddle/fluid/primitive/base/decomp_trans.cc` | `call_decomp_rule` / `call_decomp_vjp` 入口 |
| `paddle/fluid/pir/dialect/operator/interface/decomp.h` | DecompInterface 接口定义 |
| `paddle/fluid/pir/dialect/operator/interface/decomp_vjp.h` | DecompVjpInterface 接口定义 |
| `paddle/fluid/pir/dialect/operator/interface/vjp.h` | VjpInterface 接口定义 |
| `paddle/fluid/primitive/vjp_interface/vjp.h` | VJP 统一入口头文件 |
| `paddle/fluid/primitive/vjp_interface/manual/manual_vjp.cc` | 手写 VJP 实现 |
| `paddle/fluid/primitive/primitive/primitive.h` | 基础算子集声明 |
| `paddle/fluid/primitive/codegen/decomp_vjp_gen.py` | VJP 代码生成器 |
| `test/prim/` | 组合算子测试目录 |
