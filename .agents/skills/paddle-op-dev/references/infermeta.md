# InferMeta 函数开发

## 目录
- [概述](#概述)
- [开发规则](#开发规则)
- [函数签名约定](#函数签名约定)
- [常用 InferMeta 函数](#常用-infermeta-函数)
- [TraceInferMeta 完整示例](#traceinfermeta-完整示例)

## 概述

InferMeta 函数在 Kernel 前执行，根据输入参数推断输出 Tensor 的 `shape` 和 `data type`，同时检查输入数据维度、类型等合法性。每个算子只需实现一个 InferMeta 函数（不区分硬件设备）。

## 开发规则

1. **命名**：`XxxInferMeta`
2. **文件位置**：`paddle/phi/infermeta/` 目录，按输入 Tensor 个数分类：
   - `nullary.h/cc`：无输入 Tensor
   - `unary.h/cc`：单输入 Tensor
   - `binary.h/cc`：双输入 Tensor
   - `ternary.h/cc`：三输入 Tensor
   - `multiary.h/cc`：更多输入 Tensor
3. **头文件声明**：在对应的 `.h` 文件中声明
4. **输入合法性检查**：使用 `PADDLE_ENFORCE_XX` 宏
5. **输出设置**：调用 `out->set_dims()` 和 `out->set_dtype()`

## 函数签名约定

```cpp
void XxxInferMeta(
    const MetaTensor& input_tensor,   // Tensor 输入用 const MetaTensor&
    int attr1,                         // 属性用对应 C++ 类型
    float attr2,
    MetaTensor* out                    // 输出用 MetaTensor* 指针
);
```

参数列表与 YAML 配置中的 `args` 对应：
- `Tensor` -> `const MetaTensor&`
- `Tensor[]` -> `const std::vector<const MetaTensor*>&`
- `int`, `float`, `bool` 等 -> 直接使用对应 C++ 类型
- `int[]` -> `const std::vector<int>&`
- `IntArray` -> `const IntArray&`
- `Scalar` -> `const Scalar&`
- 输出 `Tensor` -> `MetaTensor*`
- 输出 `Tensor[]` -> `std::vector<MetaTensor*>`

## 常用 InferMeta 函数

以下 InferMeta 函数可直接复用，无需重新实现：

| 函数名 | 说明 |
|---|---|
| `UnchangedInferMeta` | 输出与输入有相同 shape 和 dtype |
| `UnchangedInferMetaCheckAxis` | 类似上面，额外检查 axis |
| `GeneralUnaryGradInferMeta` | 通用一元反向算子 |
| `GeneralBinaryGradInferMeta` | 通用二元反向算子 |

## PADDLE_ENFORCE 宏

```cpp
// 常用检查宏
PADDLE_ENFORCE_EQ(a, b, common::errors::InvalidArgument("..."));
PADDLE_ENFORCE_NE(a, b, common::errors::InvalidArgument("..."));
PADDLE_ENFORCE_GT(a, b, common::errors::InvalidArgument("..."));
PADDLE_ENFORCE_GE(a, b, common::errors::InvalidArgument("..."));
PADDLE_ENFORCE_LT(a, b, common::errors::OutOfRange("..."));
PADDLE_ENFORCE_LE(a, b, common::errors::OutOfRange("..."));
```

## TraceInferMeta 完整示例

文件：`paddle/phi/infermeta/unary.cc`

```cpp
void TraceInferMeta(
    const MetaTensor& x,
    int offset,
    int axis1,
    int axis2,
    MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  // 检查输入维度 >= 2
  PADDLE_ENFORCE_GE(
      x_dims.size(), 2,
      common::errors::OutOfRange(
          "Input(x)'s dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));

  // 检查 axis1 范围
  PADDLE_ENFORCE_LT(
      dim1_, x_dims.size(),
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, %ld], but got %ld).",
          -(x_dims.size()), (x_dims.size() - 1), dim1));
  PADDLE_ENFORCE_GE(
      dim1_, 0,
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, %ld], but got %ld).",
          -(x_dims.size()), (x_dims.size() - 1), dim1));

  // 检查 axis2 范围
  PADDLE_ENFORCE_LT(
      dim2_, x_dims.size(),
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, %ld], but got %ld).",
          -(x_dims.size()), (x_dims.size() - 1), dim2));
  PADDLE_ENFORCE_GE(
      dim2_, 0,
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, %ld], but got %ld).",
          -(x_dims.size()), (x_dims.size() - 1), dim2));

  // axis1 != axis2
  PADDLE_ENFORCE_NE(
      dim1_, dim2_,
      common::errors::InvalidArgument(
          "The dimensions should not be identical %d vs %d.", dim1, dim2));

  // 推导输出维度
  auto sizes = common::vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  out->set_dims(common::make_ddim(sizes));
  out->set_dtype(x.dtype());
}
```
