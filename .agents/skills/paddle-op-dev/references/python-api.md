# Python API 封装

## 目录
- [概述](#概述)
- [API 放置位置](#api-放置位置)
- [API 实现规范](#api-实现规范)
- [trace API 完整示例](#trace-api-完整示例)
- [注册到公开接口](#注册到公开接口)

## 概述

Python API 是用户直接使用的接口（如 `paddle.trace()`），底层调用 C++ 算子。API 需要完成：参数检查与处理、调用底层 C++ 算子、编写规范的 docstring。

## API 放置位置

- 文件位置：`python/paddle/` 目录下的相应子目录
- 遵循相似功能放在同一文件夹的原则
- 例如 `trace` 属于数学运算，放在 `python/paddle/tensor/math.py`

## API 实现规范

### 函数签名

```python
def xxx(input, param1, param2=default_value, name=None):
```

### 必要内容

1. **参数检查**：验证参数类型和值的合法性
2. **动态图/静态图兼容**：使用 `in_dynamic_or_pir_mode()` 判断
3. **调用底层算子**：
   - 动态图：`_C_ops.xxx(args...)`
   - 静态图：创建 `LayerHelper`，使用 `helper.create_variable_for_type_inference()` 和 `helper.append_op()`
4. **docstring**：包含功能描述、参数说明、返回值、代码示例

### 调用底层算子的方式

```python
from paddle import _C_ops

# 动态图模式直接调用
if in_dynamic_or_pir_mode():
    return _C_ops.op_name(args...)
```

```python
# 静态图模式
helper = LayerHelper('op_name', **locals())
out = helper.create_variable_for_type_inference(dtype=input.dtype)
helper.append_op(
    type='op_name',
    inputs={'X': input},
    outputs={'Out': out},
    attrs={'attr1': value1}
)
return out
```

## trace API 完整示例

文件：`python/paddle/tensor/math.py`

```python
def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """
    Computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.
    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be
            float32, float64, int32, int64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Name for the operation. Default: None.

    Returns:
        Tensor: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2, 3],
            ...                       [4, 5, 6],
            ...                       [7, 8, 9]])
            >>> out = paddle.trace(x)
            >>> print(out)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            15)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.trace(x, offset, axis1, axis2)
    else:
        inputs = {'Input': [x]}
        attrs = {'offset': offset, 'axis1': axis1, 'axis2': axis2}
        helper = LayerHelper('trace', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='trace',
            inputs={'Input': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out
```

## 注册到公开接口

API 实现后，需在 `python/paddle/__init__.py` 或对应模块的 `__init__.py` 中导出，使用户可以通过 `paddle.xxx()` 调用。

注意事项：
- `name` 参数在所有 API 中均为可选，用于调试
- docstring 中的 Examples 必须可运行（会被 CI 自动测试）
- 参数类型和默认值需与 YAML 配置一致
