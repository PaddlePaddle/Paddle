# 算子 YAML 配置规则

## 目录
- [概述](#概述)
- [前向算子 ops.yaml 配置](#前向算子-opsyaml-配置)
- [反向算子 backward.yaml 配置](#反向算子-backwardyaml-配置)
- [特殊配置项](#特殊配置项)
- [data_transform 配置](#data_transform-配置)
- [trace 算子完整示例](#trace-算子完整示例)

## 概述

在 `paddle/phi/ops/yaml/ops.yaml` 和 `paddle/phi/ops/yaml/backward.yaml` 文件中对算子进行描述及定义。框架编译时根据 YAML 配置自动生成 C++ 端的相关代码接口及内部实现，将上层 Python API 与底层 Kernel 连接起来。

## 前向算子 ops.yaml 配置

### 基本配置项

| 配置项 | 说明 |
|---|---|
| `op` | 算子名称，与 Python API 函数名相同，全小写+下划线命名 |
| `args` | 输入参数，与 Python API 输入参数对应。Tensor 类型参数为 Input，非 Tensor 类型为 Attribute |
| `output` | 输出类型，支持 `Tensor` 和 `Tensor[]`，多输出逗号分隔，用 `()` 标记名字（默认 `out`） |
| `infer_meta:func` | InferMeta 函数名 |
| `infer_meta:param` | InferMeta 输入参数（默认传入 args 全部参数） |
| `kernel:func` | Kernel 函数注册名 |
| `kernel:param` | Kernel 输入参数（默认传入 args 全部参数） |
| `kernel:data_type` | 指定推导 kernel data_type 的参数（默认根据输入 Tensor 自动推导） |
| `kernel:backend` | 指定推导 kernel Backend 的参数（默认根据输入 Tensor 自动推导） |
| `backward` | 对应的反向算子名称 |

### args 支持的数据类型

`Tensor`, `Tensor[]`, `float`, `double`, `bool`, `int`, `int64_t`, `int[]`, `int64_t[]`, `str`, `Place`, `DataType`, `DataLayout`, `IntArray`, `Scalar`

- `Tensor[]`：Tensor 数组
- `IntArray`：int 类型数组，用于 shape/index/axes，可用 Tensor 或普通整型数组构造
- `Scalar`：标量，支持不同普通数据类型

### output 规则

- `Tensor[]` 需在 `{}` 内指定 size 表达式：`Tensor[]{input.size()}`
- 未标记名字默认为 `out`

### invoke 配置

复用已有算子或自定义 C++ API 时使用 `invoke`，此时不需要配置 `infer_meta` 和 `kernel`：
- 复用已有算子：被复用算子须为前向算子且返回值类型相同（参考 `zeros_like`）
- 自定义 C++ API：在 `paddle/phi/api/lib/api_custom_impl.h` 声明，`api_custom_impl.cc` 实现（参考 `embedding`）

### intermediate 配置

标记前向计算中用于反向计算的中间变量，不出现在 Python API 返回结果中（新增算子不建议使用）。

## 反向算子 backward.yaml 配置

| 配置项 | 说明 |
|---|---|
| `backward_op` | 反向算子名称，一般为前向名称 + `_grad`，二阶为 + `_double_grad` |
| `forward` | 对应前向算子的名称、参数、返回值，需与 ops.yaml 一致 |
| `args` | 反向输入参数（详见下方排列规则） |
| `output` | 反向输出，顺序须与前向输入 Tensor 一致 |
| `infer_meta` | 同前向 |
| `kernel` | 同前向 |
| `backward` | 对应的更高阶反向算子名称 |
| `no_need_buffer` | 标记不需要内存数据的前向变量，释放内存优化显存 |

### 反向 args 排列规则

**约束 1**：所有参数须在 forward 配置项的参数中（输入、输出及输出的反向梯度）找到对应（按变量名匹配）。

**约束 2**：参数排列顺序为：
1. 前向输入 Tensor
2. 前向输出 Tensor
3. 前向输出 Tensor 的反向梯度（命名为 `{output_name}_grad`）
4. 前向非 Tensor 类型属性变量

反向不需要使用的前向变量无须添加。

### 反向 output 规则

顺序须与前向输入 Tensor 一致。例如前向 `(Tensor x, Tensor y)` -> 反向输出必须为 `Tensor(x_grad), Tensor(y_grad)`。

## 特殊配置项

### optional

指定输入 Tensor 为可选输入。参考 `dropout` 中的 `seed_tensor`。

### inplace

对指定输入做原位处理并作为输出返回。格式：`(x -> out)`。

规则：
- 算子名有 `_` 后缀：只生成 inplace 接口
- 算子名无 `_` 后缀：同时生成 inplace 接口（自动加 `_` 后缀）和普通接口

参考：`relu` 算子。

### view

与 inplace 类似，返回结果与输入共享内存但不是同一变量。格式：`(x -> out)`。参考：`reshape` 算子。

### no_need_buffer

反向算子中标记只需要前向变量的 Shape/LoD 而不需要内存数据的变量。标记后该变量的内存/显存会在前向完成后释放。

注意：由于 Tensor 内存被释放后影响 dtype 接口使用，须在 kernel 的 `data_type` 配置中指定其他 Tensor 来推导 data_type。

## data_transform 配置

控制算子输入参数的自动转换行为（dtype、backend、layout）。

| 子配置 | 说明 |
|---|---|
| `skip_transform` | 跳过指定参数的所有数据转换（最高优先级） |
| `support_trans_dtype` | 开启指定参数的自动类型转换（非复数类型） |

## trace 算子完整示例

### ops.yaml

```yaml
- op: trace
  args: (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
  output: Tensor(out)
  infer_meta:
    func: TraceInferMeta
  kernel:
    func: trace
  backward: trace_grad
```

### backward.yaml

```yaml
- backward_op: trace_grad
  forward: trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out)
  args: (Tensor x, Tensor out_grad, int offset, int axis1, int axis2)
  output: Tensor(x_grad)
  infer_meta:
    func: UnchangedInferMeta
    param: [x]
  kernel:
    func: trace_grad
    data_type: x
  no_need_buffer: x
```
