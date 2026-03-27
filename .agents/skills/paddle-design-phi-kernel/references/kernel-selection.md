# Kernel 选择与分派机制

## 概述

Kernel 选择是从 `KernelFactory` 的两级 map 中，根据当前运行时的 Backend、DataLayout、DataType（统称 BLD）三要素找到最匹配的 `Kernel` 对象。Paddle 中存在三种执行模式，各自有不同的选择路径。

## 核心概念：BLD 三要素

```
KernelKey = { Backend, DataLayout, DataType }
```

- **Backend**：计算设备，如 CPU、GPU、GPUDNN（cuDNN）、XPU、OneDNN
- **DataLayout**：数据排布，如 NCHW、NHWC、ANY（不限）
- **DataType**：数据类型，如 FLOAT32、FLOAT16、BFLOAT16、INT64

Kernel 选择的本质是：根据输入 Tensor 的实际属性和算子声明的偏好，确定一个 `KernelKey`，再从 `KernelFactory` 中查找对应的 `Kernel`。

## 新动态图（Eager Mode）选择流程

入口：`api.cc` 中自动生成的 C++ API 函数（由 `api_gen.py` 生成）。

### 步骤 1：ParseKernelKeyByInputArgs

遍历所有输入 Tensor，取出它们的 Backend、Layout、DataType。对多个输入按优先级规则合并：
- Backend：GPU > CPU（有任意 GPU 输入则选 GPU）
- DataType：取第一个非 None 输入的 dtype
- Layout：取第一个非 ANY 的 layout

### 步骤 2：YAML kernel 声明覆盖

`ops.yaml` 中的 `kernel` 字段可指定 `data_type`，例如：

```yaml
- op: cast
  kernel:
    func: cast
    data_type: x   # 强制以输入 x 的 dtype 作为 kernel dtype
```

如果 YAML 指定了 `data_type`，则覆盖步骤 1 推断的 DataType。同理 `backend` 字段可覆盖 Backend。

### 步骤 3：GetHighestPriorityKernelKey

在确定初始 `KernelKey` 后，检查 `KernelFactory` 中实际注册了哪些 key，按以下 6 步 fallback 策略逐步降级：

```
1. GPUDNN + 指定 layout    → 精确匹配 cuDNN kernel
2. GPUDNN + ALL_LAYOUT      → cuDNN kernel，放宽 layout
3. kernel_backend + layout  → 目标 backend 精确匹配
4. kernel_backend + ALL_LAYOUT → 目标 backend，放宽 layout
5. CPU + layout             → fallback 到 CPU
6. CPU + ALL_LAYOUT         → CPU 最宽松匹配
```

如果 6 步都失败，调用 `SelectKernelOrThrowError` 抛出异常，打印所有已注册的 kernel key，帮助调试。

### 步骤 4：PrepareData（数据转换）

选定 kernel 后，需要将输入 Tensor 的实际属性对齐到 `KernelKey` 和 `Kernel::args_def` 要求的属性。通过 `TransformFlag` 控制是否执行以下转换：

| 转换 | 函数 | 场景 |
|------|------|------|
| 跨设备搬运 | `TransDataPlace()` | 输入在 CPU 但 kernel 在 GPU |
| 数据类型转换 | `TransDataType()` | 输入 FP16 但 kernel 要求 FP32 |
| Layout 转换 | `TransDataLayout()` | 输入 NCHW 但 kernel 要求 NHWC |

如果 `Kernel::args_def` 中某个输入标记了 `Backend::ALL_BACKEND`（通过注册 hook 设置），则跳过 `TransDataPlace`，允许输入留在任意设备。

## 老动态图（Legacy Eager）选择流程

入口：`PreparedOp::Prepare()`，位于 `paddle/fluid/imperative/prepared_operator.cc`。

### 步骤 1：GetExpectedKernelType

每个 `OperatorWithKernel` 子类可重写 `GetExpectedKernelType()`，返回期望的 `OpKernelType`（包含 place、data_type、data_layout）。默认实现取第一个输入的 dtype 和当前执行 place。

### 步骤 2：PHI 优先于 Fluid

```cpp
if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type)) {
  // 使用 PHI kernel
} else {
  // fallback 到 fluid OpKernel
}
```

`HasCompatiblePhiKernel()` 检查算子名称是否在 PHI KernelFactory 中注册。如果同时存在 PHI kernel 和 fluid OpKernel，优先使用 PHI。

### 步骤 3：kernel_type_for_var

对于特殊算子（如 `fill_constant`），其 `GetKernelTypeForVar()` 可为不同输入变量返回不同的期望 kernel type，用于精细控制 PrepareData 行为。

## 静态图（Static Graph）选择流程

入口：`OperatorWithKernel::RunImpl()`，位于 `paddle/fluid/framework/operator.cc`。

### 核心流程

```
RunImpl()
  ├── InferShape()                    // 形状推导
  ├── GetExpectedKernelType()         // 获取期望 BLD
  ├── HasCompatiblePhiKernel()        // 检查 PHI kernel 是否存在
  │     ├── Yes → ChoosePhiKernel()   // 构造 KernelKey，查找 PHI kernel
  │     └── No  → ChooseFluidKernel() // fallback fluid
  ├── HandleComplexGradToRealGrad()   // 复数特殊处理
  ├── PrepareData()                   // 数据转换
  └── Run phi kernel / fluid kernel
```

### ChoosePhiKernel

将 fluid 的 `OpKernelType` 转换为 PHI 的 `KernelKey`：
- `OpKernelType.place_` → `Backend`
- `OpKernelType.data_layout_` → `DataLayout`
- `OpKernelType.data_type_` → `DataType`

然后调用 `KernelFactory::Instance().SelectKernel(name, key)`。

## YAML 字段与老框架的映射关系

| 老框架（fluid） | YAML 对应字段 | 说明 |
|------------------|---------------|------|
| `OpMaker` 类 | `ops.yaml` 整体 | 算子定义（输入、输出、属性） |
| `InferShape()` | `infer_meta.func` | 形状推导函数 |
| `GetExpectedKernelType()` | `kernel.data_type` / `kernel.backend` | kernel 选择偏好 |
| `GradOpMaker` | `backward.yaml` | 反向算子定义 |
| `OpKernel::Compute()` | `kernel.func` | kernel 函数名 |

## 调试技巧

### 打印 kernel 选择过程

```bash
GLOG_vmodule=phi_kernel_adaptor=4 python test.py
```

### 查看所有已注册 kernel

```python
import paddle
paddle.framework._phi_kernel_factory().get_all_kernel_names()
```

### 常见问题

1. **"Cannot find kernel for xxx"**：检查 kernel 是否对目标 Backend + DataType 注册
2. **数据类型不匹配**：检查 `ops.yaml` 中 `kernel.data_type` 的配置
3. **多余的 TransDataPlace**：检查注册 hook 是否设置了 `ALL_BACKEND`
4. **Layout 转换开销大**：检查是否有 NHWC 专用 kernel，避免 NCHW ↔ NHWC 转换

## 关键源码路径

| 文件 | 说明 |
|------|------|
| `paddle/phi/api/lib/api_gen_utils.cc` | ParseKernelKeyByInputArgs 等工具 |
| `paddle/phi/core/kernel_factory.cc` | SelectKernelOrThrowError 实现 |
| `paddle/fluid/imperative/prepared_operator.cc` | PreparedOp::Prepare（老动态图） |
| `paddle/fluid/framework/operator.cc` | OperatorWithKernel::RunImpl（静态图） |
| `paddle/phi/core/kernel_utils.h` | TransformFlag、PrepareData 相关 |
| `paddle/phi/ops/yaml/ops.yaml` | 算子 YAML 定义 |
