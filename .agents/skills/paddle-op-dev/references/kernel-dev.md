# 算子 Kernel 开发

## 目录
- [目录结构](#目录结构)
- [Kernel 函数签名](#kernel-函数签名)
- [Kernel 注册](#kernel-注册)
- [头文件示例](#头文件示例)
- [CPU Kernel 示例](#cpu-kernel-示例)
- [GPU Kernel 示例](#gpu-kernel-示例)
- [反向 Kernel](#反向-kernel)
- [开发注意事项](#开发注意事项)

## 目录结构

Kernel 在 `paddle/phi/kernels` 目录下开发。

### 设备无关的 Kernel（如 reshape）

```
paddle/phi/kernels/
├── xxx_kernel.h
├── xxx_kernel.cc
├── xxx_grad_kernel.h
└── xxx_grad_kernel.cc
```

### 设备相关、CPU & GPU 分别实现的 Kernel（如 trace）

```
paddle/phi/kernels/
├── xxx_kernel.h               # 前向声明
├── cpu/xxx_kernel.cc           # CPU 前向实现
├── gpu/xxx_kernel.cu           # GPU 前向实现
├── xxx_grad_kernel.h           # 反向声明
├── cpu/xxx_grad_kernel.cc      # CPU 反向实现
└── gpu/xxx_grad_kernel.cu      # GPU 反向实现
```

## Kernel 函数签名

```cpp
template <typename T, typename Context>
void XxxKernel(const Context& dev_ctx,
               const DenseTensor& input,   // Tensor 输入
               int attr1,                   // 属性输入
               DenseTensor* out);           // 输出
```

参数映射：
- `Tensor` -> `const DenseTensor&`
- `Tensor[]` -> `const std::vector<const DenseTensor*>&`
- 属性类型直接使用对应 C++ 类型
- 输出 `Tensor` -> `DenseTensor*`
- 输出 `Tensor[]` -> `std::vector<DenseTensor*>`
- 第一个参数始终为 `const Context& dev_ctx`

## Kernel 注册

使用 `PD_REGISTER_KERNEL` 宏注册 Kernel：

```cpp
PD_REGISTER_KERNEL(
    kernel_name,    // kernel 注册名，与 YAML 中 kernel:func 一致
    backend,        // 后端：CPU, GPU 等
    layout,         // 布局：ALL_LAYOUT
    kernel_fn,      // kernel 函数
    data_types...   // 支持的数据类型
) {}
```

示例：

```cpp
// CPU 注册
PD_REGISTER_KERNEL(
    trace, CPU, ALL_LAYOUT, phi::TraceKernel, float, double, int, int64_t, phi::dtype::complex<float>, phi::dtype::complex<double>
) {}

// GPU 注册
PD_REGISTER_KERNEL(
    trace, GPU, ALL_LAYOUT, phi::TraceKernel, float, double, int, int64_t, phi::dtype::complex<float>, phi::dtype::complex<double>
) {}
```

## 头文件示例

文件：`paddle/phi/kernels/trace_kernel.h`

```cpp
#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out);

}  // namespace phi
```

## CPU Kernel 示例

文件：`paddle/phi/kernels/cpu/trace_kernel.cc`

```cpp
#include "paddle/phi/kernels/trace_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
// 其他需要的头文件...

namespace phi {

template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  // 分配输出内存
  dev_ctx.template Alloc<T>(out);

  // 实现计算逻辑...
}

}  // namespace phi

// 注册 CPU Kernel
PD_REGISTER_KERNEL(
    trace, CPU, ALL_LAYOUT, phi::TraceKernel,
    float, double, int, int64_t,
    phi::dtype::complex<float>, phi::dtype::complex<double>
) {}
```

## GPU Kernel 示例

文件：`paddle/phi/kernels/gpu/trace_kernel.cu`

```cpp
#include "paddle/phi/kernels/trace_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
// 其他需要的头文件...

namespace phi {

template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  // 分配输出内存
  dev_ctx.template Alloc<T>(out);

  // GPU 计算逻辑（CUDA kernel 调用等）...
}

}  // namespace phi

// 注册 GPU Kernel
PD_REGISTER_KERNEL(
    trace, GPU, ALL_LAYOUT, phi::TraceKernel,
    float, double, int, int64_t,
    phi::dtype::complex<float>, phi::dtype::complex<double>
) {}
```

## 反向 Kernel

### 头文件 `trace_grad_kernel.h`

```cpp
#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void TraceGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     int offset,
                     int axis1,
                     int axis2,
                     DenseTensor* x_grad);

}  // namespace phi
```

### 注册反向 Kernel

```cpp
PD_REGISTER_KERNEL(
    trace_grad, CPU, ALL_LAYOUT, phi::TraceGradKernel,
    float, double, int, int64_t,
    phi::dtype::complex<float>, phi::dtype::complex<double>
) {}
```

## 参考 PyTorch 实现

Paddle 与 PyTorch 的算子在数学逻辑上往往一致，实现 Kernel 时可参考 PyTorch 的对应实现来加速开发。

### PyTorch 算子源码位置

| 类型 | PyTorch 路径 |
|---|---|
| CPU Kernel | `aten/src/ATen/native/xxx.cpp` |
| GPU Kernel | `aten/src/ATen/native/cuda/xxx.cu` |
| 算子注册 | `aten/src/ATen/native/native_functions.yaml` |
| 数学工具 | `aten/src/ATen/native/Math.h` |

### 参考流程

1. 在 PyTorch 的 `native_functions.yaml` 中找到目标算子的声明
2. 根据 `dispatch` 字段定位 CPU/GPU 实现文件
3. 理解核心算法逻辑，尤其是边界处理和数值稳定性技巧
4. 用 Paddle 的 API 风格重写（`DenseTensor`、`dev_ctx.template Alloc<T>(out)` 等）

### 适配差异对照

| PyTorch | Paddle |
|---|---|
| `at::Tensor` | `phi::DenseTensor` |
| `tensor.data_ptr<T>()` | `tensor.data<T>()` |
| `tensor.numel()` | `tensor.numel()` |
| `tensor.dim()` | `tensor.dims().size()` |
| `tensor.size(i)` | `tensor.dims()[i]` |
| `tensor.stride(i)` | `tensor.strides()[i]` |
| `TORCH_CHECK(...)` | `PADDLE_ENFORCE_XX(...)` |
| `at::parallel_for` | `phi::funcs::ForRange` |
| CUDA: `AT_DISPATCH_FLOATING_TYPES` | 通过 `PD_REGISTER_KERNEL` 模板参数指定 |

### 注意事项

- 不要直接复制 PyTorch 代码，需理解算法后用 Paddle 规范重写
- 注意 license 合规，Paddle 使用 Apache 2.0，PyTorch 使用 BSD-style
- PyTorch 的某些优化（如 vectorized kernel）可能需要用 Paddle 自身的工具函数替代
- 反向 Kernel 的梯度计算公式可直接参考 PyTorch 的 `derivatives.yaml` 和对应 `_backward` 实现

## 开发注意事项

1. **内存分配**：Kernel 中使用 `dev_ctx.template Alloc<T>(out)` 分配输出内存
2. **命名规范**：
   - Kernel 函数：`XxxKernel`（大驼峰）
   - 反向 Kernel 函数：`XxxGradKernel`
   - 注册名与 YAML 中 `kernel:func` 一致（全小写+下划线）
3. **头文件 include**：
   - CPU：`paddle/phi/backends/cpu/cpu_context.h`
   - GPU：`paddle/phi/backends/gpu/gpu_context.h`
   - 注册宏：`paddle/phi/core/kernel_registry.h`
4. **常用数据类型**：`float`, `double`, `int`, `int64_t`, `phi::dtype::float16`, `phi::dtype::bfloat16`, `phi::dtype::complex<float>`, `phi::dtype::complex<double>`
5. **复用 functor**：如果 CPU 和 GPU 共享部分逻辑，可在 `paddle/phi/kernels/funcs/` 中实现 functor
