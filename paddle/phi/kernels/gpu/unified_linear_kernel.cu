/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// 简化版本，适配非CUDA环境
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace phi {

// 前向声明
class DenseTensor;
class GPUContext;
class Scalar;
enum class DataType : int32_t;

// 模拟kernel注册宏
#define PD_REGISTER_KERNEL(name, backend, layout, meta_kernel, dtype) \
  void __register_##name() {}

// Zero-cost hardware-agnostic unified linear kernel implementation
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& weight,
                         const paddle::optional<DenseTensor>& bias,
                         const paddle::optional<DenseTensor>& input_scale,
                         const paddle::optional<DenseTensor>& weight_scale,
                         const paddle::optional<DenseTensor>& output_scale,
                         bool transpose_x,
                         bool transpose_weight,
                         const std::string& activation,
                         float alpha,
                         float beta,
                         DataType out_dtype,
                         DenseTensor* out) {
  // Create zero-cost unified linear descriptor
  UnifiedLinearDescriptor desc;

  // Set input tensors
  desc.input = &x;
  desc.weight = &weight;
  desc.output = out;
  desc.bias = bias.get_ptr();
  desc.input_scale = input_scale.get_ptr();
  desc.weight_scale = weight_scale.get_ptr();
  desc.output_scale = output_scale.get_ptr();

  // Set operation parameters with zero-cost dispatch
  desc.transpose_input = transpose_x;
  desc.transpose_weight = transpose_weight;
  desc.activation = ParseActivationType(activation);
  desc.alpha = alpha;
  desc.beta = beta;
  desc.out_dtype = out_dtype;

  // Zero-cost descriptor validation
  ValidateUnifiedLinearDescriptor(desc);

  // Zero-cost output shape inference
  InferUnifiedLinearOutputShape(desc);

  // Zero-cost output tensor allocation
  out->Resize(desc.output->dims());
  dev_ctx.template Alloc<T>(out);

  // Execute with zero-cost dispatch to optimal implementation
  if (std::is_same<Context, GPUContext>::value) {
    ExecuteGpuUnifiedLinear<T>(dev_ctx, desc);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "UnifiedLinear only supports GPU execution"));
  }
}

// Simplified Unified Linear Kernel
template <typename T, typename Context>
void UnifiedLinearSimpleKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& weight,
                               const paddle::optional<DenseTensor>& bias,
                               bool transpose_x,
                               bool transpose_weight,
                               const std::string& activation,
                               DenseTensor* out) {
  UnifiedLinearKernel<T, Context>(dev_ctx,
                                  x,
                                  weight,
                                  bias,
                                  paddle::none,
                                  paddle::none,
                                  paddle::none,
                                  transpose_x,
                                  transpose_weight,
                                  activation,
                                  1.0f,
                                  0.0f,
                                  DataType::UNDEFINED,
                                  out);
}

// Matrix multiplication kernel (special case of unified linear)
template <typename T, typename Context>
void UnifiedMatmulKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool transpose_x,
                         bool transpose_y,
                         float alpha,
                         float beta,
                         DenseTensor* out) {
  // Create unified linear descriptor for matrix multiplication
  UnifiedLinearDescriptor desc;

  desc.input = &x;
  desc.weight = &y;
  desc.output = out;
  desc.bias = nullptr;
  desc.input_scale = nullptr;
  desc.weight_scale = nullptr;
  desc.output_scale = nullptr;

  desc.transpose_input = transpose_x;
  desc.transpose_weight = transpose_y;
  desc.activation = ActivationType::NONE;
  desc.alpha = alpha;
  desc.beta = beta;
  desc.out_dtype = DataType::UNDEFINED;

  // Validate and infer shape
  ValidateUnifiedLinearDescriptor(desc);
  InferUnifiedLinearOutputShape(desc);

  // Allocate output
  out->Resize(desc.output->dims());
  dev_ctx.template Alloc<T>(out);

  // Execute on GPU
  if (std::is_same<Context, GPUContext>::value) {
    ExecuteGpuUnifiedLinear<T>(dev_ctx, desc);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "UnifiedMatmul only supports GPU execution"));
  }
}

// Vector-matrix multiplication kernel
template <typename T, typename Context>
void UnifiedMvKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& weight,
                     bool transpose_weight,
                     DenseTensor* out) {
  // Ensure input is a vector
  if (x.dims().size() != 1) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input must be a vector for mv operation"));
  }

  // Create unified linear descriptor
  UnifiedLinearDescriptor desc;

  desc.input = &x;
  desc.weight = &weight;
  desc.output = out;
  desc.bias = nullptr;
  desc.input_scale = nullptr;
  desc.weight_scale = nullptr;
  desc.output_scale = nullptr;

  desc.transpose_input = false;  // Vector is never transposed
  desc.transpose_weight = transpose_weight;
  desc.activation = ActivationType::NONE;
  desc.alpha = 1.0f;
  desc.beta = 0.0f;
  desc.out_dtype = DataType::UNDEFINED;

  // Validate and infer shape
  ValidateUnifiedLinearDescriptor(desc);
  InferUnifiedLinearOutputShape(desc);

  // Allocate output
  out->Resize(desc.output->dims());
  dev_ctx.template Alloc<T>(out);

  // Execute on GPU
  if (std::is_same<Context, GPUContext>::value) {
    ExecuteGpuUnifiedLinear<T>(dev_ctx, desc);
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("UnifiedMv only supports GPU execution"));
  }
}

// Batched matrix multiplication kernel
template <typename T, typename Context>
void UnifiedBmmKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* out) {
  // Ensure inputs are batched matrices
  if (x.dims().size() < 3 || y.dims().size() < 3) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Inputs must be batched matrices for bmm operation"));
  }

  // Create unified linear descriptor
  UnifiedLinearDescriptor desc;

  desc.input = &x;
  desc.weight = &y;
  desc.output = out;
  desc.bias = nullptr;
  desc.input_scale = nullptr;
  desc.weight_scale = nullptr;
  desc.output_scale = nullptr;

  desc.transpose_input = transpose_x;
  desc.transpose_weight = transpose_y;
  desc.activation = ActivationType::NONE;
  desc.alpha = 1.0f;
  desc.beta = 0.0f;
  desc.out_dtype = DataType::UNDEFINED;

  // Validate and infer shape
  ValidateUnifiedLinearDescriptor(desc);
  InferUnifiedLinearOutputShape(desc);

  // Allocate output
  out->Resize(desc.output->dims());
  dev_ctx.template Alloc<T>(out);

  // Execute on GPU
  if (std::is_same<Context, GPUContext>::value) {
    ExecuteGpuUnifiedLinear<T>(dev_ctx, desc);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "UnifiedBmm only supports GPU execution"));
  }
}

// GPU execution function
template <typename T>
void ExecuteGpuUnifiedLinear(const GPUContext& dev_ctx,
                             const UnifiedLinearDescriptor& desc) {
  // Create unified linear CUDA executor
  UnifiedLinearCuda executor(dev_ctx);

  // Execute the operation
  executor.Execute<T>(desc);

  // Check for errors
  if (executor.HasError()) {
    PADDLE_THROW(common::errors::External("UnifiedLinear execution failed: " +
                                          executor.GetLastError()));
  }
}

// Helper function to parse activation type
ActivationType ParseActivationType(const std::string& activation) {
  if (activation.empty() || activation == "none") {
    return ActivationType::NONE;
  } else if (activation == "relu") {
    return ActivationType::RELU;
  } else if (activation == "gelu") {
    return ActivationType::GELU;
  } else if (activation == "tanh") {
    return ActivationType::TANH;
  } else if (activation == "sigmoid") {
    return ActivationType::SIGMOID;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported activation type: " + activation));
  }
}

// Kernel registrations
PD_REGISTER_KERNEL(unified_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedLinearKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->InputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_KERNEL(unified_linear_simple,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedLinearSimpleKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->InputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_KERNEL(unified_matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedMatmulKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->InputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_KERNEL(unified_mv,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedMvKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->InputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_KERNEL(unified_bmm,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedBmmKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->InputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
}

}  // namespace phi
