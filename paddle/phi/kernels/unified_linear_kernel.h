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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// 简化版本，适配非CUDA环境
namespace phi {

// 前向声明，避免依赖具体实现
class DenseTensor;
class Scalar;
enum class DataType : int32_t {
  UNDEFINED = 0,
  BOOL,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  BFLOAT16,
  COMPLEX64,
  COMPLEX128,
  // ... 其他类型
};

// Unified linear operation descriptor
struct UnifiedLinearDescriptor {
  // Input tensors
  const DenseTensor* input = nullptr;
  const DenseTensor* weight = nullptr;
  const DenseTensor* bias = nullptr;

  // Scale tensors for narrow precision types (fp8, etc.)
  const DenseTensor* input_scale = nullptr;
  const DenseTensor* weight_scale = nullptr;
  const DenseTensor* output_scale = nullptr;

  // Operation parameters
  bool transpose_input = false;
  bool transpose_weight = false;

  // Scaling factors
  Scalar alpha = 1.0f;
  Scalar beta = 0.0f;

  // Activation function
  std::string activation = "none";  // "none", "relu", "gelu", etc.

  // Data type configuration
  DataType compute_dtype = DataType::UNDEFINED;
  bool use_fast_accum = false;

  // Epilogue configuration
  bool use_bias = false;
  bool use_activation = false;

  // Constructor
  UnifiedLinearDescriptor() = default;

  // Validation
  bool IsValid() const;
  std::string GetErrorMessage() const;
};

// Hardware-agnostic unified linear kernel
// This is the top-level interface that handles:
// 1. Shape and stride validation
// 2. Framework tensor abstraction
// 3. Narrow precision type support (fp8, etc.)
// 4. Scale tensor handling
// 5. Dispatch to library-agnostic layer
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const UnifiedLinearDescriptor& desc,
                         DenseTensor* output);

// Simplified interface for common cases
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& bias,
                         bool transpose_input,
                         bool transpose_weight,
                         const std::string& activation,
                         DenseTensor* output);

// MatMul specialization (no bias, no activation)
template <typename T, typename Context>
void UnifiedMatMulKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool transpose_x,
                         bool transpose_y,
                         DenseTensor* out);

// Linear specialization (with bias and optional activation)
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& weight,
                         const paddle::optional<DenseTensor>& bias,
                         bool transpose_x,
                         bool transpose_weight,
                         const std::string& activation,
                         DenseTensor* out);

// Batch matrix multiplication
template <typename T, typename Context>
void UnifiedBatchMatMulKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              bool transpose_x,
                              bool transpose_y,
                              DenseTensor* out);

// Helper functions for descriptor creation
UnifiedLinearDescriptor CreateMatmulDescriptor(const DenseTensor& x,
                                               const DenseTensor& y,
                                               bool transpose_x,
                                               bool transpose_y);

UnifiedLinearDescriptor CreateLinearDescriptor(
    const DenseTensor& x,
    const DenseTensor& weight,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_weight,
    const std::string& activation);

// Shape inference helpers
std::vector<int64_t> InferUnifiedLinearShape(const DDim& x_dims,
                                             const DDim& weight_dims,
                                             bool transpose_x,
                                             bool transpose_weight);

// Utility functions for narrow precision support
bool IsNarrowPrecisionType(DataType dtype);
bool RequiresScaleTensor(DataType dtype);
DataType GetComputeType(DataType input_dtype, bool use_fast_accum);

}  // namespace phi
