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
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/unified_linear_kernel.h"

namespace phi {

// Implementation of descriptor validation
inline bool UnifiedLinearDescriptor::IsValid() const {
  if (!input || !weight || !output) {
    return false;
  }

  // Validate dimensions
  const auto& input_dims = input->dims();
  const auto& weight_dims = weight->dims();

  if (input_dims.size() < 1 || weight_dims.size() < 2) {
    return false;
  }

  // Check matrix multiplication compatibility
  int64_t input_last_dim = transpose_input ? input_dims[input_dims.size() - 2]
                                           : input_dims[input_dims.size() - 1];
  int64_t weight_first_dim = transpose_weight
                                 ? weight_dims[weight_dims.size() - 1]
                                 : weight_dims[weight_dims.size() - 2];

  if (input_last_dim != weight_first_dim) {
    return false;
  }

  // Validate scale tensors for narrow precision types
  if (IsNarrowPrecisionType(input->dtype()) && !input_scale) {
    return false;
  }

  if (IsNarrowPrecisionType(weight->dtype()) && !weight_scale) {
    return false;
  }

  return true;
}

inline std::string UnifiedLinearDescriptor::GetErrorMessage() const {
  if (!input) return "Input tensor is null";
  if (!weight) return "Weight tensor is null";
  if (!output) return "Output tensor is null";

  const auto& input_dims = input->dims();
  const auto& weight_dims = weight->dims();

  if (input_dims.size() < 1)
    return "Input tensor must have at least 1 dimension";
  if (weight_dims.size() < 2)
    return "Weight tensor must have at least 2 dimensions";

  int64_t input_last_dim = transpose_input ? input_dims[input_dims.size() - 2]
                                           : input_dims[input_dims.size() - 1];
  int64_t weight_first_dim = transpose_weight
                                 ? weight_dims[weight_dims.size() - 1]
                                 : weight_dims[weight_dims.size() - 2];

  if (input_last_dim != weight_first_dim) {
    return "Input last dimension (" + std::to_string(input_last_dim) +
           ") must match weight first dimension (" +
           std::to_string(weight_first_dim) + ")";
  }

  if (IsNarrowPrecisionType(input->dtype()) && !input_scale) {
    return "Input scale tensor required for narrow precision type";
  }

  if (IsNarrowPrecisionType(weight->dtype()) && !weight_scale) {
    return "Weight scale tensor required for narrow precision type";
  }

  return "Unknown error";
}

// Utility function implementations
inline bool IsNarrowPrecisionType(DataType dtype) {
  return dtype == DataType::FLOAT8_E4M3FN || dtype == DataType::FLOAT8_E5M2 ||
         dtype == DataType::INT4 || dtype == DataType::INT8;
}

inline bool RequiresScaleTensor(DataType dtype) {
  return IsNarrowPrecisionType(dtype);
}

inline DataType GetComputeType(DataType input_dtype, bool use_fast_accum) {
  switch (input_dtype) {
    case DataType::FLOAT8_E4M3FN:
    case DataType::FLOAT8_E5M2:
      return use_fast_accum ? DataType::FLOAT16 : DataType::FLOAT32;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      return use_fast_accum ? DataType::FLOAT16 : DataType::FLOAT32;
    case DataType::FLOAT32:
      return DataType::FLOAT32;
    case DataType::FLOAT64:
      return DataType::FLOAT64;
    case DataType::INT8:
      return DataType::INT32;
    default:
      return DataType::FLOAT32;
  }
}

// Shape inference implementation
inline std::vector<int64_t> InferUnifiedLinearShape(const DDim& x_dims,
                                                    const DDim& weight_dims,
                                                    bool transpose_x,
                                                    bool transpose_weight) {
  std::vector<int64_t> out_dims;

  // Handle batch dimensions
  int x_ndim = x_dims.size();
  int weight_ndim = weight_dims.size();

  // For matmul/linear, we expect at least 2D weight
  PADDLE_ENFORCE_GE(weight_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "Weight tensor must have at least 2 dimensions, got %d",
                        weight_ndim));

  // Get the matrix multiplication dimensions
  int64_t x_last_dim = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  int64_t weight_first_dim = transpose_weight ? weight_dims[weight_ndim - 1]
                                              : weight_dims[weight_ndim - 2];

  PADDLE_ENFORCE_EQ(
      x_last_dim,
      weight_first_dim,
      common::errors::InvalidArgument("Input last dimension (%lld) must match "
                                      "weight first dimension (%lld)",
                                      x_last_dim,
                                      weight_first_dim));

  // Build output shape
  // Batch dimensions from input (excluding the last dimension)
  for (int i = 0; i < x_ndim - 1; ++i) {
    out_dims.push_back(x_dims[i]);
  }

  // Last dimension from weight
  int64_t weight_last_dim = transpose_weight ? weight_dims[weight_ndim - 2]
                                             : weight_dims[weight_ndim - 1];
  out_dims.push_back(weight_last_dim);

  return out_dims;
}

// Descriptor creation helpers
inline UnifiedLinearDescriptor CreateMatmulDescriptor(const DenseTensor& x,
                                                      const DenseTensor& y,
                                                      bool transpose_x,
                                                      bool transpose_y) {
  UnifiedLinearDescriptor desc;
  desc.input = &x;
  desc.weight = &y;
  desc.transpose_input = transpose_x;
  desc.transpose_weight = transpose_y;
  desc.use_bias = false;
  desc.use_activation = false;
  desc.activation = "none";

  // Infer compute type
  desc.compute_dtype = GetComputeType(x.dtype(), desc.use_fast_accum);

  return desc;
}

inline UnifiedLinearDescriptor CreateLinearDescriptor(
    const DenseTensor& x,
    const DenseTensor& weight,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_weight,
    const std::string& activation) {
  UnifiedLinearDescriptor desc;
  desc.input = &x;
  desc.weight = &weight;
  desc.transpose_input = transpose_x;
  desc.transpose_weight = transpose_weight;
  desc.activation = activation;
  desc.use_bias = bias.is_initialized();
  desc.use_activation = (activation != "none");

  if (desc.use_bias) {
    desc.bias = bias.get_ptr();
  }

  // Infer compute type
  desc.compute_dtype = GetComputeType(x.dtype(), desc.use_fast_accum);

  return desc;
}

// Main unified linear kernel implementation
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const UnifiedLinearDescriptor& desc,
                         DenseTensor* output) {
  // Validate descriptor
  PADDLE_ENFORCE_EQ(
      desc.IsValid(),
      true,
      common::errors::InvalidArgument("Invalid UnifiedLinearDescriptor: %s",
                                      desc.GetErrorMessage().c_str()));

  // Infer output shape
  const auto& x_dims = desc.input->dims();
  const auto& weight_dims = desc.weight->dims();
  auto out_shape = InferUnifiedLinearShape(
      x_dims, weight_dims, desc.transpose_input, desc.transpose_weight);

  // Allocate output tensor
  output->Resize(common::make_ddim(out_shape));
  dev_ctx.template Alloc<T>(output);

  // Set output in descriptor for library-agnostic layer
  desc.output = output;

  // Dispatch to library-agnostic layer
  // This will be implemented in the CUDA-specific file
  // For now, we'll use a placeholder
  PADDLE_ENFORCE_NOT_NULL(
      desc.output,
      common::errors::InvalidArgument("Output tensor not allocated"));

  // TODO(Pan Zhaowu): Implement actual dispatch to library-agnostic layer
  // This will be done in the unified_linear_cuda.cu file
}

// Simplified interface implementations
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& bias,
                         bool transpose_input,
                         bool transpose_weight,
                         const std::string& activation,
                         DenseTensor* output) {
  auto desc = CreateLinearDescriptor(input,
                                     weight,
                                     paddle::optional<DenseTensor>(bias),
                                     transpose_input,
                                     transpose_weight,
                                     activation);
  UnifiedLinearKernel<T, Context>(dev_ctx, desc, output);
}

template <typename T, typename Context>
void UnifiedMatMulKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool transpose_x,
                         bool transpose_y,
                         DenseTensor* out) {
  auto desc = CreateMatmulDescriptor(x, y, transpose_x, transpose_y);
  UnifiedLinearKernel<T, Context>(dev_ctx, desc, out);
}

template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& weight,
                         const paddle::optional<DenseTensor>& bias,
                         bool transpose_x,
                         bool transpose_weight,
                         const std::string& activation,
                         DenseTensor* out) {
  auto desc = CreateLinearDescriptor(
      x, weight, bias, transpose_x, transpose_weight, activation);
  UnifiedLinearKernel<T, Context>(dev_ctx, desc, out);
}

template <typename T, typename Context>
void UnifiedBatchMatMulKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              bool transpose_x,
                              bool transpose_y,
                              DenseTensor* out) {
  auto desc = CreateMatmulDescriptor(x, y, transpose_x, transpose_y);
  UnifiedLinearKernel<T, Context>(dev_ctx, desc, out);
}

}  // namespace phi
