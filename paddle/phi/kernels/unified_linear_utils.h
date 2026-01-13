/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/unified_linear_kernel.h"

namespace phi {
namespace unified_linear {

// Namespace for utility functions and structures
namespace utils {

// Enum for scale data types
enum class ScaledDataType { kNone = 0, kStatic = 1, kTensor = 2 };

// Template for scaled tensor support
template <typename T>
struct ScaledTensor {
  const DenseTensor* tensor;
  const DenseTensor* scale;
  ScaledDataType scale_type;

  // Constructor
  ScaledTensor()
      : tensor(nullptr), scale(nullptr), scale_type(ScaledDataType::kNone) {}

  // Constructor with tensor and optional scale
  ScaledTensor(const DenseTensor* t,
               const paddle::optional<DenseTensor>* s = nullptr,
               ScaledDataType st = ScaledDataType::kNone)
      : tensor(t), scale(nullptr), scale_type(st) {
    if (s && s->is_initialized()) {
      scale = &(*s);
      scale_type = st;
    }
  }
};

// Determine if a tensor should be treated as scaled
template <typename T>
ScaledDataType DetermineScaleType(const DenseTensor& tensor,
                                  const paddle::optional<DenseTensor>& scale) {
  if (!scale.is_initialized()) {
    return ScaledDataType::kNone;
  }

  // Check if tensor is narrow precision (int8, fp8, etc.)
  if (tensor.dtype() == DataType::INT8 || tensor.dtype() == DataType::UINT8 ||
      tensor.dtype() == DataType::FP8E5M2 ||
      tensor.dtype() == DataType::FP8E4M3FN) {
    return ScaledDataType::kTensor;
  }

  return ScaledDataType::kNone;
}

// Validate scale tensor dimensions
void ValidateScaleTensor(const DenseTensor& tensor, const DenseTensor& scale) {
  const auto& scale_dims = scale.dims();

  // Scale tensor should be 1D with size 1 (scalar) or match the tensor's batch
  // dimension
  if (scale_dims.size() == 1 && scale_dims[0] == 1) {
    return;  // Scalar scale
  }

  // For batched operations, scale can match the batch dimension
  const auto& tensor_dims = tensor.dims();
  if (tensor_dims.size() > 2 && scale_dims.size() == 1 &&
      scale_dims[0] == tensor_dims[0]) {
    return;  // Batch scale
  }

  PADDLE_THROW(phi::errors::InvalidArgument(
      "Invalid scale tensor dimensions. Scale should be scalar or match batch "
      "dimension."));
}

// Check if dimensions are compatible for matrix multiplication
bool AreDimensionsCompatible(const DenseTensor& A,
                             const DenseTensor& B,
                             bool trans_A,
                             bool trans_B) {
  const auto& A_dims = A.dims();
  const auto& B_dims = B.dims();

  // Get effective dimensions after transpose
  int A_cols = trans_A ? A_dims[1] : A_dims[0];
  int B_rows = trans_B ? B_dims[1] : B_dims[0];

  return A_cols == B_rows;
}

// Prepare tensors for optimal performance
template <typename T>
void PrepareTensors(const DenseTensor& A,
                    const DenseTensor& B,
                    const DenseTensor& C,
                    DenseTensor* out_D,
                    const OperationConfig& config,
                    const phi::DeviceContext& dev_ctx) {
  // This function would handle tensor preparation like:
  // - Ensuring contiguous memory layout
  // - Pre-allocating workspace if needed
  // - Setting up tensor descriptors

  // Implementation would be device-specific
}

// Compute output scale for scaled operations
template <typename T>
void ComputeOutputScale(const ScaledTensor<T>& A,
                        const ScaledTensor<T>& B,
                        const ScaledTensor<T>& C,
                        const paddle::optional<DenseTensor>& D_scale,
                        DenseTensor* out_D_scale,
                        const phi::DeviceContext& dev_ctx) {
  // This function would compute the output scale based on input scales
  // Implementation would be device-specific
}

}  // namespace utils
}  // namespace unified_linear
}  // namespace phi
