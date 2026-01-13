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

namespace phi {

// Forward declarations for hardware-agnostic layer
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         const DenseTensor& D,
                         const paddle::optional<DenseTensor>& Bias,
                         const paddle::optional<DenseTensor>& A_scale,
                         const paddle::optional<DenseTensor>& B_scale,
                         const paddle::optional<DenseTensor>& C_scale,
                         const paddle::optional<DenseTensor>& D_scale,
                         float alpha,
                         float beta,
                         bool trans_A,
                         bool trans_B,
                         bool trans_C,
                         const std::string& activation,
                         DenseTensor* out_D,
                         DenseTensor* out_D_scale);

// Hardware-agnostic helper functions
namespace unified_linear {

// Enum for operation types
enum class LinearOpType {
  kDot = 0,    // Vector dot product
  kMv = 1,     // Matrix-vector multiplication
  kMm = 2,     // Matrix-matrix multiplication
  kBmm = 3,    // Batched matrix-matrix multiplication
  kLinear = 4  // Linear transformation (with bias)
};

// Enum for activation types
enum class ActivationType {
  kNone = 0,
  kRelu = 1,
  kGelu = 2,
  kSigmoid = 3,
  kTanh = 4
};

// Enum for backend types
enum class BackendType { kCublas = 0, kCublasLt = 1 };

// Structure to hold operation configuration
struct OperationConfig {
  BackendType backend;
  bool use_fused_epilogue;
  bool enable_fast_math;
  bool allow_tf32;
  bool auto_tune;

  // Constructor with default values
  OperationConfig()
      : backend(BackendType::kCublasLt),
        use_fused_epilogue(true),
        enable_fast_math(true),
        allow_tf32(true),
        auto_tune(true) {}
};

// Convert string to activation type
inline ActivationType StringToActivationType(const std::string& activation) {
  if (activation == "relu" || activation == "Relu") {
    return ActivationType::kRelu;
  } else if (activation == "gelu" || activation == "Gelu") {
    return ActivationType::kGelu;
  } else if (activation == "sigmoid" || activation == "Sigmoid") {
    return ActivationType::kSigmoid;
  } else if (activation == "tanh" || activation == "Tanh") {
    return ActivationType::kTanh;
  }
  return ActivationType::kNone;
}

// Determine operation type based on tensor dimensions
inline LinearOpType DetermineOpType(const DenseTensor& A,
                                    const DenseTensor& B,
                                    const DenseTensor& C) {
  const auto& A_dims = A.dims();
  const auto& B_dims = B.dims();
  const auto& C_dims = C.dims();

  // Dot product (1D * 1D)
  if (A_dims.size() == 1 && B_dims.size() == 1) {
    return LinearOpType::kDot;
  }

  // Matrix-vector multiplication
  if ((A_dims.size() == 2 && B_dims.size() == 1) ||
      (A_dims.size() == 1 && B_dims.size() == 2)) {
    return LinearOpType::kMv;
  }

  // Batched matrix-matrix multiplication
  if (A_dims.size() > 2 && B_dims.size() > 2) {
    return LinearOpType::kBmm;
  }

  // Check if this is a linear transformation (has bias)
  if (C_dims.size() == 0 && C.numel() == 0) {
    return LinearOpType::kLinear;
  }

  // Default to matrix-matrix multiplication
  return LinearOpType::kMm;
}

// Heuristic function to determine optimal backend
template <typename T>
OperationConfig DetermineOptimalConfig(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    const paddle::optional<DenseTensor>& bias,
    ActivationType activation,
    const phi::DeviceContext& dev_ctx) {
  OperationConfig config;

  // Default to cuBLASLt for most cases, fallback to cuBLAS for small tensors
  // This aligns with PyTorch's strategy
  if (A.numel() < 1024 || B.numel() < 1024) {
    config.backend = BackendType::kCublas;
  } else {
    config.backend = BackendType::kCublasLt;
  }

  // Use fused epilogue if bias or activation is present
  config.use_fused_epilogue =
      bias.is_initialized() || activation != ActivationType::kNone;

  // Enable fast math for non-critical computations
  config.enable_fast_math = true;

  // Allow TF32 for non-int8 computations
  config.allow_tf32 =
      A.dtype() != DataType::INT8 && B.dtype() != DataType::INT8;

  return config;
}

// Validate tensor dimensions and compatibility
template <typename T>
void ValidateTensorDims(const DenseTensor& A,
                        const DenseTensor& B,
                        const DenseTensor& C,
                        bool trans_A,
                        bool trans_B) {
  const auto& A_dims = A.dims();
  const auto& B_dims = B.dims();

  // Get effective dimensions after transpose
  int A_cols = trans_A ? A_dims[1] : A_dims[0];
  int B_rows = trans_B ? B_dims[1] : B_dims[0];

  // Check matrix multiplication compatibility
  PADDLE_ENFORCE_EQ(
      A_cols,
      B_rows,
      phi::errors::InvalidArgument(
          "The number of columns in A must equal the number of rows in B, "
          "but received A_cols=%d and B_rows=%d.",
          A_cols,
          B_rows));
}

// Prepare output dimensions based on operation type
phi::DDim PrepareOutputDims(const DenseTensor& A,
                            const DenseTensor& B,
                            const DenseTensor& C,
                            bool trans_A,
                            bool trans_B,
                            LinearOpType op_type) {
  const auto& A_dims = A.dims();
  const auto& B_dims = B.dims();
  const auto& C_dims = C.dims();

  switch (op_type) {
    case LinearOpType::kDot: {
      // Dot product returns a scalar
      return phi::make_ddim({1});
    }
    case LinearOpType::kMv: {
      // Matrix-vector multiplication
      if (A_dims.size() == 2 && B_dims.size() == 1) {
        int m = trans_A ? A_dims[1] : A_dims[0];
        return phi::make_ddim({m});
      } else {
        int n = trans_B ? B_dims[1] : B_dims[0];
        return phi::make_ddim({n});
      }
    }
    case LinearOpType::kMm: {
      // Matrix-matrix multiplication
      int m = trans_A ? A_dims[1] : A_dims[0];
      int n = trans_B ? B_dims[0] : B_dims[1];
      return phi::make_ddim({m, n});
    }
    case LinearOpType::kBmm: {
      // Batched matrix-matrix multiplication
      int batch_size = A_dims[0];
      int m = trans_A ? A_dims[2] : A_dims[1];
      int n = trans_B ? B_dims[1] : B_dims[2];
      return phi::make_ddim({batch_size, m, n});
    }
    case LinearOpType::kLinear: {
      // Linear transformation
      int m = trans_A ? A_dims[1] : A_dims[0];
      int n = trans_B ? B_dims[0] : B_dims[1];
      return phi::make_ddim({m, n});
    }
    default: {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Unsupported operation type for unified linear"));
    }
  }
}

// Apply activation function
template <typename T>
void ApplyActivation(const DenseTensor& input,
                     ActivationType activation,
                     DenseTensor* output) {
  switch (activation) {
    case ActivationType::kRelu:
      // ReLU activation
      // Implementation would go here
      break;
    case ActivationType::kGelu:
      // GELU activation
      // Implementation would go here
      break;
    case ActivationType::kSigmoid:
      // Sigmoid activation
      // Implementation would go here
      break;
    case ActivationType::kTanh:
      // Tanh activation
      // Implementation would go here
      break;
    case ActivationType::kNone:
    default:
      // No activation
      break;
  }
}

}  // namespace unified_linear
}  // namespace phi
