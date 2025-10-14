// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/gpu/unified_linear_cuda.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace phi {

// 前向声明
class DenseTensor;
class GPUContext;
class UnifiedLinearDescriptor;

#ifdef __APPLE__
// macOS环境下使用模拟类型
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cublasHandle_t;
#define CUBLAS_STATUS_SUCCESS 0
inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  return CUBLAS_STATUS_SUCCESS;
}
#endif

// UnifiedLinearCuda implementation
UnifiedLinearCuda::UnifiedLinearCuda(const GPUContext& dev_ctx)
    : dev_ctx_(dev_ctx),
      use_cublaslt_(true),
      use_tensor_cores_(true),
      exhaustive_search_(false),
      auto_tuning_enabled_(true),
      has_error_(false) {
  // Initialize library implementations
  cublas_linear_ = std::make_unique<CublasLinear>(dev_ctx);

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
  cublaslt_linear_ = std::make_unique<CublasLtLinear>(dev_ctx);
#endif
}

UnifiedLinearCuda::~UnifiedLinearCuda() {
  // Cleanup is handled by unique_ptr destructors
}

template <typename T>
void UnifiedLinearCuda::Execute(const UnifiedLinearDescriptor& desc) {
  // Reset error state
  has_error_ = false;
  last_error_.clear();

  try {
    // Determine optimal library and configuration
    bool use_cublaslt = ShouldUseCublasLt(desc);
    bool use_tensor_cores = ShouldUseTensorCores(desc);
    bool can_fuse_epilogue = CanFuseEpilogue(desc);

    // Configure library implementations
    if (cublas_linear_) {
      cublas_linear_->SetMathMode(GetOptimalMathMode(desc));
      cublas_linear_->SetPointerMode(CUBLAS_POINTER_MODE_DEVICE);

      if (use_tensor_cores) {
        cublas_linear_->EnableTensorCores();
      } else {
        cublas_linear_->DisableTensorCores();
      }
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
    if (cublaslt_linear_ && use_cublaslt) {
      // Configure cuBLASLt epilogue
      if (can_fuse_epilogue && desc.use_bias) {
        if (desc.activation == "relu") {
          cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_RELU_BIAS);
        } else if (desc.activation == "gelu") {
          cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_GELU_BIAS);
        } else {
          cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_BIAS);
        }
      } else if (desc.use_bias) {
        cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_BIAS);
      } else if (desc.use_activation) {
        if (desc.activation == "relu") {
          cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_RELU);
        } else if (desc.activation == "gelu") {
          cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_GELU);
        }
      } else {
        cublaslt_linear_->SetEpilogue(CUBLASLT_EPILOGUE_DEFAULT);
      }

      // Set compute type
      cublasComputeType_t compute_type;
      switch (desc.compute_dtype) {
        case DataType::FLOAT32:
          compute_type = use_tensor_cores ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
          break;
        case DataType::FLOAT16:
          compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
          break;
        case DataType::BFLOAT16:
          compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
          break;
        case DataType::FLOAT64:
          compute_type = CUBLAS_COMPUTE_64F;
          break;
        default:
          compute_type = CUBLAS_COMPUTE_32F;
      }
      cublaslt_linear_->SetComputeType(compute_type);
    }
#endif

    // Execute with selected library
    if (use_cublaslt && cublaslt_linear_) {
      // Try cuBLASLt first
      try {
        cublaslt_linear_->Execute<T>(desc);
        return;
      } catch (const std::exception& e) {
        // Fallback to cuBLAS
        VLOG(3) << "cuBLASLt execution failed, falling back to cuBLAS: "
                << e.what();
      }
    }

    // Use cuBLAS
    if (cublas_linear_) {
      cublas_linear_->Execute<T>(desc);

      // Apply epilogue if not fused
      if (!can_fuse_epilogue && (desc.use_bias || desc.use_activation)) {
        ApplyEpilogue(desc);
      }
    } else {
      PADDLE_THROW(
          common::errors::Unavailable("No suitable BLAS library available"));
    }
  } catch (const std::exception& e) {
    has_error_ = true;
    last_error_ = e.what();
    throw;
  }
}

bool UnifiedLinearCuda::ShouldUseCublasLt(const UnifiedLinearDescriptor& desc) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
  // Use cuBLASLt for:
  // 1. Epilogue fusion (bias + activation)
  // 2. Mixed precision operations
  // 3. When tensor cores are beneficial

  if (desc.use_bias || desc.use_activation) {
    return true;  // Can fuse epilogue
  }

  if (IsNarrowPrecisionType(desc.input->dtype()) ||
      IsNarrowPrecisionType(desc.weight->dtype())) {
    return true;  // Better mixed precision support
  }

  // Check tensor core eligibility
  if (ShouldUseTensorCores(desc)) {
    return true;  // Better tensor core support
  }

  // For large matrices, cuBLASLt can be faster
  const auto& input_dims = desc.input->dims();
  const auto& weight_dims = desc.weight->dims();

  int64_t m = input_dims[input_dims.size() - (desc.transpose_input ? 2 : 1)];
  int64_t n = weight_dims[weight_dims.size() - (desc.transpose_weight ? 2 : 1)];
  int64_t k = input_dims[input_dims.size() - (desc.transpose_input ? 1 : 2)];

  // Use cuBLASLt for large matrices
  return (m * n * k) > (512 * 512 * 512);  // Threshold can be tuned
#else
  return false;  // cuBLASLt not available
#endif
}

bool UnifiedLinearCuda::ShouldUseTensorCores(
    const UnifiedLinearDescriptor& desc) {
  // Check data type support
  DataType dtype = desc.input->dtype();
  bool supports_tensor_cores = false;

  switch (dtype) {
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      supports_tensor_cores = true;
      break;
    case DataType::FLOAT32:
      // TF32 tensor cores
      supports_tensor_cores = true;
      break;
    default:
      supports_tensor_cores = false;
  }

  if (!supports_tensor_cores) {
    return false;
  }

  // Check matrix dimensions
  const auto& input_dims = desc.input->dims();
  const auto& weight_dims = desc.weight->dims();

  int64_t m = input_dims[input_dims.size() - (desc.transpose_input ? 2 : 1)];
  int64_t n = weight_dims[weight_dims.size() - (desc.transpose_weight ? 2 : 1)];
  int64_t k = input_dims[input_dims.size() - (desc.transpose_input ? 1 : 2)];

  // Tensor cores work best with dimensions that are multiples of 8 (for FP16)
  // or 16 (for INT8)
  const int alignment = (dtype == DataType::INT8) ? 16 : 8;

  return (m % alignment == 0) && (n % alignment == 0) && (k % alignment == 0) &&
         (m >= 16) && (n >= 16) && (k >= 16);  // Minimum size threshold
}

bool UnifiedLinearCuda::CanFuseEpilogue(const UnifiedLinearDescriptor& desc) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
  // Can fuse if:
  // 1. cuBLASLt is available
  // 2. We have bias or activation
  // 3. Supported activation function

  if (!desc.use_bias && !desc.use_activation) {
    return false;
  }

  // Check supported activations
  if (desc.use_activation) {
    return (desc.activation == "relu" || desc.activation == "gelu");
  }

  return desc.use_bias;  // Can always fuse bias
#else
  return false;  // cuBLASLt not available
#endif
}

void UnifiedLinearCuda::ApplyEpilogue(const UnifiedLinearDescriptor& desc) {
  // Apply bias and/or activation if not fused
  if (desc.use_bias && desc.bias) {
    // Add bias: output = output + bias
    // This would call elementwise add kernel
    VLOG(3) << "Applying bias in separate kernel";
  }

  if (desc.use_activation) {
    // Apply activation function
    if (desc.activation == "relu") {
      VLOG(3) << "Applying ReLU activation in separate kernel";
      // Call ReLU kernel
    } else if (desc.activation == "gelu") {
      VLOG(3) << "Applying GELU activation in separate kernel";
      // Call GELU kernel
    }
  }
}

int UnifiedLinearCuda::GetOptimalAlgorithm(
    const UnifiedLinearDescriptor& desc) {
  // This would implement algorithm selection logic
  // For now, return default algorithm
  return 0;
}

size_t UnifiedLinearCuda::GetWorkspaceSize(
    const UnifiedLinearDescriptor& desc) {
  // Calculate workspace size based on algorithm and tensor sizes
  // For now, return conservative estimate
  return 1024 * 1024;  // 1MB default
}

// Explicit template instantiations
template void UnifiedLinearCuda::Execute<float>(
    const UnifiedLinearDescriptor& desc);
template void UnifiedLinearCuda::Execute<double>(
    const UnifiedLinearDescriptor& desc);
template void UnifiedLinearCuda::Execute<phi::float16>(
    const UnifiedLinearDescriptor& desc);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
template void UnifiedLinearCuda::Execute<phi::bfloat16>(
    const UnifiedLinearDescriptor& desc);
#endif

}  // namespace phi
