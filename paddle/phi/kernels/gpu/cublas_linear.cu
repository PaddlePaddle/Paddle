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

#include "paddle/phi/kernels/gpu/cublas_linear.h"
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
typedef int cublasMath_t;
typedef int cublasPointerMode_t;
typedef int cublasAtomicsMode_t;
typedef int cublasGemmAlgo_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0

inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t handle,
                                        cublasMath_t mode) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetPointerMode(cublasHandle_t handle,
                                           cublasPointerMode_t mode) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle,
                                           cublasAtomicsMode_t mode) {
  return CUBLAS_STATUS_SUCCESS;
}
#endif

// CublasLinear implementation
CublasLinear::CublasLinear(const GPUContext& dev_ctx)
    : dev_ctx_(dev_ctx),
      math_mode_(CUBLAS_DEFAULT_MATH),
      pointer_mode_(CUBLAS_POINTER_MODE_DEVICE),
      atomics_mode_(CUBLAS_ATOMICS_ALLOWED),
      gemm_algo_(CUBLAS_GEMM_DEFAULT),
      has_error_(false) {
  InitializeHandle();
}

CublasLinear::~CublasLinear() { CleanupHandle(); }

void CublasLinear::InitializeHandle() {
  cublasStatus_t status = cublasCreate(&handle_);
  CheckCublasError(status, "cublasCreate");

  // Set default configurations
  status = cublasSetMathMode(handle_, math_mode_);
  CheckCublasError(status, "cublasSetMathMode");

  status = cublasSetPointerMode(handle_, pointer_mode_);
  CheckCublasError(status, "cublasSetPointerMode");

  status = cublasSetAtomicsMode(handle_, atomics_mode_);
  CheckCublasError(status, "cublasSetAtomicsMode");

  // Set stream from context
  stream_ = dev_ctx_.stream();
  status = cublasSetStream(handle_, stream_);
  CheckCublasError(status, "cublasSetStream");
}

void CublasLinear::CleanupHandle() {
  if (handle_) {
    cublasStatus_t status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "Failed to destroy cuBLAS handle: "
                 << GetCublasErrorString(status);
    }
    handle_ = nullptr;
  }
}

template <typename T>
void CublasLinear::Execute(const UnifiedLinearDescriptor& desc) {
  has_error_ = false;
  last_error_.clear();

  try {
    // Determine execution strategy
    const auto& input_dims = desc.input->dims();
    const auto& weight_dims = desc.weight->dims();

    int64_t batch_size = 1;
    for (int i = 0; i < input_dims.size() - 2; ++i) {
      batch_size *= input_dims[i];
    }

    if (batch_size == 1) {
      // Single matrix multiplication
      ExecuteGemmInternal<T>(desc);
    } else {
      // Batched matrix multiplication
      if (desc.input->stride() == desc.weight->stride()) {
        ExecuteGemmStridedBatchedInternal<T>(desc);
      } else {
        ExecuteGemmBatchedInternal<T>(desc);
      }
    }
    // Record end event for profiling
    RecordEndEvent();
  } catch (const std::exception& e) {
    has_error_ = true;
    last_error_ = e.what();
    throw;
  }
}

template <typename T>
void CublasLinear::ExecuteGemmInternal(const UnifiedLinearDescriptor& desc) {
  // Extract matrix dimensions
  const auto& input_dims = desc.input->dims();
  const auto& weight_dims = desc.weight->dims();

  int m = input_dims[input_dims.size() - (desc.transpose_input ? 2 : 1)];
  int n = weight_dims[weight_dims.size() - (desc.transpose_weight ? 2 : 1)];
  int k = input_dims[input_dims.size() - (desc.transpose_input ? 1 : 2)];

  // Get data pointers
  const T* A = desc.input->data<T>();
  const T* B = desc.weight->data<T>();
  T* C = desc.output->data<T>();

  // Convert scalar values to appropriate types
  float alpha_float = desc.alpha.to<float>();
  float beta_float = desc.beta.to<float>();

  // Get cuBLAS operation types
  cublasOperation_t trans_a = GetCublasOperation(desc.transpose_input);
  cublasOperation_t trans_b = GetCublasOperation(desc.transpose_weight);

  // Get leading dimensions
  int lda = (trans_a == CUBLAS_OP_N) ? k : m;
  int ldb = (trans_b == CUBLAS_OP_N) ? n : k;
  int ldc = n;

  // Get optimal algorithm
  cublasGemmAlgo_t algo = GetOptimalGemmAlgorithm(desc);

  // Record start event for profiling
  RecordStartEvent();

  // Execute GEMM based on data type
  if (std::is_same<T, float>::value) {
    cublasStatus_t status =
        cublasSgemmEx(handle_,
                      trans_b,
                      trans_a,  // Note: cuBLAS uses column-major order
                      n,
                      m,
                      k,
                      &alpha_float,
                      B,
                      CUDA_R_32F,
                      ldb,
                      A,
                      CUDA_R_32F,
                      lda,
                      &beta_float,
                      C,
                      CUDA_R_32F,
                      ldc);
    CheckCublasError(status, "cublasSgemmEx");
  } else if (std::is_same<T, double>::value) {
    cublasStatus_t status = cublasDgemm(handle_,
                                        trans_b,
                                        trans_a,
                                        n,
                                        m,
                                        k,
                                        &alpha_float,
                                        B,
                                        ldb,
                                        A,
                                        lda,
                                        &beta_float,
                                        C,
                                        ldc);
    CheckCublasError(status, "cublasDgemm");
  } else if (std::is_same<T, phi::float16>::value) {
    // For FP16, use cublasGemmEx for better performance
    cublasStatus_t status = cublasGemmEx(handle_,
                                         trans_b,
                                         trans_a,
                                         n,
                                         m,
                                         k,
                                         &alpha_float,
                                         B,
                                         CUDA_R_16F,
                                         ldb,
                                         A,
                                         CUDA_R_16F,
                                         lda,
                                         &beta_float,
                                         C,
                                         CUDA_R_16F,
                                         ldc,
                                         CUBLAS_COMPUTE_32F_FAST_16F,
                                         algo);
    CheckCublasError(status, "cublasGemmEx");
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Unsupported data type for cuBLAS GEMM"));
  }
}

template <typename T>
void CublasLinear::ExecuteGemmBatchedInternal(
    const UnifiedLinearDescriptor& desc) {
  // Extract matrix dimensions
  const auto& input_dims = desc.input->dims();
  const auto& weight_dims = desc.weight->dims();

  int batch_count = 1;
  for (int i = 0; i < input_dims.size() - 2; ++i) {
    batch_count *= input_dims[i];
  }

  int m = input_dims[input_dims.size() - (desc.transpose_input ? 2 : 1)];
  int n = weight_dims[weight_dims.size() - (desc.transpose_weight ? 2 : 1)];
  int k = input_dims[input_dims.size() - (desc.transpose_input ? 1 : 2)];

  // Prepare batched pointers
  std::vector<const T*> A_array(batch_count);
  std::vector<const T*> B_array(batch_count);
  std::vector<T*> C_array(batch_count);

  // Fill pointer arrays (simplified - actual implementation would handle
  // strides)
  for (int i = 0; i < batch_count; ++i) {
    A_array[i] = desc.input->data<T>() + i * m * k;
    B_array[i] = desc.weight->data<T>() + i * k * n;
    C_array[i] = desc.output->data<T>() + i * m * n;
  }

  // Convert scalar values
  float alpha_float = desc.alpha.to<float>();
  float beta_float = desc.beta.to<float>();

  // Get cuBLAS operation types
  cublasOperation_t trans_a = GetCublasOperation(desc.transpose_input);
  cublasOperation_t trans_b = GetCublasOperation(desc.transpose_weight);

  // Get leading dimensions
  int lda = (trans_a == CUBLAS_OP_N) ? k : m;
  int ldb = (trans_b == CUBLAS_OP_N) ? n : k;
  int ldc = n;

  // Execute batched GEMM
  if (std::is_same<T, float>::value) {
    cublasStatus_t status = cublasSgemmBatched(
        handle_,
        trans_b,
        trans_a,
        n,
        m,
        k,
        &alpha_float,
        const_cast<const float**>(reinterpret_cast<float**>(B_array.data())),
        ldb,
        const_cast<const float**>(reinterpret_cast<float**>(A_array.data())),
        lda,
        &beta_float,
        reinterpret_cast<float**>(C_array.data()),
        ldc,
        batch_count);
    CheckCublasError(status, "cublasSgemmBatched");
  } else if (std::is_same<T, double>::value) {
    cublasStatus_t status = cublasDgemmBatched(
        handle_,
        trans_b,
        trans_a,
        n,
        m,
        k,
        &alpha_float,
        const_cast<const double**>(reinterpret_cast<double**>(B_array.data())),
        ldb,
        const_cast<const double**>(reinterpret_cast<double**>(A_array.data())),
        lda,
        &beta_float,
        reinterpret_cast<double**>(C_array.data()),
        ldc,
        batch_count);
    CheckCublasError(status, "cublasDgemmBatched");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported data type for batched GEMM"));
  }
}

template <typename T>
void CublasLinear::ExecuteGemmStridedBatchedInternal(
    const UnifiedLinearDescriptor& desc) {
  // Similar to batched but with strides
  // Implementation would handle strided batched operations
  PADDLE_THROW(common::errors::Unimplemented(
      "Strided batched GEMM not yet implemented"));
}

// Configuration methods
void CublasLinear::SetMathMode(cublasMath_t math_mode) {
  math_mode_ = math_mode;
  if (handle_) {
    cublasStatus_t status = cublasSetMathMode(handle_, math_mode);
    CheckCublasError(status, "cublasSetMathMode");
  }
}

void CublasLinear::SetPointerMode(cublasPointerMode_t pointer_mode) {
  pointer_mode_ = pointer_mode;
  if (handle_) {
    cublasStatus_t status = cublasSetPointerMode(handle_, pointer_mode);
    CheckCublasError(status, "cublasSetPointerMode");
  }
}

void CublasLinear::SetAtomicsMode(cublasAtomicsMode_t atomics_mode) {
  atomics_mode_ = atomics_mode;
  if (handle_) {
    cublasStatus_t status = cublasSetAtomicsMode(handle_, atomics_mode);
    CheckCublasError(status, "cublasSetAtomicsMode");
  }
}

void CublasLinear::SetGemmAlgorithm(cublasGemmAlgo_t algo) {
  gemm_algo_ = algo;
}

void CublasLinear::EnableTensorCores() {
#if CUDA_VERSION >= 9000
  SetMathMode(CUBLAS_TENSOR_OP_MATH);
#endif
}

void CublasLinear::DisableTensorCores() { SetMathMode(CUBLAS_DEFAULT_MATH); }

void CublasLinear::SetStream(cudaStream_t stream) {
  stream_ = stream;
  if (handle_) {
    cublasStatus_t status = cublasSetStream(handle_, stream);
    CheckCublasError(status, "cublasSetStream");
  }
}

void CublasLinear::RecordEvent(cudaEvent_t event) {
  // Implementation would record events for synchronization
}

void CublasLinear::RecordStartEvent() {
  // Implementation would record start event for profiling
}

void CublasLinear::RecordEndEvent() {
  // Implementation would record end event for profiling
}

// Helper methods
cublasOperation_t CublasLinear::GetCublasOperation(bool transpose) {
  return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}

cublasGemmAlgo_t CublasLinear::GetOptimalGemmAlgorithm(
    const UnifiedLinearDescriptor& desc) {
  // For now, return the configured algorithm
  // In a full implementation, this would consider tensor sizes, data types,
  // etc.
  return gemm_algo_;
}

void CublasLinear::CheckCublasError(cublasStatus_t status,
                                    const std::string& operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string error_msg =
        "cuBLAS error in " + operation + ": " + GetCublasErrorString(status);
    SetError(error_msg);
    PADDLE_THROW(common::errors::External(error_msg));
  }
}

void CublasLinear::SetError(const std::string& error_message) {
  has_error_ = true;
  last_error_ = error_message;
}

std::string CublasLinear::GetCublasErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
      return "Unknown cuBLAS error: " + std::to_string(status);
  }
}

// Explicit template instantiations
template void CublasLinear::Execute<float>(const UnifiedLinearDescriptor& desc);
template void CublasLinear::Execute<double>(
    const UnifiedLinearDescriptor& desc);
template void CublasLinear::Execute<phi::float16>(
    const UnifiedLinearDescriptor& desc);

}  // namespace phi
