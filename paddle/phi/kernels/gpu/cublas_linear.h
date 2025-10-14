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
#include <string>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/unified_linear_kernel.h"

#ifdef __APPLE__
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cublasHandle_t;
typedef int cublasMath_t;
typedef int cublasPointerMode_t;
typedef int cublasAtomicsMode_t;
typedef int cublasGemmAlgo_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifdef PADDLE_WITH_CUBLASLT
#include <cublasLt.h>
#endif
#endif

namespace phi {

// cuBLAS-specific linear operation implementation
// This class handles:
// 1. cuBLAS handle management
// 2. GEMM operation configuration
// 3. Math mode and pointer mode settings
// 4. Error handling and event recording
// 5. Flag and mode parsing without global side effects
class CublasLinear {
 public:
  explicit CublasLinear(const GPUContext& dev_ctx);
  ~CublasLinear();

  // Main execution interface
  template <typename T>
  void Execute(const UnifiedLinearDescriptor& desc);

  // Configuration methods
  void SetMathMode(cublasMath_t math_mode);
  void SetPointerMode(cublasPointerMode_t pointer_mode);
  void SetAtomicsMode(cublasAtomicsMode_t atomics_mode);

  // Algorithm selection
  void SetGemmAlgorithm(cublasGemmAlgo_t algo);
  void EnableTensorCores();
  void DisableTensorCores();

  // Stream and event management
  void SetStream(cudaStream_t stream);
  void RecordEvent(cudaEvent_t event);

  // Error handling
  std::string GetLastError() const { return last_error_; }
  bool HasError() const { return has_error_; }

 private:
  const GPUContext& dev_ctx_;
  cublasHandle_t handle_;
  cudaStream_t stream_;

  // Configuration states
  cublasMath_t math_mode_;
  cublasPointerMode_t pointer_mode_;
  cublasAtomicsMode_t atomics_mode_;
  cublasGemmAlgo_t gemm_algo_;

  // Error tracking
  bool has_error_;
  std::string last_error_;

  // Internal methods
  void InitializeHandle();
  void CleanupHandle();

  // GEMM execution methods
  template <typename T>
  void ExecuteGemmInternal(const UnifiedLinearDescriptor& desc);

  template <typename T>
  void ExecuteGemmBatchedInternal(const UnifiedLinearDescriptor& desc);

  template <typename T>
  void ExecuteGemmStridedBatchedInternal(const UnifiedLinearDescriptor& desc);

  // Type conversion helpers
  template <typename T>
  struct CublasTypeTraits;

  // cuBLAS operation configuration
  cublasOperation_t GetCublasOperation(bool transpose);
  cublasGemmAlgo_t GetOptimalGemmAlgorithm(const UnifiedLinearDescriptor& desc);

  // Workspace management
  void* AllocateWorkspace(size_t size);
  void FreeWorkspace(void* workspace);

  // Error handling
  void SetError(const std::string& error_message);
  void CheckCublasError(cublasStatus_t status, const std::string& operation);

  // Event recording for profiling
  void RecordStartEvent();
  void RecordEndEvent();

  // Math mode utilities
  cublasMath_t GetOptimalMathMode(const UnifiedLinearDescriptor& desc);
  bool ShouldUseTensorCores(const UnifiedLinearDescriptor& desc);
};

// cuBLASLt-specific implementation (if available)
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000

class CublasLtLinear {
 public:
  explicit CublasLtLinear(const GPUContext& dev_ctx);
  ~CublasLtLinear();

  // Main execution interface
  template <typename T>
  void Execute(const UnifiedLinearDescriptor& desc);

  // Configuration methods
  void SetEpilogue(cublasLtEpilogue_t epilogue);
  void SetComputeType(cublasComputeType_t compute_type);
  void SetScaleType(cudaDataType_t scale_type);

  // Algorithm selection
  void SetAlgorithm(const cublasLtMatmulAlgo_t& algo);
  void EnableHeuristicSearch();
  void DisableHeuristicSearch();

  // Workspace configuration
  void SetWorkspaceSize(size_t workspace_size);
  void* GetWorkspace();

 private:
  const GPUContext& dev_ctx_;
  cublasLtHandle_t handle_;

  // Operation descriptors
  cublasLtMatmulDesc_t operation_desc_;
  cublasLtMatrixLayout_t input_layout_;
  cublasLtMatrixLayout_t weight_layout_;
  cublasLtMatrixLayout_t output_layout_;
  cublasLtMatrixLayout_t bias_layout_;

  // Configuration
  cublasLtEpilogue_t epilogue_;
  cublasComputeType_t compute_type_;
  cudaDataType_t scale_type_;

  // Algorithm selection
  cublasLtMatmulAlgo_t selected_algo_;
  bool use_heuristic_search_;

  // Workspace
  void* workspace_;
  size_t workspace_size_;

  // Internal methods
  void InitializeHandle();
  void CleanupHandle();

  // Descriptor creation
  void CreateOperationDescriptor(const UnifiedLinearDescriptor& desc);
  void CreateMatrixLayoutDescriptors(const UnifiedLinearDescriptor& desc);
  void DestroyDescriptors();

  // Algorithm selection
  cublasLtMatmulAlgo_t SelectOptimalAlgorithm(
      const UnifiedLinearDescriptor& desc);

  // Execution
  template <typename T>
  void ExecuteMatmulInternal(const UnifiedLinearDescriptor& desc);

  // Error handling
  void CheckCublasLtError(cublasStatus_t status, const std::string& operation);
};

#endif  // CUDART_VERSION >= 11000

// Type traits for cuBLAS operations
template <typename T>
struct CublasTypeTraits {
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static constexpr bool supports_tensor_cores = false;
};

// Specializations for different data types
template <>
struct CublasTypeTraits<float> {
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static constexpr bool supports_tensor_cores = true;
};

template <>
struct CublasTypeTraits<double> {
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_64F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
  static constexpr bool supports_tensor_cores = false;
};

template <>
struct CublasTypeTraits<phi::float16> {
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_16F;
  static constexpr cublasComputeType_t compute_type =
      CUBLAS_COMPUTE_32F_FAST_16F;
  static constexpr bool supports_tensor_cores = true;
};

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
template <>
struct CublasTypeTraits<phi::bfloat16> {
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_16BF;
  static constexpr cublasComputeType_t compute_type =
      CUBLAS_COMPUTE_32F_FAST_16BF;
  static constexpr bool supports_tensor_cores = true;
};
#endif

// Utility functions for cuBLAS operations
namespace cublas_linear {

// Get optimal GEMM algorithm for given configuration
cublasGemmAlgo_t GetOptimalGemmAlgorithm(
    int m, int n, int k, DataType dtype, bool use_tensor_cores);

// Check if tensor cores should be used for given configuration
bool ShouldUseTensorCores(int m, int n, int k, DataType dtype);

// Get optimal math mode for given operation
cublasMath_t GetOptimalMathMode(const UnifiedLinearDescriptor& desc);

// Convert activation to cuBLAS math mode flags
int ConvertActivationToFlags(const std::string& activation);

// Get workspace size for given algorithm
size_t GetWorkspaceSize(cublasGemmAlgo_t algo, int m, int n, int k);

}  // namespace cublas_linear

}  // namespace phi
