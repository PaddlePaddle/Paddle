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

// 简化版本，适配非CUDA环境
namespace phi {

// 前向声明
class DenseTensor;
class GPUContext;

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
#else
// CUDA环境
#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifdef PADDLE_WITH_CUBLASLT
#include <cublasLt.h>
#endif
#endif
#endif

// Forward declaration
class CublasLinear;
class CublasLtLinear;

// Library-agnostic unified linear CUDA implementation
// This layer handles:
// 1. Library selection (cuBLAS vs cuBLASLt)
// 2. Fallback strategies
// 3. Epilogue handling (bias, activation)
// 4. Performance heuristics
class UnifiedLinearCuda {
 public:
  explicit UnifiedLinearCuda(const GPUContext& dev_ctx);
  ~UnifiedLinearCuda();

  // Main execution function
  template <typename T>
  void Execute(const UnifiedLinearDescriptor& desc);

  // Configuration
  void SetUseCublasLt(bool use_cublaslt) { use_cublaslt_ = use_cublaslt; }
  void SetUseTensorCores(bool use_tensor_cores) {
    use_tensor_cores_ = use_tensor_cores;
  }
  void SetExhaustiveSearch(bool exhaustive_search) {
    exhaustive_search_ = exhaustive_search;
  }

  // Performance tuning
  void EnableAutoTuning() { auto_tuning_enabled_ = true; }
  void DisableAutoTuning() { auto_tuning_enabled_ = false; }

 private:
  const GPUContext& dev_ctx_;

  // Library selection flags
  bool use_cublaslt_ = true;
  bool use_tensor_cores_ = true;
  bool exhaustive_search_ = false;
  bool auto_tuning_enabled_ = true;

  // Library implementations
  std::unique_ptr<CublasLinear> cublas_linear_;
  std::unique_ptr<CublasLtLinear> cublaslt_linear_;

  // Internal methods
  bool ShouldUseCublasLt(const UnifiedLinearDescriptor& desc);
  bool ShouldUseTensorCores(const UnifiedLinearDescriptor& desc);
  bool CanFuseEpilogue(const UnifiedLinearDescriptor& desc);

  // Fallback logic
  void FallbackToCublas(const UnifiedLinearDescriptor& desc);
  void FallbackToReference(const UnifiedLinearDescriptor& desc);

  // Epilogue handling
  void ApplyEpilogue(const UnifiedLinearDescriptor& desc);

  // Performance heuristics
  int GetOptimalAlgorithm(const UnifiedLinearDescriptor& desc);
  size_t GetWorkspaceSize(const UnifiedLinearDescriptor& desc);
};

// cuBLAS-specific implementation
class CublasLinear {
 public:
  explicit CublasLinear(const GPUContext& dev_ctx);
  ~CublasLinear();

  template <typename T>
  void ExecuteGemm(const UnifiedLinearDescriptor& desc);

  template <typename T>
  void ExecuteGemmBatched(const UnifiedLinearDescriptor& desc);

  void SetMathMode(cublasMath_t math_mode) { math_mode_ = math_mode; }
  void SetPointerMode(cublasPointerMode_t pointer_mode) {
    pointer_mode_ = pointer_mode;
  }

 private:
  const GPUContext& dev_ctx_;
  cublasHandle_t handle_;
  cublasMath_t math_mode_ = CUBLAS_DEFAULT_MATH;
  cublasPointerMode_t pointer_mode_ = CUBLAS_POINTER_MODE_DEVICE;

  // Internal helper methods
  cublasOperation_t GetCublasOperation(bool transpose);
  cudaDataType_t GetCudaDataType(DataType dtype);
  cublasGemmAlgo_t GetGemmAlgorithm(bool use_tensor_cores);

  // Error handling
  void CheckCublasError(cublasStatus_t status, const std::string& operation);
};

// cuBLASLt-specific implementation
class CublasLtLinear {
 public:
  explicit CublasLtLinear(const GPUContext& dev_ctx);
  ~CublasLtLinear();

  template <typename T>
  void ExecuteMatmul(const UnifiedLinearDescriptor& desc);

  void SetEpilogue(cublasLtEpilogue_t epilogue) { epilogue_ = epilogue; }
  void SetComputeType(cublasComputeType_t compute_type) {
    compute_type_ = compute_type;
  }

 private:
  const GPUContext& dev_ctx_;
  cublasLtHandle_t handle_;
  cublasLtEpilogue_t epilogue_ = CUBLASLT_EPILOGUE_DEFAULT;
  cublasComputeType_t compute_type_ = CUBLAS_COMPUTE_32F_FAST_16F;

  // Descriptor management
  cublasLtMatmulDesc_t CreateOperationDesc(const UnifiedLinearDescriptor& desc);
  cublasLtMatrixLayout_t CreateMatrixLayout(const DenseTensor& tensor,
                                            bool transpose);

  // Algorithm selection
  cublasLtMatmulAlgo_t SelectAlgorithm(const UnifiedLinearDescriptor& desc,
                                       void* workspace,
                                       size_t workspace_size);

  // Workspace management
  void* AllocateWorkspace(size_t size);
  void FreeWorkspace(void* workspace);

  // Error handling
  void CheckCublasLtError(cublasStatus_t status, const std::string& operation);
};

// Performance tuning and auto-tuning support
struct UnifiedLinearAutoTuner {
  struct TuningResult {
    bool use_cublaslt;
    bool use_tensor_cores;
    cublasGemmAlgo_t cublas_algo;
    cublasLtMatmulAlgo_t cublaslt_algo;
    size_t workspace_size;
    float execution_time;
  };

  // Auto-tuning for specific configuration
  TuningResult AutoTune(const GPUContext& dev_ctx,
                        const UnifiedLinearDescriptor& desc);

  // Cache management
  void CacheResult(const std::string& key, const TuningResult& result);
  bool GetCachedResult(const std::string& key, TuningResult* result);

 private:
  // Benchmark specific configuration
  float BenchmarkConfiguration(const GPUContext& dev_ctx,
                               const UnifiedLinearDescriptor& desc,
                               const TuningResult& config);

  // Generate cache key
  std::string GenerateCacheKey(const UnifiedLinearDescriptor& desc);
};

// Error handling utilities
class UnifiedLinearError {
 public:
  static void ThrowIfError(bool condition, const std::string& message);
  static void CheckCudaError(cudaError_t error, const std::string& operation);
  static void CheckCublasError(cublasStatus_t status,
                               const std::string& operation);
  static void CheckCublasLtError(cublasStatus_t status,
                                 const std::string& operation);

 private:
  static std::string GetCudaErrorString(cudaError_t error);
  static std::string GetCublasErrorString(cublasStatus_t status);
  static std::string GetCublasLtErrorString(cublasStatus_t status);
};

// Helper functions for CUDA-specific operations
namespace unified_linear_cuda {

// Convert activation string to cuBLASLt epilogue type
cublasLtEpilogue_t ConvertActivationToEpilogue(const std::string& activation);

// Check if tensor supports tensor cores
bool SupportsTensorCores(const DenseTensor& tensor);

// Get optimal tile size for given configuration
std::pair<int, int> GetOptimalTileSize(const UnifiedLinearDescriptor& desc);

// Check if operation can be fused
bool CanFuseBiasActivation(const std::string& activation,
                           const DenseTensor& bias);

// Memory layout optimization
bool ShouldTransposeForPerformance(const DenseTensor& tensor, bool transpose);

}  // namespace unified_linear_cuda

}  // namespace phi
