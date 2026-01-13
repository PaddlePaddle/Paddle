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
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/unified_linear_kernel.h"
#include "paddle/phi/kernels/unified_linear_utils.h"

#ifdef PADDLE_WITH_CUDA
#include <cublasLt.h>
#include <cuda.h>
#include "paddle/phi/kernels/funcs/cublaslt.h"
#endif

namespace phi {

namespace unified_linear {

namespace cuda {

namespace cublasLt {

// cuBLASLt-specific linear operation implementation
// This is the lowest layer that directly interfaces with cuBLASLt library
// All flag, math_mode parsing and transformation are encapsulated here
// No global side effects are produced

template <typename T>
class CublasLtLinear {
 public:
  explicit CublasLtLinear(const phi::DeviceContext& dev_ctx);
  ~CublasLtLinear();

  // Disable copy constructor and assignment operator
  CublasLtLinear(const CublasLtLinear&) = delete;
  CublasLtLinear& operator=(const CublasLtLinear&) = delete;

  // Determine optimal algorithm for matrix multiplication
  int DetermineOptimalAlgorithm(
      const DenseTensor& A,
      const DenseTensor& B,
      const DenseTensor& C,
      bool trans_A,
      bool trans_B,
      const unified_linear_utils::OperationConfig& config);

  // Matrix-matrix multiplication
  void MatrixMatrix(const DenseTensor& A,
                    const DenseTensor& B,
                    const DenseTensor& C,
                    bool trans_A,
                    bool trans_B,
                    T alpha,
                    T beta,
                    DenseTensor* out,
                    const unified_linear_utils::OperationConfig& config);

  // Batched matrix-matrix multiplication
  void BatchedMatrixMatrix(const DenseTensor& A,
                           const DenseTensor& B,
                           const DenseTensor& C,
                           bool trans_A,
                           bool trans_B,
                           T alpha,
                           T beta,
                           DenseTensor* out,
                           const unified_linear_utils::OperationConfig& config);

  // Linear transformation with fused bias and activation
  void Linear(const DenseTensor& A,
              const DenseTensor& B,
              const DenseTensor& C,
              const paddle::optional<DenseTensor>& bias,
              bool trans_A,
              bool trans_B,
              T alpha,
              T beta,
              DenseTensor* out,
              const unified_linear_utils::OperationConfig& config,
              unified_linear::ActivationType activation);

  // Compute output scale for scaled tensors
  void ComputeOutputScale(const DenseTensor& A,
                          const DenseTensor& B,
                          const DenseTensor& C,
                          const paddle::optional<DenseTensor>& D_scale,
                          DenseTensor* out_D_scale);

 private:
  const phi::DeviceContext& dev_ctx_;
  cublasLtHandle_t cublaslt_handle_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatmulAlgo_t algo_;

  // Helper functions for cuBLASLt operation types
  cublasOperation_t GetCublasOperationType(bool transpose);

  // Helper functions for cuBLASLt data types
  cudaDataType GetCublasDataType();

  // Helper functions for cuBLASLt compute types
  cublasComputeType_t GetCublasComputeType();

  // Helper functions for cuBLASLt scale types
  cublasLtScaleType_t GetCublasScaleType();

  // Helper functions for cuBLASLt epilogue types
  cublasLtEpilogue_t GetCublasEpilogueType(
      unified_linear::ActivationType activation);

  // Helper functions for cuBLASLt matrix layouts
  cublasLtOrder_t GetCublasOrder();

  // Helper functions for cuBLASLt matrix descriptors
  void CreateMatrixDescriptor(cublasLtMatrixLayout_t* mat_desc,
                              int rows,
                              int cols,
                              int ld,
                              cudaDataType data_type);

  // Helper functions for cuBLASLt matrix multiplication
  void CublasLtMatmul(cublasOperation_t trans_a,
                      cublasOperation_t trans_b,
                      const void* alpha,
                      const void* A,
                      cudaDataType A_type,
                      int lda,
                      const void* B,
                      cudaDataType B_type,
                      int ldb,
                      const void* beta,
                      const void* C,
                      cudaDataType C_type,
                      int ldc,
                      void* D,
                      cudaDataType D_type,
                      int ldd,
                      cublasComputeType_t compute_type,
                      cublasLtEpilogue_t epilogue,
                      const void* bias,
                      const void* A_scale,
                      const void* B_scale,
                      const void* C_scale,
                      void* D_scale);

  // Helper functions for error handling
  void CheckCublasLtStatus(cublasStatus_t status, const std::string& operation);

  // Helper functions for algorithm selection
  void FindBestAlgorithm(const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         bool trans_A,
                         bool trans_B,
                         const unified_linear_utils::OperationConfig& config);
};

// Hardware-specific functions for cuBLASLt
template <typename T>
int DetermineOptimalAlgorithm(
    const phi::DeviceContext& dev_ctx,
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    const unified_linear_utils::OperationConfig& config);

template <typename T>
void MatrixMatrix(const phi::DeviceContext& dev_ctx,
                  const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  bool trans_A,
                  bool trans_B,
                  T alpha,
                  T beta,
                  DenseTensor* out,
                  const unified_linear_utils::OperationConfig& config);

template <typename T>
void BatchedMatrixMatrix(const phi::DeviceContext& dev_ctx,
                         const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         bool trans_A,
                         bool trans_B,
                         T alpha,
                         T beta,
                         DenseTensor* out,
                         const unified_linear_utils::OperationConfig& config);

template <typename T>
void Linear(const phi::DeviceContext& dev_ctx,
            const DenseTensor& A,
            const DenseTensor& B,
            const DenseTensor& C,
            const paddle::optional<DenseTensor>& bias,
            bool trans_A,
            bool trans_B,
            T alpha,
            T beta,
            DenseTensor* out,
            const unified_linear_utils::OperationConfig& config,
            unified_linear::ActivationType activation);

template <typename T>
void ComputeOutputScale(const phi::DeviceContext& dev_ctx,
                        const DenseTensor& A,
                        const DenseTensor& B,
                        const DenseTensor& C,
                        const paddle::optional<DenseTensor>& D_scale,
                        DenseTensor* out_D_scale);

}  // namespace cublasLt
}  // namespace cuda
}  // namespace unified_linear
}  // namespace phi
