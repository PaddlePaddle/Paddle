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
#include "paddle/phi/kernels/unified_linear_utils.h"

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include "paddle/phi/kernels/funcs/blas/blas.h"
#endif

namespace phi {

namespace cublas {

// cuBLAS-specific linear operation implementation
// This is the lowest layer that directly interfaces with cuBLAS library
// All flag, math_mode parsing and transformation are encapsulated here
// No global side effects are produced

template <typename T>
class CublasLinear {
 public:
  explicit CublasLinear(const phi::DeviceContext& dev_ctx);
  ~CublasLinear() = default;

  // Disable copy constructor and assignment operator
  CublasLinear(const CublasLinear&) = delete;
  CublasLinear& operator=(const CublasLinear&) = delete;

  // Dot product operation
  void DotProduct(const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  T alpha,
                  T beta,
                  DenseTensor* out);

  // Matrix-vector multiplication
  void MatrixVector(const DenseTensor& A,
                    const DenseTensor& B,
                    const DenseTensor& C,
                    bool trans_A,
                    bool trans_B,
                    T alpha,
                    T beta,
                    DenseTensor* out);

  // Matrix-matrix multiplication
  void MatrixMatrix(const DenseTensor& A,
                    const DenseTensor& B,
                    const DenseTensor& C,
                    bool trans_A,
                    bool trans_B,
                    T alpha,
                    T beta,
                    DenseTensor* out);

  // Batched matrix-matrix multiplication
  void BatchedMatrixMatrix(const DenseTensor& A,
                           const DenseTensor& B,
                           const DenseTensor& C,
                           bool trans_A,
                           bool trans_B,
                           T alpha,
                           T beta,
                           DenseTensor* out);

 private:
  const phi::DeviceContext& dev_ctx_;
  cublasHandle_t cublas_handle_;

  // Helper functions for cuBLAS operation types
  cublasOperation_t GetCublasOperationType(bool transpose);

  // Helper functions for cuBLAS data types
  cudaDataType GetCublasDataType();

  // Helper functions for cuBLAS compute types
  cublasComputeType_t GetCublasComputeType();

  // Helper functions for cuBLAS GEMM
  void CublasGemm(cublasOperation_t trans_a,
                  cublasOperation_t trans_b,
                  int m,
                  int n,
                  int k,
                  const T* alpha,
                  const T* A,
                  int lda,
                  const T* B,
                  int ldb,
                  const T* beta,
                  T* C,
                  int ldc);

  // Helper functions for cuBLAS GEMV
  void CublasGemv(cublasOperation_t trans_a,
                  int m,
                  int n,
                  const T* alpha,
                  const T* A,
                  int lda,
                  const T* x,
                  int incx,
                  const T* beta,
                  T* y,
                  int incy);

  // Helper functions for cuBLAS DOT
  void CublasDot(int n, const T* x, int incx, const T* y, int incy, T* result);

  // Helper functions for cuBLAS batched GEMM
  void CublasBatchedGemm(cublasOperation_t trans_a,
                         cublasOperation_t trans_b,
                         int m,
                         int n,
                         int k,
                         const T* alpha,
                         const T* Aarray,
                         int lda,
                         const T* Barray,
                         int ldb,
                         const T* beta,
                         T* Carray,
                         int ldc,
                         int batch_count);

  // Helper functions for error handling
  void CheckCublasStatus(cublasStatus_t status, const std::string& operation);
};

// Hardware-specific functions for cuBLAS
template <typename T>
void DotProduct(const phi::DeviceContext& dev_ctx,
                const DenseTensor& A,
                const DenseTensor& B,
                const DenseTensor& C,
                T alpha,
                T beta,
                DenseTensor* out);

template <typename T>
void MatrixVector(const phi::DeviceContext& dev_ctx,
                  const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  bool trans_A,
                  bool trans_B,
                  T alpha,
                  T beta,
                  DenseTensor* out);

template <typename T>
void MatrixMatrix(const phi::DeviceContext& dev_ctx,
                  const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  bool trans_A,
                  bool trans_B,
                  T alpha,
                  T beta,
                  DenseTensor* out);

template <typename T>
void BatchedMatrixMatrix(const phi::DeviceContext& dev_ctx,
                         const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         bool trans_A,
                         bool trans_B,
                         T alpha,
                         T beta,
                         DenseTensor* out);

}  // namespace cublas
}  // namespace phi
