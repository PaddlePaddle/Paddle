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

#include "paddle/phi/kernels/gpu/cublas_linear.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include "paddle/phi/backends/dynload/cublas.h"
#endif

namespace phi {

namespace cublas {

// Implementation of CublasLinear class
template <typename T>
CublasLinear<T>::CublasLinear(const phi::DeviceContext& dev_ctx)
    : dev_ctx_(dev_ctx), cublas_handle_(nullptr) {
  auto* gpu_ctx = dynamic_cast<const phi::GPUContext*>(&dev_ctx_);
  PADDLE_ENFORCE_NOT_NULL(
      gpu_ctx,
      phi::errors::InvalidArgument("GPU context is required for CublasLinear"));

  cublas_handle_ = gpu_ctx->cublas_handle();
}

template <typename T>
cublasOperation_t CublasLinear<T>::GetCublasOperationType(bool transpose) {
  return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}

template <typename T>
cudaDataType CublasLinear<T>::GetCublasDataType() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    return CUDA_R_16F;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    return CUDA_R_16BF;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported data type for cuBLAS operations"));
  }
}

template <typename T>
cublasComputeType_t CublasLinear<T>::GetCublasComputeType() {
  if (std::is_same<T, float>::value) {
    return CUBLAS_COMPUTE_32F;
  } else if (std::is_same<T, double>::value) {
    return CUBLAS_COMPUTE_64F;
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    return CUBLAS_COMPUTE_32F;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    return CUBLAS_COMPUTE_32F;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported data type for cuBLAS operations"));
  }
}

template <typename T>
void CublasLinear<T>::CublasGemm(cublasOperation_t trans_a,
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
                                 int ldc) {
  cublasStatus_t status;

  if (std::is_same<T, float>::value) {
    status = phi::dynload::cublasSgemm(cublas_handle_,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       B,
                                       ldb,
                                       beta,
                                       C,
                                       ldc);
  } else if (std::is_same<T, double>::value) {
    status = phi::dynload::cublasDgemm(cublas_handle_,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       B,
                                       ldb,
                                       beta,
                                       C,
                                       ldc);
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    status = phi::dynload::cublasHgemm(cublas_handle_,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       B,
                                       ldb,
                                       beta,
                                       C,
                                       ldc);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unsupported data type for cuBLAS GEMM"));
  }

  CheckCublasStatus(status, "GEMM");
}

template <typename T>
void CublasLinear<T>::CublasGemv(cublasOperation_t trans_a,
                                 int m,
                                 int n,
                                 const T* alpha,
                                 const T* A,
                                 int lda,
                                 const T* x,
                                 int incx,
                                 const T* beta,
                                 T* y,
                                 int incy) {
  cublasStatus_t status;

  if (std::is_same<T, float>::value) {
    status = phi::dynload::cublasSgemv(
        cublas_handle_, trans_a, m, n, alpha, A, lda, x, incx, beta, y, incy);
  } else if (std::is_same<T, double>::value) {
    status = phi::dynload::cublasDgemv(
        cublas_handle_, trans_a, m, n, alpha, A, lda, x, incx, beta, y, incy);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unsupported data type for cuBLAS GEMV"));
  }

  CheckCublasStatus(status, "GEMV");
}

template <typename T>
void CublasLinear<T>::CublasDot(
    int n, const T* x, int incx, const T* y, int incy, T* result) {
  cublasStatus_t status;

  if (std::is_same<T, float>::value) {
    status =
        phi::dynload::cublasSdot(cublas_handle_, n, x, incx, y, incy, result);
  } else if (std::is_same<T, double>::value) {
    status =
        phi::dynload::cublasDdot(cublas_handle_, n, x, incx, y, incy, result);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unsupported data type for cuBLAS DOT"));
  }

  CheckCublasStatus(status, "DOT");
}

template <typename T>
void CublasLinear<T>::CublasBatchedGemm(cublasOperation_t trans_a,
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
                                        int batch_count) {
  cublasStatus_t status;

  if (std::is_same<T, float>::value) {
    status = phi::dynload::cublasSgemmBatched(cublas_handle_,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              Aarray,
                                              lda,
                                              Barray,
                                              ldb,
                                              beta,
                                              Carray,
                                              ldc,
                                              batch_count);
  } else if (std::is_same<T, double>::value) {
    status = phi::dynload::cublasDgemmBatched(cublas_handle_,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              Aarray,
                                              lda,
                                              Barray,
                                              ldb,
                                              beta,
                                              Carray,
                                              ldc,
                                              batch_count);
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    status = phi::dynload::cublasHgemmBatched(cublas_handle_,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              Aarray,
                                              lda,
                                              Barray,
                                              ldb,
                                              beta,
                                              Carray,
                                              ldc,
                                              batch_count);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported data type for cuBLAS Batched GEMM"));
  }

  CheckCublasStatus(status, "Batched GEMM");
}

template <typename T>
void CublasLinear<T>::CheckCublasStatus(cublasStatus_t status,
                                        const std::string& operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string error_msg;
    switch (status) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        error_msg = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        error_msg = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        error_msg = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        error_msg = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        error_msg = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        error_msg = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        error_msg = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      default:
        error_msg = "Unknown cuBLAS error";
        break;
    }

    PADDLE_THROW(phi::errors::External("cuBLAS operation failed: " + operation +
                                       ", error: " + error_msg));
  }
}

template <typename T>
void CublasLinear<T>::DotProduct(const DenseTensor& A,
                                 const DenseTensor& B,
                                 const DenseTensor& C,
                                 T alpha,
                                 T beta,
                                 DenseTensor* out) {
  VLOG(4) << "CublasLinear::DotProduct: Executing dot product";

  // Get tensor data
  const T* A_data = A.data<T>();
  const T* B_data = B.data<T>();
  const T* C_data = C.data<T>();
  T* out_data = out->data<T>();

  // Get tensor dimensions
  int n = A.numel();

  // Compute alpha * dot(A, B) + beta * C
  T dot_result;
  CublasDot(n, A_data, 1, B_data, 1, &dot_result);

  // Compute final result: alpha * dot_result + beta * C[0]
  *out_data = alpha * dot_result + beta * C_data[0];
}

template <typename T>
void CublasLinear<T>::MatrixVector(const DenseTensor& A,
                                   const DenseTensor& B,
                                   const DenseTensor& C,
                                   bool trans_A,
                                   bool trans_B,
                                   T alpha,
                                   T beta,
                                   DenseTensor* out) {
  VLOG(4)
      << "CublasLinear::MatrixVector: Executing matrix-vector multiplication";

  // Get tensor data
  const T* A_data = A.data<T>();
  const T* B_data = B.data<T>();
  const T* C_data = C.data<T>();
  T* out_data = out->data<T>();

  // Get tensor dimensions
  auto A_dims = A.dims();
  auto B_dims = B.dims();

  int m = trans_A ? A_dims[1] : A_dims[0];
  int n = trans_A ? A_dims[0] : A_dims[1];

  // Ensure B is a vector
  PADDLE_ENFORCE_EQ(
      B_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "B must be a 1D tensor for matrix-vector multiplication"));

  // Ensure dimensions are compatible
  PADDLE_ENFORCE_EQ(
      n,
      B_dims[0],
      phi::errors::InvalidArgument("Dimensions of A and B are not compatible "
                                   "for matrix-vector multiplication"));

  // Compute alpha * A * B + beta * C
  CublasGemv(GetCublasOperationType(trans_A),
             m,
             n,
             &alpha,
             A_data,
             A_dims[1],
             B_data,
             1,
             &beta,
             C_data,
             1,
             out_data,
             1);
}

template <typename T>
void CublasLinear<T>::MatrixMatrix(const DenseTensor& A,
                                   const DenseTensor& B,
                                   const DenseTensor& C,
                                   bool trans_A,
                                   bool trans_B,
                                   T alpha,
                                   T beta,
                                   DenseTensor* out) {
  VLOG(4)
      << "CublasLinear::MatrixMatrix: Executing matrix-matrix multiplication";

  // Get tensor data
  const T* A_data = A.data<T>();
  const T* B_data = B.data<T>();
  const T* C_data = C.data<T>();
  T* out_data = out->data<T>();

  // Get tensor dimensions
  auto A_dims = A.dims();
  auto B_dims = B.dims();

  int m = trans_A ? A_dims[1] : A_dims[0];
  int k = trans_A ? A_dims[0] : A_dims[1];
  int k_b = trans_B ? B_dims[1] : B_dims[0];
  int n = trans_B ? B_dims[0] : B_dims[1];

  // Ensure dimensions are compatible
  PADDLE_ENFORCE_EQ(
      k,
      k_b,
      phi::errors::InvalidArgument("Dimensions of A and B are not compatible "
                                   "for matrix-matrix multiplication"));

  // Compute alpha * A * B + beta * C
  CublasGemm(GetCublasOperationType(trans_A),
             GetCublasOperationType(trans_B),
             m,
             n,
             k,
             &alpha,
             A_data,
             A_dims[1],
             B_data,
             B_dims[1],
             &beta,
             C_data,
             out->dims()[1],
             out_data,
             out->dims()[1]);
}

template <typename T>
void CublasLinear<T>::BatchedMatrixMatrix(const DenseTensor& A,
                                          const DenseTensor& B,
                                          const DenseTensor& C,
                                          bool trans_A,
                                          bool trans_B,
                                          T alpha,
                                          T beta,
                                          DenseTensor* out) {
  VLOG(4) << "CublasLinear::BatchedMatrixMatrix: Executing batched "
             "matrix-matrix multiplication";

  // Get tensor data
  const T* A_data = A.data<T>();
  const T* B_data = B.data<T>();
  const T* C_data = C.data<T>();
  T* out_data = out->data<T>();

  // Get tensor dimensions
  auto A_dims = A.dims();
  auto B_dims = B.dims();

  int batch_size = A_dims[0];
  int m = trans_A ? A_dims[2] : A_dims[1];
  int k = trans_A ? A_dims[1] : A_dims[2];
  int k_b = trans_B ? B_dims[2] : B_dims[1];
  int n = trans_B ? B_dims[1] : B_dims[2];

  // Ensure batch sizes are compatible
  PADDLE_ENFORCE_EQ(
      batch_size,
      B_dims[0],
      phi::errors::InvalidArgument("Batch sizes of A and B are not compatible "
                                   "for batched matrix-matrix multiplication"));

  // Ensure dimensions are compatible
  PADDLE_ENFORCE_EQ(
      k,
      k_b,
      phi::errors::InvalidArgument("Dimensions of A and B are not compatible "
                                   "for batched matrix-matrix multiplication"));

  // Create array of pointers for batched GEMM
  const T** Aarray = nullptr;
  const T** Barray = nullptr;
  T** Carray = nullptr;

  auto* gpu_ctx = dynamic_cast<const phi::GPUContext*>(&dev_ctx_);
  PADDLE_ENFORCE_NOT_NULL(
      gpu_ctx,
      phi::errors::InvalidArgument("GPU context is required for CublasLinear"));

  // Allocate temporary memory for pointer arrays
  phi::Allocator* allocator = gpu_ctx->allocator();
  Aarray = static_cast<const T**>(allocator->Allocate(sizeof(T*) * batch_size));
  Barray = static_cast<const T**>(allocator->Allocate(sizeof(T*) * batch_size));
  Carray = static_cast<T**>(allocator->Allocate(sizeof(T*) * batch_size));

  // Set up pointer arrays
  for (int i = 0; i < batch_size; ++i) {
    Aarray[i] = A_data + i * A_dims[1] * A_dims[2];
    Barray[i] = B_data + i * B_dims[1] * B_dims[2];
    Carray[i] = out_data + i * out->dims()[1] * out->dims()[2];
  }

  // Compute alpha * A * B + beta * C for each batch
  CublasBatchedGemm(GetCublasOperationType(trans_A),
                    GetCublasOperationType(trans_b),
                    m,
                    n,
                    k,
                    &alpha,
                    Aarray,
                    A_dims[2],
                    Barray,
                    B_dims[2],
                    &beta,
                    Carray,
                    out->dims()[2],
                    batch_size);

  // Free temporary memory
  allocator->Free(Aarray);
  allocator->Free(Barray);
  allocator->Free(Carray);
}

// Implementation of hardware-specific functions
template <typename T>
void DotProduct(const phi::DeviceContext& dev_ctx,
                const DenseTensor& A,
                const DenseTensor& B,
                const DenseTensor& C,
                T alpha,
                T beta,
                DenseTensor* out) {
  VLOG(4) << "DotProduct: Executing dot product with cuBLAS";

  CublasLinear<T> impl(dev_ctx);
  impl.DotProduct(A, B, C, alpha, beta, out);
}

template <typename T>
void MatrixVector(const phi::DeviceContext& dev_ctx,
                  const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  bool trans_A,
                  bool trans_B,
                  T alpha,
                  T beta,
                  DenseTensor* out) {
  VLOG(4) << "MatrixVector: Executing matrix-vector multiplication with cuBLAS";

  CublasLinear<T> impl(dev_ctx);
  impl.MatrixVector(A, B, C, trans_A, trans_B, alpha, beta, out);
}

template <typename T>
void MatrixMatrix(const phi::DeviceContext& dev_ctx,
                  const DenseTensor& A,
                  const DenseTensor& B,
                  const DenseTensor& C,
                  bool trans_A,
                  bool trans_B,
                  T alpha,
                  T beta,
                  DenseTensor* out) {
  VLOG(4) << "MatrixMatrix: Executing matrix-matrix multiplication with cuBLAS";

  CublasLinear<T> impl(dev_ctx);
  impl.MatrixMatrix(A, B, C, trans_A, trans_B, alpha, beta, out);
}

template <typename T>
void BatchedMatrixMatrix(const phi::DeviceContext& dev_ctx,
                         const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         bool trans_A,
                         bool trans_B,
                         T alpha,
                         T beta,
                         DenseTensor* out) {
  VLOG(4) << "BatchedMatrixMatrix: Executing batched matrix-matrix "
             "multiplication with cuBLAS";

  CublasLinear<T> impl(dev_ctx);
  impl.BatchedMatrixMatrix(A, B, C, trans_A, trans_B, alpha, beta, out);
}

// Explicit instantiation for common data types
template class CublasLinear<float>;
template class CublasLinear<double>;
template class CublasLinear<phi::dtype::float16>;
template class CublasLinear<phi::dtype::bfloat16>;

template void DotProduct<float>(const phi::DeviceContext&,
                                const DenseTensor&,
                                const DenseTensor&,
                                const DenseTensor&,
                                float,
                                float,
                                DenseTensor*);
template void MatrixVector<float>(const phi::DeviceContext&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  bool,
                                  bool,
                                  float,
                                  float,
                                  DenseTensor*);
template void MatrixMatrix<float>(const phi::DeviceContext&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  bool,
                                  bool,
                                  float,
                                  float,
                                  DenseTensor*);
template void BatchedMatrixMatrix<float>(const phi::DeviceContext&,
                                         const DenseTensor&,
                                         const DenseTensor&,
                                         const DenseTensor&,
                                         bool,
                                         bool,
                                         float,
                                         float,
                                         DenseTensor*);

}  // namespace cublas
}  // namespace phi
