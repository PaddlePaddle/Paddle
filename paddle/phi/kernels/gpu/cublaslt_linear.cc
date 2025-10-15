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

#include "paddle/phi/kernels/gpu/cublaslt_linear.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cublaslt.h"

#ifdef PADDLE_WITH_CUDA
#include <cublasLt.h>
#include <cuda.h>
#include "paddle/phi/backends/dynload/cublaslt.h"
#endif

namespace phi {

namespace unified_linear {

namespace cuda {

namespace cublasLt {

// Implementation of CublasLtLinear class
template <typename T>
CublasLtLinear<T>::CublasLtLinear(const phi::DeviceContext& dev_ctx)
    : dev_ctx_(dev_ctx), cublaslt_handle_(nullptr), matmul_desc_(nullptr) {
  auto* gpu_ctx = dynamic_cast<const phi::GPUContext*>(&dev_ctx_);
  PADDLE_ENFORCE_NOT_NULL(gpu_ctx,
                          phi::errors::InvalidArgument(
                              "GPU context is required for CublasLtLinear"));

  // Initialize cuBLASLt handle
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtCreate(&cublaslt_handle_));

  // Initialize matmul descriptor
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescCreate(
      &matmul_desc_, GetCublasComputeType(), GetCublasScaleType()));
}

template <typename T>
CublasLtLinear<T>::~CublasLtLinear() {
  if (matmul_desc_ != nullptr) {
    cublasLtMatmulDescDestroy(matmul_desc_);
    matmul_desc_ = nullptr;
  }

  if (cublaslt_handle_ != nullptr) {
    cublasLtDestroy(cublaslt_handle_);
    cublaslt_handle_ = nullptr;
  }
}

template <typename T>
cublasOperation_t CublasLtLinear<T>::GetCublasOperationType(bool transpose) {
  return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}

template <typename T>
cudaDataType CublasLtLinear<T>::GetCublasDataType() {
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
        "Unsupported data type for cuBLASLt operations"));
  }
}

template <typename T>
cublasComputeType_t CublasLtLinear<T>::GetCublasComputeType() {
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
        "Unsupported data type for cuBLASLt operations"));
  }
}

template <typename T>
cublasLtScaleType_t CublasLtLinear<T>::GetCublasScaleType() {
  if (std::is_same<T, float>::value) {
    return CUBLASLT_SCALE_SCALAR;
  } else if (std::is_same<T, double>::value) {
    return CUBLASLT_SCALE_SCALAR;
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    return CUBLASLT_SCALE_SCALAR;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    return CUBLASLT_SCALE_SCALAR;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported data type for cuBLASLt operations"));
  }
}

template <typename T>
cublasLtEpilogue_t CublasLtLinear<T>::GetCublasEpilogueType(
    unified_linear::ActivationType activation) {
  switch (activation) {
    case unified_linear::ActivationType::kNone:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case unified_linear::ActivationType::kRelu:
      return CUBLASLT_EPILOGUE_RELU;
    case unified_linear::ActivationType::kGelu:
      return CUBLASLT_EPILOGUE_GELU;
    case unified_linear::ActivationType::kSigmoid:
      return CUBLASLT_EPILOGUE_SIGMOID;
    case unified_linear::ActivationType::kTanh:
      return CUBLASLT_EPILOGUE_TANH;
    default:
      return CUBLASLT_EPILOGUE_DEFAULT;
  }
}

template <typename T>
cublasLtOrder_t CublasLtLinear<T>::GetCublasOrder() {
  return CUBLASLT_ORDER_ROW_MAJOR;
}

template <typename T>
void CublasLtLinear<T>::CreateMatrixDescriptor(cublasLtMatrixLayout_t* mat_desc,
                                               int rows,
                                               int cols,
                                               int ld,
                                               cudaDataType data_type) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatrixLayoutCreate(mat_desc, data_type, rows, cols, ld));
}

template <typename T>
void CublasLtLinear<T>::CublasLtMatmul(cublasOperation_t trans_a,
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
                                       void* D_scale) {
  // Create matrix layouts
  cublasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;
  CreateMatrixDescriptor(&A_desc,
                         trans_a == CUBLAS_OP_N ? lda : lda,
                         trans_a == CUBLAS_OP_N ? lda : lda,
                         lda,
                         A_type);
  CreateMatrixDescriptor(&B_desc,
                         trans_b == CUBLAS_OP_N ? ldb : ldb,
                         trans_b == CUBLAS_OP_N ? ldb : ldb,
                         ldb,
                         B_type);
  CreateMatrixDescriptor(&C_desc, ldc, ldc, ldc, C_type);
  CreateMatrixDescriptor(&D_desc, ldd, ldd, ldd, D_type);

  // Set matmul descriptor attributes
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatmulDescSetAttribute(matmul_desc_,
                                     CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                                     &compute_type,
                                     sizeof(compute_type)));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatmulDescSetAttribute(matmul_desc_,
                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &epilogue,
                                     sizeof(epilogue)));

  // Set bias if provided
  if (bias != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescSetAttribute(
        matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }

  // Set scales if provided
  if (A_scale != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cublasLtMatmulDescSetAttribute(matmul_desc_,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &A_scale,
                                       sizeof(A_scale)));
  }

  if (B_scale != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cublasLtMatmulDescSetAttribute(matmul_desc_,
                                       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &B_scale,
                                       sizeof(B_scale)));
  }

  if (C_scale != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cublasLtMatmulDescSetAttribute(matmul_desc_,
                                       CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                       &C_scale,
                                       sizeof(C_scale)));
  }

  if (D_scale != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cublasLtMatmulDescSetAttribute(matmul_desc_,
                                       CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                       &D_scale,
                                       sizeof(D_scale)));
  }

  // Execute matmul
  cublasStatus_t status = phi::dynload::cublasLtMatmul(cublaslt_handle_,
                                                       matmul_desc_,
                                                       alpha,
                                                       A,
                                                       A_desc,
                                                       B,
                                                       B_desc,
                                                       beta,
                                                       C,
                                                       C_desc,
                                                       D,
                                                       D_desc,
                                                       &algo_,
                                                       nullptr,
                                                       0,
                                                       nullptr);

  // Clean up matrix layouts
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(C_desc);
  cublasLtMatrixLayoutDestroy(D_desc);

  CheckCublasLtStatus(status, "Matmul");
}

template <typename T>
void CublasLtLinear<T>::CheckCublasLtStatus(cublasStatus_t status,
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
        error_msg = "Unknown cuBLASLt error";
        break;
    }

    PADDLE_THROW(phi::errors::External(
        "cuBLASLt operation failed: " + operation + ", error: " + error_msg));
  }
}

template <typename T>
void CublasLtLinear<T>::FindBestAlgorithm(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    const unified_linear_utils::OperationConfig& config) {
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

  // Create matrix layouts
  cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CreateMatrixDescriptor(&A_desc, m, k, A_dims[1], GetCublasDataType());
  CreateMatrixDescriptor(&B_desc, k, n, B_dims[1], GetCublasDataType());
  CreateMatrixDescriptor(&C_desc, m, n, C.dims()[1], GetCublasDataType());

  // Set matmul descriptor attributes
  cublasOperation_t trans_a = GetCublasOperationType(trans_A);
  cublasOperation_t trans_b = GetCublasOperationType(trans_B);
  cublasComputeType_t compute_type = GetCublasComputeType();

  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatmulDescSetAttribute(matmul_desc_,
                                     CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                                     &compute_type,
                                     sizeof(compute_type)));

  // Find best algorithm
  int algo_count = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle_,
                                                            matmul_desc_,
                                                            A_desc,
                                                            B_desc,
                                                            C_desc,
                                                            C_desc,
                                                            1,
                                                            &algo_,
                                                            &algo_count));

  // Clean up matrix layouts
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(C_desc);

  // If no algorithm found, use default
  if (algo_count == 0) {
    algo_ = 0;
  }
}

template <typename T>
int CublasLtLinear<T>::DetermineOptimalAlgorithm(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    const unified_linear_utils::OperationConfig& config) {
  FindBestAlgorithm(A, B, C, trans_A, trans_B, config);
  return algo_;
}

template <typename T>
void CublasLtLinear<T>::MatrixMatrix(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    T alpha,
    T beta,
    DenseTensor* out,
    const unified_linear_utils::OperationConfig& config) {
  VLOG(4) << "CublasLtLinear::MatrixMatrix: Executing matrix-matrix "
             "multiplication with cuBLASLt";

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

  // Find best algorithm if not already determined
  if (config.auto_tune) {
    FindBestAlgorithm(A, B, C, trans_A, trans_B, config);
  }

  // Execute matmul
  CublasLtMatmul(GetCublasOperationType(trans_A),
                 GetCublasOperationType(trans_B),
                 &alpha,
                 A_data,
                 GetCublasDataType(),
                 A_dims[1],
                 B_data,
                 GetCublasDataType(),
                 B_dims[1],
                 &beta,
                 C_data,
                 GetCublasDataType(),
                 out->dims()[1],
                 out_data,
                 GetCublasDataType(),
                 out->dims()[1],
                 GetCublasComputeType(),
                 CUBLASLT_EPILOGUE_DEFAULT,
                 nullptr,
                 nullptr,
                 nullptr,
                 nullptr,
                 nullptr);
}

template <typename T>
void CublasLtLinear<T>::BatchedMatrixMatrix(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    T alpha,
    T beta,
    DenseTensor* out,
    const unified_linear_utils::OperationConfig& config) {
  VLOG(4) << "CublasLtLinear::BatchedMatrixMatrix: Executing batched "
             "matrix-matrix multiplication with cuBLASLt";

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

  // Find best algorithm if not already determined
  if (config.auto_tune) {
    // Use first batch for algorithm selection
    DenseTensor A_batch = A.Slice(0, 1);
    DenseTensor B_batch = B.Slice(0, 1);
    DenseTensor C_batch = C.Slice(0, 1);
    FindBestAlgorithm(A_batch, B_batch, C_batch, trans_A, trans_B, config);
  }

  // Execute batched matmul
  for (int i = 0; i < batch_size; ++i) {
    DenseTensor A_batch = A.Slice(i, i + 1);
    DenseTensor B_batch = B.Slice(i, i + 1);
    DenseTensor C_batch = C.Slice(i, i + 1);
    DenseTensor out_batch = out->Slice(i, i + 1);

    CublasLtMatmul(GetCublasOperationType(trans_A),
                   GetCublasOperationType(trans_B),
                   &alpha,
                   A_batch.data<T>(),
                   GetCublasDataType(),
                   A_dims[2],
                   B_batch.data<T>(),
                   GetCublasDataType(),
                   B_dims[2],
                   &beta,
                   C_batch.data<T>(),
                   GetCublasDataType(),
                   out->dims()[2],
                   out_batch.data<T>(),
                   GetCublasDataType(),
                   out->dims()[2],
                   GetCublasComputeType(),
                   CUBLASLT_EPILOGUE_DEFAULT,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr);
  }
}

template <typename T>
void CublasLtLinear<T>::Linear(
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
    unified_linear::ActivationType activation) {
  VLOG(4) << "CublasLtLinear::Linear: Executing linear transformation with "
             "cuBLASLt";

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
                                   "for linear transformation"));

  // Find best algorithm if not already determined
  if (config.auto_tune) {
    FindBestAlgorithm(A, B, C, trans_A, trans_B, config);
  }

  // Get epilogue type
  cublasLtEpilogue_t epilogue = GetCublasEpilogueType(activation);

  // Get bias data if provided
  const T* bias_data = nullptr;
  if (bias.is_initialized()) {
    bias_data = bias.get().data<T>();
  }

  // Execute matmul with fused bias and activation
  CublasLtMatmul(GetCublasOperationType(trans_A),
                 GetCublasOperationType(trans_B),
                 &alpha,
                 A_data,
                 GetCublasDataType(),
                 A_dims[1],
                 B_data,
                 GetCublasDataType(),
                 B_dims[1],
                 &beta,
                 C_data,
                 GetCublasDataType(),
                 out->dims()[1],
                 out_data,
                 GetCublasDataType(),
                 out->dims()[1],
                 GetCublasComputeType(),
                 epilogue,
                 bias_data,
                 nullptr,
                 nullptr,
                 nullptr,
                 nullptr);
}

template <typename T>
void CublasLtLinear<T>::ComputeOutputScale(
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    const paddle::optional<DenseTensor>& D_scale,
    DenseTensor* out_D_scale) {
  VLOG(4) << "CublasLtLinear::ComputeOutputScale: Computing output scale with "
             "cuBLASLt";

  // Get tensor data
  const T* A_data = A.data<T>();
  const T* B_data = B.data<T>();
  const T* C_data = C.data<T>();
  float* out_D_scale_data = out_D_scale->data<float>();

  // Get tensor dimensions
  auto A_dims = A.dims();
  auto B_dims = B.dims();

  int m = A_dims[0];
  int k = A_dims[1];
  int n = B_dims[1];

  // Create matrix layouts
  cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CreateMatrixDescriptor(&A_desc, m, k, A_dims[1], GetCublasDataType());
  CreateMatrixDescriptor(&B_desc, k, n, B_dims[1], GetCublasDataType());
  CreateMatrixDescriptor(&C_desc, m, n, out_D_scale->dims()[1], CUDA_R_32F);

  // Set matmul descriptor attributes
  cublasComputeType_t compute_type = GetCublasComputeType();

  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatmulDescSetAttribute(matmul_desc_,
                                     CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                                     &compute_type,
                                     sizeof(compute_type)));

  // Set epilogue to compute scale
  PADDLE_ENFORCE_GPU_SUCCESS(
      cublasLtMatmulDescSetAttribute(matmul_desc_,
                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &CUBLASLT_EPILOGUE_DEFAULT,
                                     sizeof(CUBLASLT_EPILOGUE_DEFAULT)));

  // Get D scale if provided
  const float* D_scale_data = nullptr;
  if (D_scale.is_initialized()) {
    D_scale_data = D_scale.get().data<float>();
  }

  // Execute matmul to compute scale
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t status = phi::dynload::cublasLtMatmul(cublaslt_handle_,
                                                       matmul_desc_,
                                                       &alpha,
                                                       A_data,
                                                       A_desc,
                                                       B_data,
                                                       B_desc,
                                                       &beta,
                                                       C_data,
                                                       C_desc,
                                                       out_D_scale_data,
                                                       C_desc,
                                                       &algo_,
                                                       nullptr,
                                                       0,
                                                       nullptr);

  // Clean up matrix layouts
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(C_desc);

  CheckCublasLtStatus(status, "ComputeOutputScale");
}

// Implementation of hardware-specific functions
template <typename T>
int DetermineOptimalAlgorithm(
    const phi::DeviceContext& dev_ctx,
    const DenseTensor& A,
    const DenseTensor& B,
    const DenseTensor& C,
    bool trans_A,
    bool trans_B,
    const unified_linear_utils::OperationConfig& config) {
  VLOG(4)
      << "DetermineOptimalAlgorithm: Finding optimal algorithm with cuBLASLt";

  CublasLtLinear<T> impl(dev_ctx);
  return impl.DetermineOptimalAlgorithm(A, B, C, trans_A, trans_B, config);
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
                  DenseTensor* out,
                  const unified_linear_utils::OperationConfig& config) {
  VLOG(4)
      << "MatrixMatrix: Executing matrix-matrix multiplication with cuBLASLt";

  CublasLtLinear<T> impl(dev_ctx);
  impl.MatrixMatrix(A, B, C, trans_A, trans_B, alpha, beta, out, config);
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
                         DenseTensor* out,
                         const unified_linear_utils::OperationConfig& config) {
  VLOG(4) << "BatchedMatrixMatrix: Executing batched matrix-matrix "
             "multiplication with cuBLASLt";

  CublasLtLinear<T> impl(dev_ctx);
  impl.BatchedMatrixMatrix(A, B, C, trans_A, trans_B, alpha, beta, out, config);
}

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
            unified_linear::ActivationType activation) {
  VLOG(4) << "Linear: Executing linear transformation with cuBLASLt";

  CublasLtLinear<T> impl(dev_ctx);
  impl.Linear(
      A, B, C, bias, trans_A, trans_B, alpha, beta, out, config, activation);
}

template <typename T>
void ComputeOutputScale(const phi::DeviceContext& dev_ctx,
                        const DenseTensor& A,
                        const DenseTensor& B,
                        const DenseTensor& C,
                        const paddle::optional<DenseTensor>& D_scale,
                        DenseTensor* out_D_scale) {
  VLOG(4) << "ComputeOutputScale: Computing output scale with cuBLASLt";

  CublasLtLinear<T> impl(dev_ctx);
  impl.ComputeOutputScale(A, B, C, D_scale, out_D_scale);
}

// Explicit instantiation for common data types
template class CublasLtLinear<float>;
template class CublasLtLinear<double>;
template class CublasLtLinear<phi::dtype::float16>;
template class CublasLtLinear<phi::dtype::bfloat16>;

template int DetermineOptimalAlgorithm<float>(
    const phi::DeviceContext&,
    const DenseTensor&,
    const DenseTensor&,
    const DenseTensor&,
    bool,
    bool,
    const unified_linear_utils::OperationConfig&);
template void MatrixMatrix<float>(const phi::DeviceContext&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  const DenseTensor&,
                                  bool,
                                  bool,
                                  float,
                                  float,
                                  DenseTensor*,
                                  const unified_linear_utils::OperationConfig&);
template void BatchedMatrixMatrix<float>(
    const phi::DeviceContext&,
    const DenseTensor&,
    const DenseTensor&,
    const DenseTensor&,
    bool,
    bool,
    float,
    float,
    DenseTensor*,
    const unified_linear_utils::OperationConfig&);
template void Linear<float>(const phi::DeviceContext&,
                            const DenseTensor&,
                            const DenseTensor&,
                            const DenseTensor&,
                            const paddle::optional<DenseTensor>&,
                            bool,
                            bool,
                            float,
                            float,
                            DenseTensor*,
                            const unified_linear_utils::OperationConfig&,
                            unified_linear::ActivationType);
template void ComputeOutputScale<float>(const phi::DeviceContext&,
                                        const DenseTensor&,
                                        const DenseTensor&,
                                        const DenseTensor&,
                                        const paddle::optional<DenseTensor>&,
                                        DenseTensor*);

}  // namespace cublasLt
}  // namespace cuda
}  // namespace unified_linear
}  // namespace phi
