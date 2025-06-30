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

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <cstdint>

#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/common/memory_utils.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_gemm_search.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

namespace {

// Helper function to check if dtype is FP8
bool IsFp8Dtype(phi::DataType dtype) {
  return dtype == phi::DataType::FLOAT8_E4M3FN ||
         dtype == phi::DataType::FLOAT8_E5M2;
}

// Convert phi::DataType to cudaDataType_t
cudaDataType_t ScalarTypeToCudaDataType(phi::DataType dtype) {
  switch (dtype) {
    case phi::DataType::FLOAT8_E4M3FN:
      return CUDA_R_8F_E4M3;
    case phi::DataType::FLOAT8_E5M2:
      return CUDA_R_8F_E5M2;
    case phi::DataType::BFLOAT16:
      return CUDA_R_16BF;
    case phi::DataType::FLOAT32:
      return CUDA_R_32F;
    case phi::DataType::FLOAT16:
      return CUDA_R_16F;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Unsupported data type"));
  }
}

// cuBLAS error checking macro
#define PADDLE_CUDABLAS_CHECK(func)                                    \
  do {                                                                 \
    cublasStatus_t status = func;                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
      PADDLE_THROW(phi::errors::External("cuBLAS error: %d", status)); \
    }                                                                  \
  } while (0)

template <typename Context>
void cublas_gemm_blockwise_impl(const Context& dev_ctx,
                                const DenseTensor& A,
                                const DenseTensor& A_decode_scale,
                                const DenseTensor& B,
                                const DenseTensor& B_decode_scale,
                                DenseTensor* D,
                                const DenseTensor& bias,
                                DenseTensor* pre_gelu_out,
                                bool transa,
                                bool transb,
                                bool grad,
                                DenseTensor* workspace,
                                bool accumulate,
                                bool use_split_accumulator,
                                int math_sm_count,
                                bool is_A_1d_scaled,
                                bool is_B_1d_scaled,
                                cudaStream_t stream) {
  // Sanity checks
  PADDLE_ENFORCE_EQ(
      transa,
      true,
      phi::errors::InvalidArgument("Only transa == true is supported"));
  PADDLE_ENFORCE_EQ(
      transb,
      false,
      phi::errors::InvalidArgument("Only transb == false is supported"));
  PADDLE_ENFORCE_EQ(
      A.place().GetType(),
      phi::AllocationType::GPU,
      phi::errors::InvalidArgument("Input tensor A must be on CUDA device."));
  PADDLE_ENFORCE_EQ(
      B.place().GetType(),
      phi::AllocationType::GPU,
      phi::errors::InvalidArgument("Input tensor B must be on CUDA device."));
  PADDLE_ENFORCE_EQ(
      D->place().GetType(),
      phi::AllocationType::GPU,
      phi::errors::InvalidArgument("Output tensor D must be on CUDA device."));
  PADDLE_ENFORCE_EQ(IsFp8Dtype(A.dtype()),
                    true,
                    phi::errors::InvalidArgument("A must be FP8"));
  PADDLE_ENFORCE_EQ(IsFp8Dtype(B.dtype()),
                    true,
                    phi::errors::InvalidArgument("B must be FP8"));
  PADDLE_ENFORCE_EQ(
      D->dtype() == phi::DataType::BFLOAT16 ||
          D->dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("D must be BFloat16 or float"));
  PADDLE_ENFORCE_EQ(
      A_decode_scale.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("A_decode_scale must be float"));
  PADDLE_ENFORCE_EQ(
      B_decode_scale.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("B_decode_scale must be float"));
  PADDLE_ENFORCE_EQ(
      A.dims().size() == 2, true, phi::errors::InvalidArgument("A must be 2D"));
  PADDLE_ENFORCE_EQ(
      B.dims().size() == 2, true, phi::errors::InvalidArgument("B must be 2D"));
  PADDLE_ENFORCE_EQ(D->dims().size() == 2,
                    true,
                    phi::errors::InvalidArgument("D must be 2D"));

  const int m = transa ? A.dims()[0] : A.dims()[1];
  const int k = transa ? A.dims()[1] : A.dims()[0];
  const int n = transb ? B.dims()[1] : B.dims()[0];

  int lda = k, ldb = k, ldc = m, ldd = m;
  float alpha = 1.0, beta = accumulate ? 1.0 : 0.0;

  cublasLtHandle_t ltHandle = dev_ctx.cublaslt_handle();
  // Create operation descriptor
  cublasLtMatmulDesc_t operationDesc = nullptr;
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescCreate(
      &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

#if CUBLAS_VERSION >= 120805 && CUDA_VERSION >= 12080
  // Setup scaling for A and B
  cublasLtMatmulMatrixScale_t A_scale_mode, B_scale_mode;
  // Note: in cuBLAS term, tensor name A and B are swapped.
  if (is_B_1d_scaled && is_A_1d_scaled) {
    A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
    B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  } else if (!is_B_1d_scaled && is_A_1d_scaled) {
    // So this corresponds to 2Dx1D GEMM.
    A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
    B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  } else if (is_B_1d_scaled && !is_A_1d_scaled) {
    // So this corresponds to 1Dx2D GEMM.
    A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
    B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("2Dx2D scaling is not supported"));
  }
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &A_scale_mode,
      sizeof(A_scale_mode)));
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &B_scale_mode,
      sizeof(B_scale_mode)));
#else
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Sub-channel FP8 GEMM requires CUDA 12.8 and cuBLAS 12.8.5 or later."));
#endif

  // setup transa and transb
  const cublasOperation_t transa_type = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb_type = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  PADDLE_CUDABLAS_CHECK(
      phi::dynload::cublasLtMatmulDescSetAttribute(operationDesc,
                                                   CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transa_type,
                                                   sizeof(transa_type)));
  PADDLE_CUDABLAS_CHECK(
      phi::dynload::cublasLtMatmulDescSetAttribute(operationDesc,
                                                   CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &transb_type,
                                                   sizeof(transb_type)));

  const void* A_decode_scale_ptr = A_decode_scale.data();
  const void* B_decode_scale_ptr = B_decode_scale.data();
  const cudaDataType_t Atype = ScalarTypeToCudaDataType(A.dtype());
  const cudaDataType_t Btype = ScalarTypeToCudaDataType(B.dtype());
  const cudaDataType_t Dtype = ScalarTypeToCudaDataType(D->dtype());

  // split_accumulator is always true
  const int8_t fast_accum_mode = 0;
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_FAST_ACCUM,
      &fast_accum_mode,
      sizeof(fast_accum_mode)));
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &A_decode_scale_ptr,
      sizeof(A_decode_scale_ptr)));
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &B_decode_scale_ptr,
      sizeof(B_decode_scale_ptr)));

  // Setup mat layout descriptors
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutCreate(
      &Adesc,
      Atype,
      transa_type == CUBLAS_OP_N ? m : k,
      transa_type == CUBLAS_OP_N ? k : m,
      lda));
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutCreate(
      &Bdesc,
      Btype,
      transb_type == CUBLAS_OP_N ? k : n,
      transb_type == CUBLAS_OP_N ? n : k,
      ldb));

  PADDLE_CUDABLAS_CHECK(
      phi::dynload::cublasLtMatrixLayoutCreate(&Cdesc, Dtype, m, n, ldc));
  PADDLE_CUDABLAS_CHECK(
      phi::dynload::cublasLtMatrixLayoutCreate(&Ddesc, Dtype, m, n, ldd));

  // setup epilogue attributes
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));

  // setup preference attributes
  cublasLtMatmulPreference_t preference = nullptr;
  PADDLE_CUDABLAS_CHECK(
      phi::dynload::cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_size = workspace->dims()[0];

  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size)));

  PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatmul(ltHandle,
                                                     operationDesc,
                                                     &alpha,
                                                     A.data(),
                                                     Adesc,
                                                     B.data(),
                                                     Bdesc,
                                                     &beta,
                                                     D->data(),
                                                     Cdesc,
                                                     D->data(),
                                                     Ddesc,
                                                     /*algo*/ nullptr,
                                                     workspace->data(),
                                                     workspace_size,
                                                     stream));
  // Cleanup
  if (preference)
    PADDLE_CUDABLAS_CHECK(
        phi::dynload::cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc)
    PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc)
    PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    PADDLE_CUDABLAS_CHECK(phi::dynload::cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    PADDLE_CUDABLAS_CHECK(
        phi::dynload::cublasLtMatmulDescDestroy(operationDesc));
}

}  // anonymous namespace

template <typename T, typename Context>
void Fp8GemmBlockwiseKernel(const Context& dev_ctx,
                            const DenseTensor& A,
                            const DenseTensor& A_scale,
                            const DenseTensor& B,
                            const DenseTensor& B_scale,
                            const DenseTensor& input_result,
                            const DenseTensor& bias,
                            const DenseTensor& pre_gelu,
                            const DenseTensor& workspace,
                            bool transa,
                            bool transb,
                            bool grad,
                            bool accumulate,
                            bool use_split_accumulator,
                            int math_sm_count,
                            bool is_A_1d_scaled,
                            bool is_B_1d_scaled,
                            DenseTensor* output,
                            DenseTensor* pre_gelu_out,
                            DenseTensor* workspace_out) {
  cublas_gemm_blockwise_impl<Context>(dev_ctx,
                                      A,
                                      A_scale,
                                      B,
                                      B_scale,
                                      output,
                                      bias,
                                      pre_gelu_out,
                                      transa,
                                      transb,
                                      grad,
                                      workspace_out,
                                      accumulate,
                                      use_split_accumulator,
                                      math_sm_count,
                                      is_A_1d_scaled,
                                      is_B_1d_scaled,
                                      dev_ctx.stream());
}

}  // namespace phi

// Register the kernel
PD_REGISTER_KERNEL(fp8_gemm_blockwise,
                   GPU,
                   ALL_LAYOUT,
                   phi::Fp8GemmBlockwiseKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   uint8_t,
                   float,
                   double) {}
