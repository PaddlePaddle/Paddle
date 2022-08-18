/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <sstream>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace operators {
class CublasLtHelper {
 public:
  CublasLtHelper(int m, int k, int n)
      : alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
    cublasStatus_t status;
    // handle and matmul desc
    status = dyl::cublasLtCreate(&handle_);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

    status = dyl::cublasLtMatmulDescCreate(
        &matmul_desc_, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatmulDescCreate"));
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        platform::errors::Fatal("cublasLtMatmulDescSetAttribute"));

    // matrix desc
    status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

    status = dyl::cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

    status = dyl::cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
  }
  ~CublasLtHelper() {
    if (handle_) dyl::cublasLtDestroy(handle_);
    if (matmul_desc_) dyl::cublasLtMatmulDescDestroy(matmul_desc_);
    if (A_desc_) dyl::cublasLtMatrixLayoutDestroy(A_desc_);
    if (B_desc_) dyl::cublasLtMatrixLayoutDestroy(B_desc_);
    if (C_desc_) dyl::cublasLtMatrixLayoutDestroy(C_desc_);
  }

  void GEMM(int8_t* A_dev,
            const int8_t* B_dev,
            int32_t* C_dev,
            cudaStream_t stream) {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

    cublasStatus_t status;
    VLOG(1) << "m=" << m_ << "k=" << k_ << "n=" << n_;

    cublasLtMatmulAlgo_t algo;
    int algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 15;
    int splitK_val = 0;
    int reductionScheme = 0;
#if CUDA_VERSION >= 11000
    int stages = 23;
#endif

    dyl::cublasLtMatmulAlgoInit(handle_,
                                CUBLAS_COMPUTE_32I,
                                CUDA_R_32I,
                                CUDA_R_8I,
                                CUDA_R_8I,
                                CUDA_R_32I,
                                CUDA_R_32I,
                                algoId,
                                &algo);
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
#if CUDA_VERSION >= 11000
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif

    status = dyl::cublasLtMatmul(handle_,
                                 matmul_desc_,
                                 &alpha_,
                                 B_dev,
                                 B_desc_,
                                 A_dev,
                                 A_desc_,
                                 &beta_,
                                 C_dev,
                                 C_desc_,
                                 C_dev,
                                 C_desc_,
                                 &algo,
                                 nullptr,
                                 0,
                                 stream);
    PADDLE_ENFORCE_EQ(status,
                      CUBLAS_STATUS_SUCCESS,
                      platform::errors::Fatal("cublasLtMatmul"));
    VLOG(1) << "gemm finsh";
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }

 private:
  cublasLtHandle_t handle_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t A_desc_;
  cublasLtMatrixLayout_t B_desc_;
  cublasLtMatrixLayout_t C_desc_;
  int32_t alpha_;
  int32_t beta_;

  int m_;
  int k_;
  int n_;
};

}  // namespace operators
}  // namespace paddle
