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
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/core/dense_tensor.h"

namespace dyl = phi::dynload;

namespace phi {

struct CublasLtAlgoParam {
  int algoId;
  int swizzle;
  int customOption;
  int tile;
  int splitK_val;
  int reductionScheme;
  int stages;
  size_t workspace_size;
};

const std::map<std::tuple<int, int, int>, CublasLtAlgoParam> AlgoParamCache{};

class CublasLtHelper {
 public:
  CublasLtHelper(int m, int k, int n, cublasLtHandle_t handle)
      : handle_(handle), alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
    cublasStatus_t status;
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t cudaComputeType = CUDA_R_32I;
#else
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
#endif

    // matmul desc
#if CUBLAS_VER_MAJOR < 11
    status = dyl::cublasLtMatmulDescCreate(&matmul_desc_, cudaComputeType);
#else
    status = dyl::cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32I);
#endif

    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatmulDescCreate execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatmulDescSetAttribute execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));

    // matrix desc
    status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatrixLayoutCreate execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));

    status = dyl::cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatrixLayoutCreate execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));

    status = dyl::cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatrixLayoutCreate execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));

#if CUDA_VERSION >= 11020

    int algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 15;
    int splitK_val = 0;
    int reductionScheme = 0;
    int stages = 23;
    workspace_size_ = 0;
    if (m >= 128) {
      tile = 20;
      stages = 17;
    }

    std::tuple<int, int, int> key(m_, k_, n_);
    if (AlgoParamCache.count(key) != 0) {
      auto value = AlgoParamCache.at(key);
      algoId = value.algoId;
      swizzle = value.swizzle;
      customOption = value.customOption;
      tile = value.tile;
      splitK_val = value.splitK_val;
      reductionScheme = value.reductionScheme;
      stages = value.stages;
      workspace_size_ = value.workspace_size;
    }

    dyl::cublasLtMatmulAlgoInit(handle_,
                                cudaComputeType,
                                CUDA_R_32I,
                                CUDA_R_8I,
                                CUDA_R_8I,
                                CUDA_R_32I,
                                CUDA_R_32I,
                                algoId,
                                &algo_);
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo_,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        &(swizzle),
        sizeof(swizzle));
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
#if CUDA_VERSION >= 11000
    dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
#endif
  }
  ~CublasLtHelper() {}

  void GEMM(const int8_t* A_dev,
            const int8_t* B_dev,
            int32_t* C_dev,
            cudaStream_t stream,
            void* workspace = nullptr) {
    cublasStatus_t status;

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
#if CUDA_VERSION >= 11020
                                 &algo_,
                                 workspace,
                                 workspace_size_,
#else
                                 nullptr,
                                 nullptr,
                                 0,
#endif
                                 stream);
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        phi::errors::External(
            "cublasLtMatmul execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));
  }

 private:
  cublasLtHandle_t handle_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t A_desc_;
  cublasLtMatrixLayout_t B_desc_;
  cublasLtMatrixLayout_t C_desc_;

  cublasLtMatmulAlgo_t algo_;

  int32_t alpha_ = 1;
  int32_t beta_ = 0;

  int m_ = 0;
  int k_ = 0;
  int n_ = 0;

  size_t workspace_size_ = 0;
};

}  // namespace phi
