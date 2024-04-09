// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef PADDLE_WITH_CUDNN_FRONTEND
#include "paddle/phi/backends/dynload/cudnn_frontend.h"

#define CUDNN_CALL(func)                                                   \
  {                                                                        \
    auto status = func;                                                    \
    if (status != CUDNN_STATUS_SUCCESS) {                                  \
      std::stringstream ss;                                                \
      ss << "CUDNN Error : " << phi::dynload::cudnnGetErrorString(status); \
      PADDLE_THROW(phi::errors::Fatal(ss.str()));                          \
    }                                                                      \
  }

enum class MHA_Layout {
  NOT_INTERLEAVED = 0,
  QKV_INTERLEAVED = 1,
  KV_INTERLEAVED = 2
};

enum class MHA_Matrix {
  Q_Matrix = 0,            // queries
  K_Matrix = 1,            // keys
  K_Matrix_Transpose = 2,  // keys transposed
  V_Matrix = 3,            // values
  V_Matrix_Transpose = 4,  // value matrix transposed
  S_Matrix = 5,            // output of GEMM1
  O_Matrix = 6,            // final output
};

enum class MHA_Mask_Type { NO_MASK = 0, CAUSAL_MASK = 1, PADDING_MASK = 2 };

enum class MHA_Bias_Type {
  NO_BIAS = 0,
  PRE_SCALE_BIAS = 1,
  POST_SCALE_BIAS = 2
};

void fused_attn_arbitrary_seqlen_fwd(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    bool is_training,
    float scaling_factor,
    float dropout_probability,
    MHA_Layout layout,
    MHA_Mask_Type mask_type,
    void* devPtrQ,
    void* devPtrK,
    void* devPtrV,
    void* devPtrSoftmaxStats,
    void* devPtrO,
    void* devPtrMask,
    // void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV,
    void* devPtrDropoutSeed,
    void* devPtrDropoutOffset,
    cudnnDataType_t tensorType,
    cudaStream_t stream,
    cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    float dropout_probability,
    MHA_Layout layout,
    MHA_Mask_Type mask_type,
    void* devPtrQ,
    void* devPtrK,
    void* devPtrV,
    void* devPtrO,
    void* devPtrSoftmaxStats,
    void* devPtrdQ,
    void* devPtrdK,
    void* devPtrdV,
    void* devPtrdO,
    void* devPtrMask,
    // void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV,
    void* devPtrDropoutSeed,
    void* devPtrDropoutOffset,
    cudnnDataType_t tensorType,
    cudaStream_t stream,
    cudnnHandle_t handle,
    bool use_workspace_opt);

#endif  // PADDLE_WITH_CUDNN_FRONTEND
