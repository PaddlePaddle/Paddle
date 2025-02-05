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
#include "paddle/common/errors.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

enum class MHA_Layout {
  BS3HD = 0,
  BSHD_BS2HD = 1,
  BSHD_BSHD_BSHD = 2,
  // more layout to be added if needed in the future.
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

enum class MHA_Mask_Type {
  NO_MASK = 0,
  CAUSAL_MASK = 1,
  PADDING_MASK = 2,
  PADDING_CAUSAL_MASK = 3,
};

enum class MHA_Bias_Type {
  NO_BIAS = 0,
  PRE_SCALE_BIAS = 1,
  POST_SCALE_BIAS = 2
  // ALIBI = 3,
};

struct FADescriptor_v1 {
  std::int64_t b;
  std::int64_t h;
  std::int64_t hg;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  std::int64_t bias_b;
  std::int64_t bias_h;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  MHA_Layout layout;
  MHA_Bias_Type bias_type;
  MHA_Mask_Type mask_type;
  cudnn_frontend::DataType_t tensor_type;

  bool operator<(const FADescriptor_v1& rhs) const {
    return std::tie(b,
                    h,
                    hg,
                    s_q,
                    s_kv,
                    d,
                    bias_b,
                    bias_h,
                    attnScale,
                    isTraining,
                    dropoutProbability,
                    layout,
                    mask_type,
                    bias_type,
                    tensor_type) < std::tie(rhs.b,
                                            rhs.h,
                                            rhs.hg,
                                            rhs.s_q,
                                            rhs.s_kv,
                                            rhs.d,
                                            rhs.bias_b,
                                            rhs.bias_h,
                                            rhs.attnScale,
                                            rhs.isTraining,
                                            rhs.dropoutProbability,
                                            rhs.layout,
                                            rhs.mask_type,
                                            rhs.bias_type,
                                            rhs.tensor_type);
  }
};

void fused_attn_arbitrary_seqlen_fwd_impl(int64_t b,
                                          int64_t h,
                                          int64_t hg,
                                          int64_t s_q,
                                          int64_t s_kv,
                                          int64_t d,
                                          int64_t bias_b,
                                          int64_t bias_h,
                                          bool is_training,
                                          float scaling_factor,
                                          float dropout_probability,
                                          MHA_Layout layout,
                                          MHA_Bias_Type bias_type,
                                          MHA_Mask_Type mask_type,
                                          void* devPtrQ,
                                          void* devPtrK,
                                          void* devPtrV,
                                          void* devPtrBias,
                                          void* devPtrSoftmaxStats,
                                          void* devPtrO,
                                          void* devPtrDropoutSeed,
                                          void* devPtrDropoutOffset,
                                          void* devPtrCuSeqlensQ,
                                          void* devPtrCuSeqlensKV,
                                          cudnn_frontend::DataType_t tensorType,
                                          void* workspace,
                                          size_t* workspace_size,
                                          const phi::GPUContext& dev_ctx);

void fused_attn_arbitrary_seqlen_bwd_impl(int64_t b,
                                          int64_t h,
                                          int64_t hg,
                                          int64_t s_q,
                                          int64_t s_kv,
                                          int64_t d,
                                          int64_t bias_b,
                                          int64_t bias_h,
                                          float scaling_factor,
                                          float dropout_probability,
                                          MHA_Layout layout,
                                          MHA_Bias_Type bias_type,
                                          MHA_Mask_Type mask_type,
                                          bool deterministic,
                                          void* devPtrQ,
                                          void* devPtrKTranspose,
                                          void* devPtrVTranspose,
                                          void* devPtrO,
                                          void* devPtrSoftmaxStats,
                                          void* devPtrBias,
                                          void* devPtrdQ,
                                          void* devPtrdK,
                                          void* devPtrdV,
                                          void* devPtrdO,
                                          void* devPtrdBias,
                                          void* devPtrDropoutSeed,
                                          void* devPtrDropoutOffset,
                                          void* devPtrCuSeqlensQ,
                                          void* devPtrCuSeqlensKV,
                                          cudnn_frontend::DataType_t tensorType,
                                          void* workspace,
                                          size_t* workspace_size,
                                          const phi::GPUContext& dev_ctx);

#endif  // PADDLE_WITH_CUDNN_FRONTEND
