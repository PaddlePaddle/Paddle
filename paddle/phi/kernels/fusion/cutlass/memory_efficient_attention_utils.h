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

#include "paddle/phi/core/ddim.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

inline int64_t GetMemoryEfficientBiasStrideB(const phi::DDim &bias_dims,
                                             const phi::DDim &q_dims,
                                             const phi::DDim &k_dims) {
  int bias_dims_rank = bias_dims.size();
  if (bias_dims_rank != 2) {
    PADDLE_ENFORCE_EQ(bias_dims_rank,
                      4,
                      phi::errors::InvalidArgument(
                          "The rank of attn_bias should be 2 or 4."));
  }
  PADDLE_ENFORCE_EQ(
      bias_dims[bias_dims_rank - 1],
      k_dims[1],
      phi::errors::InvalidArgument("The last dim of attn_bias should be "
                                   "equal to the sequence length of key."));
  PADDLE_ENFORCE_EQ(bias_dims[bias_dims_rank - 2],
                    q_dims[1],
                    phi::errors::InvalidArgument(
                        "The 2nd last dim of attn_bias should be equal to "
                        "the sequence length of query."));

  if (bias_dims_rank == 2) {
    return 0;
  }

  if (bias_dims[0] == q_dims[0] && bias_dims[1] == q_dims[2]) {
    return q_dims[2] * q_dims[1] * k_dims[1];
  }

  PADDLE_ENFORCE_EQ(
      bias_dims[0],
      1,
      phi::errors::InvalidArgument(
          "The first dim of attn_bias should be 1 or batch size."));
  PADDLE_ENFORCE_EQ(
      bias_dims[1],
      1,
      phi::errors::InvalidArgument(
          "The second dim of attn_bias should be 1 or num_heads."));
  return 0;
}

#define PD_MEA_CHECK_OVERFLOW(__dst, ...)                                    \
  do {                                                                       \
    auto __tmp = (__VA_ARGS__);                                              \
    using __SrcType = decltype(&__tmp);                                      \
    using __DstType = typename std::remove_reference<decltype(__dst)>::type; \
    auto __max_dst_value = std::numeric_limits<__DstType>::max();            \
    if (!std::is_same<__SrcType, __DstType>::value) {                        \
      PADDLE_ENFORCE_LE(                                                     \
          __tmp,                                                             \
          __max_dst_value,                                                   \
          phi::errors::InvalidArgument(#__dst " exceeds maximum value."));   \
    }                                                                        \
    __dst = __tmp;                                                           \
  } while (0)

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
