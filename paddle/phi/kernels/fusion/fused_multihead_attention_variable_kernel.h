// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const DenseTensor& seq_lens,
    const paddle::optional<DenseTensor>& mask,
    const float scale,
    const bool causal,
    DenseTensor* output);

template <typename T, typename Context>
void MultiHeadAttentionVariableWrapper(const Context& ctx,
                                       T* query,
                                       T* key,
                                       T* value,
                                       const int* seq_lens,
                                       const phi::DenseTensor* mask_tensor,
                                       const float scale,
                                       const bool causal,
                                       const int64_t batch_size,
                                       const int64_t num_heads,
                                       const int64_t seq_len,
                                       const int64_t out_seq_len,
                                       const int64_t head_size,
                                       const int64_t value_head_size,
                                       const int prompt_num,
                                       T* output);

}  // namespace fusion
}  // namespace phi
