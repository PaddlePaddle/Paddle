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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/fusion/gpu/masked_multiquery_attention_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MMQAKernel(const Context& dev_ctx,
                const DenseTensor& query,
                const DenseTensor& key,
                const DenseTensor& value,
                const DenseTensor& cache_kv,
                const paddle::optional<DenseTensor>& src_mask,
                const paddle::optional<DenseTensor>& cum_offsets,
                const paddle::optional<DenseTensor>& sequence_lengths,
                const paddle::optional<DenseTensor>& rotary_tensor,
                const paddle::optional<DenseTensor>& beam_cache_offset,
                const paddle::optional<DenseTensor>& out_linear_shift,
                const paddle::optional<DenseTensor>& out_linear_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const float out_linear_in_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                DenseTensor* out,
                DenseTensor* cache_kv_out,
                DenseTensor* beam_cache_offset_out);

}  // namespace fusion
}  // namespace phi
