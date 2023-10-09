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
void BlockMultiheadAttentionKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& key_cache,
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& block_tables,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out);

}  // namespace fusion
}  // namespace phi
