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

#pragma once
namespace phi {
template <typename T, typename Context>
void FlashAttnV3GradKernel(
    const Context &ctx,
    const DenseTensor
        &q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const DenseTensor
        &k,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const DenseTensor
        &v,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const DenseTensor
        &out,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const DenseTensor
        &softmax_lse,  // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    const DenseTensor &
        out_grad,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    int const sm_margin,
    DenseTensor *dq,
    DenseTensor *dk,
    DenseTensor *dv);
}  // namespace phi
