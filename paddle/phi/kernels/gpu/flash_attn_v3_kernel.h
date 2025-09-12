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
void FlashAttnV3Kernel(const Context &ctx,
                       const DenseTensor &q,
                       const DenseTensor &k,
                       const DenseTensor &v,
                       const paddle::optional<DenseTensor> &q_v_,
                       const paddle::optional<DenseTensor> &q_descale_,
                       const paddle::optional<DenseTensor> &k_descale_,
                       const paddle::optional<DenseTensor> &v_descale_,
                       const float softmax_scale,
                       bool is_causal,
                       int window_size_left,
                       int window_size_right,
                       const float softcap,
                       int num_splits,
                       const bool manual_set_pack_gqa,
                       const bool pack_gqa_,
                       const int sm_margin,
                       DenseTensor *out,
                       DenseTensor *softmax_lse);
}  // namespace phi
