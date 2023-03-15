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

template <typename T, typename Context>
void MemoryEfficientAttentionForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& cu_seqlens_q,
    const paddle::optional<DenseTensor>& cu_seqlens_k,
    const paddle::optional<DenseTensor>& causal_diagonal,
    const paddle::optional<DenseTensor>& seqlen_k,
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
    const bool causal,
    const double dropout_p,
    const float scale,
    const bool is_test,
    DenseTensor* output,
    DenseTensor* logsumexp,
    DenseTensor* seed_and_offset);

}  // namespace phi
