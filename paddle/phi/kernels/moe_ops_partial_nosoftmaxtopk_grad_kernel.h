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

namespace phi {
template <typename T, typename Context>
void MoeGateDispatchPartialNoSoftMaxTopkGradKernel(
    const Context& dev_ctx,
    const DenseTensor& combine_weights_out,
    const DenseTensor& scatter_index,
    const DenseTensor& scatter_index_rev,
    const DenseTensor& expert_offset,
    const DenseTensor& expert_offset_local,
    const DenseTensor& y_grad,
    const DenseTensor& combine_weights_out_grad,
    int64_t k,
    int64_t capacity,
    bool use_pad,
    int64_t expert_start_index,
    int64_t expert_end_index,
    DenseTensor* x_grad,
    DenseTensor* combine_weights_grad);

}  // namespace phi
