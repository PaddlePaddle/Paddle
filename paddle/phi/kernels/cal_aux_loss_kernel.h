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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CalAuxLossKernel(const Context& dev_ctx,
                      const DenseTensor& gate_prob,
                      const DenseTensor& dispatch_mask,
                      const paddle::optional<DenseTensor>& tokens_mask,
                      const paddle::optional<DenseTensor>& dispatch_tokens_mask,
                      int64_t num_experts,
                      bool use_group,
                      int64_t moe_k,
                      float clip_min,
                      DenseTensor* l_aux_loss,
                      DenseTensor* seqlen_float,
                      DenseTensor* ce);

}  // namespace phi
