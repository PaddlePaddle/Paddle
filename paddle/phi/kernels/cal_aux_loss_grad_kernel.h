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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void CalAuxLossGradKernel(const Context& dev_ctx,
                          const DenseTensor& gate_prob,
                          const DenseTensor& seqlen_float,
                          const DenseTensor& ce,
                          const DenseTensor& l_aux_loss_grad,
                          const int64_t num_experts,
                          const bool use_group,
                          const int64_t moe_k,
                          DenseTensor* gate_prob_grad);

}  // namespace phi
