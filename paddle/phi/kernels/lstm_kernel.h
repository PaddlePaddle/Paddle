// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void LSTMKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const paddle::optional<DenseTensor>& h0,
                const paddle::optional<DenseTensor>& c0,
                const DenseTensor& weight,
                const DenseTensor& bias,
                bool use_peepholes,
                bool is_reverse,
                bool is_test,
                const std::string& gate_activation,
                const std::string& cell_activation,
                const std::string& candidate_activation,
                DenseTensor* hidden,
                DenseTensor* cell,
                DenseTensor* batch_gate,
                DenseTensor* batch_cell_pre_act);

template <typename T, typename Context>
void LSTMGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const paddle::optional<DenseTensor>& h0,
                    const paddle::optional<DenseTensor>& c0,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    const DenseTensor& hidden,
                    const DenseTensor& cell,
                    const DenseTensor& batch_gate,
                    const DenseTensor& batch_cell_pre_act,
                    const DenseTensor& hidden_grad,
                    bool use_peepholes,
                    bool is_reverse,
                    bool is_test,
                    const std::string& gate_activation,
                    const std::string& cell_activation,
                    const std::string& candidate_activation,
                    DenseTensor* input_grad,
                    DenseTensor* h0_grad,
                    DenseTensor* c0_grad,
                    DenseTensor* weight_grad,
                    DenseTensor* bias_grad);

}  // namespace phi
