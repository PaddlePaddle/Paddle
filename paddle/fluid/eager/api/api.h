// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/tcmpt/hapi/all.h"

namespace egr {

void RegisterGradientHookForTensor(
    const paddle::experimental::Tensor& tensor,
    std::function<paddle::experimental::Tensor(
        const paddle::experimental::Tensor&)>& hook);
void RegisterReduceHookForTensor(const paddle::experimental::Tensor& tensor,
                                 const std::function<void(void)>& hook);
void RetainGradForTensor(const paddle::experimental::Tensor& tensor);

paddle::experimental::Tensor scale(const paddle::experimental::Tensor& x,
                                   float scale, float bias,
                                   bool bias_after_scale, bool trace_backward);

}  // namespace egr
