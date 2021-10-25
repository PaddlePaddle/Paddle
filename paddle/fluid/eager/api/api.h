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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/pten/hapi/all.h"
namespace egr {

void RegisterGradientHookForTensor(
    const egr::EagerTensor& tensor,
    std::function<egr::EagerTensor(const egr::EagerTensor&)>& hook);
void RegisterReduceHookForTensor(const egr::EagerTensor& tensor,
                                 const std::function<void(void)>& hook);
void RetainGradForTensor(const egr::EagerTensor& tensor);

egr::EagerTensor scale(const egr::EagerTensor& x, float scale, float bias,
                       bool bias_after_scale, bool trace_backward);

}  // namespace egr
