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

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"

namespace egr {

void RegisterGradientHookForTensor(pt::Tensor& tensor, std::function<pt::Tensor(const pt::Tensor&)>& hook) {
    // Find grad_node and out_rank from AutogradMeta
    int output_rank = EagerUtils::output_rank(tensor);
    std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);

    grad_node->RegisterGradientHook(output_rank, hook);
}

void RegisterReduceHookForTensor(pt::Tensor& tensor, std::function<void(void)>& hook) {
    // Find grad_node and out_rank from AutogradMeta
    std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);

    grad_node->RegisterReduceHook(hook);
}

} // namespace egr
