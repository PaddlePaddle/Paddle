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
#include "paddle/top/core/dense_tensor.h"
#include "paddle/fluid/eager/nodes/accumulation_node.h"

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

void RetainGradForTensor(pt::Tensor& tensor) {
    auto tensor_instance = std::dynamic_pointer_cast<pt::DenseTensor>(tensor.impl());
    
    AutogradMeta* meta = EagerUtils::autograd_meta(tensor);
    pt::Tensor& grad_tensor = meta->MutableGrad();

    // Define Hook
    std::function<pt::Tensor(const pt::Tensor&)> hook = [&grad_tensor](const pt::Tensor& t) { 
        // Simply Copy impl() to grad_tensor
        grad_tensor.SetImpl(t.impl());
        
        return grad_tensor;
    };

    if(EagerUtils::IsLeafTensor(tensor)) {
        // Add RetainGrad as PostHook to AccumulationNode
        std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);
        auto accumulation_grad_node = std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node);
        PADDLE_ENFORCE(accumulation_grad_node != nullptr,
                    paddle::platform::errors::Fatal("Leaf tensor should have had grad_node with type: GradNodeAccumulation"));
        accumulation_grad_node->RetainGrad(hook);
        
    } else {
        // Append to GradientHooks
        RegisterGradientHookForTensor(tensor, hook);
    }
}

} // namespace egr
