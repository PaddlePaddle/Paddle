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

#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/pten/core/dense_tensor.h"

namespace egr {
namespace egr_utils_api {

void RegisterGradientHookForTensor(
    const paddle::experimental::Tensor& tensor,
    std::function<paddle::experimental::Tensor(
        const paddle::experimental::Tensor&)>& hook) {
  // Find grad_node and out_rank from AutogradMeta
  std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);
  auto rank_info = EagerUtils::unsafe_autograd_meta(tensor)->OutRankInfo();

  grad_node->RegisterGradientHook(rank_info.first, rank_info.second, hook);
}

void RegisterReduceHookForTensor(const paddle::experimental::Tensor& tensor,
                                 const std::function<void(void)>& hook) {
  // Find grad_node and out_rank from AutogradMeta
  std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);

  grad_node->RegisterReduceHook(hook);
}

void RetainGradForTensor(const paddle::experimental::Tensor& tensor) {
  // TODO(jiabin): Support More Tensor type here
  AutogradMeta* meta = EagerUtils::unsafe_autograd_meta(tensor);
  std::weak_ptr<paddle::experimental::Tensor> weak_grad_tensor =
      meta->WeakGrad();

  // Define Hook
  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      hook = [weak_grad_tensor](const paddle::experimental::Tensor& t) {
        if (!weak_grad_tensor.expired()) {
          auto grad_tensor = weak_grad_tensor.lock();
          if (t.defined()) {
            VLOG(7) << "Set impl for RetainGrad Hook for tensor: " << t.name();
            // Simply Copy impl() to grad_tensor
            grad_tensor->set_impl(t.impl());
            return *grad_tensor.get();
          } else {
            PADDLE_THROW(paddle::platform::errors::Fatal(
                "Detected uninitialized variable, causing segmentation "
                "fault "
                "inside the hook."
                "Tensor has to be initialized while we need to set it."
                "please check tensor initialization status."));
          }
        } else {
          VLOG(7) << "Retain NULL paddle::experimental::Tensor in Grad Hook";
          return paddle::experimental::Tensor();
        }
      };

  if (IsLeafTensor(tensor)) {
    // Add RetainGrad as PostHook to AccumulationNode
    std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);
    PADDLE_ENFORCE(
        grad_node.get() != nullptr,
        paddle::platform::errors::Fatal("Detected NULL grad_node"
                                        "Leaf tensor should have had grad_node "
                                        "with type: GradNodeAccumulation"));
    auto accumulation_grad_node =
        std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node);
    accumulation_grad_node->RetainGrad(hook);

  } else {
    // Append to GradientHooks
    RegisterGradientHookForTensor(tensor, hook);
  }
}

}  // namespace egr_utils_api
}  // namespace egr
