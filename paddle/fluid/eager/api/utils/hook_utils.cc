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
#include "paddle/phi/core/dense_tensor.h"

namespace egr {
namespace egr_utils_api {

int64_t RegisterGradientHookForTensor(
    const paddle::experimental::Tensor& tensor,
    std::shared_ptr<egr::TensorHook>&& hook) {
  // Find grad_node and out_rank from AutogradMeta
  std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);
  auto rank_info = EagerUtils::unsafe_autograd_meta(tensor)->OutRankInfo();

  return grad_node->RegisterGradientHook(rank_info.first, rank_info.second,
                                         std::move(hook));
}

int64_t RegisterGradientHookForTensor(
    const paddle::experimental::Tensor& tensor,
    std::shared_ptr<egr::TensorHook>&& hook, egr::GradNodeBase* target_node) {
  auto rank_info = EagerUtils::unsafe_autograd_meta(tensor)->OutRankInfo();

  return target_node->RegisterGradientHook(rank_info.first, rank_info.second,
                                           std::move(hook));
}

void RegisterReduceHookForTensor(const paddle::experimental::Tensor& tensor,
                                 std::shared_ptr<egr::TensorVoidHook>&& hook) {
  if (IsLeafTensor(tensor)) {
    VLOG(6) << "Register ReduceHook for leaf tensor";
    std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(tensor);
    PADDLE_ENFORCE(
        grad_node.get() != nullptr,
        paddle::platform::errors::Fatal("Detected NULL grad_node,"
                                        "Leaf tensor should have had grad_node "
                                        "with type: GradNodeAccumulation"));
    auto accumulation_grad_node =
        std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node);
    accumulation_grad_node->RegisterReduceHook(std::move(hook));
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Only can register reduce hook for leaf Tensor."));
  }
}

// For grad
std::shared_ptr<paddle::experimental::Tensor> FetchGradForTensor(
    const paddle::experimental::Tensor& tensor,
    egr::GradNodeBase* target_node) {
  std::shared_ptr<paddle::experimental::Tensor> tmp{
      std::make_shared<paddle::experimental::Tensor>()};
  VLOG(6)
      << "Running in FetchGradForTensor, prepare FetchGrad Hook for tensor: "
      << tensor.name();
  auto hook = [tmp](const paddle::experimental::Tensor& t) {
    auto tmp_grad = tmp.get();
    if (t.defined()) {
      VLOG(6) << "Set impl for FetchGrad Hook for tensor: " << t.name();
      tmp_grad->set_impl(t.impl());
      tmp_grad->set_autograd_meta(t.mutable_autograd_meta());
      return t;
    } else {
      VLOG(6) << "Retain NULL paddle::experimental::Tensor in FetchGrad Hook";
      return paddle::experimental::Tensor();
    }
  };

  // Append to GradientHooks
  RegisterGradientHookForTensor(
      tensor, std::make_shared<egr::CppTensorHook>(hook), target_node);
  return tmp;
}

void RetainGradForTensor(const paddle::experimental::Tensor& tensor) {
  if (IsLeafTensor(tensor)) {
    // Leaf tensor's grad will always be retained
    // Refer to implementation of AccumulationNode for more details
    return;
  } else {
    AutogradMeta* meta = EagerUtils::unsafe_autograd_meta(tensor);
    if (meta->RetainGrads()) {
      return;
    } else {
      meta->SetRetainGrads(true);
    }

    std::weak_ptr<paddle::experimental::Tensor> weak_grad_tensor =
        meta->WeakGrad();

    // Define Hook
    auto hook = [weak_grad_tensor](const paddle::experimental::Tensor& t) {
      if (!weak_grad_tensor.expired()) {
        auto grad_tensor = weak_grad_tensor.lock();
        if (t.defined()) {
          VLOG(7) << "Set impl for RetainGrad Hook for tensor: " << t.name();
          // Simply Copy impl() to grad_tensor
          grad_tensor->set_impl(t.impl());
          grad_tensor->set_autograd_meta(t.mutable_autograd_meta());
          return *grad_tensor.get();
        } else {
          VLOG(7) << "Retain NULL paddle::experimental::Tensor in Grad Hook";
          return paddle::experimental::Tensor();
        }
      } else {
        VLOG(7) << "Retain NULL paddle::experimental::Tensor in Grad Hook";
        return paddle::experimental::Tensor();
      }
    };

    // Append to GradientHooks
    RegisterGradientHookForTensor(tensor,
                                  std::make_shared<egr::CppTensorHook>(hook));
  }
}

}  // namespace egr_utils_api
}  // namespace egr
