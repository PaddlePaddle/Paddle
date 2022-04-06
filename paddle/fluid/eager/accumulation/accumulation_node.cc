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

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"

#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"
DECLARE_bool(retain_grad_for_all_tensor);
namespace egr {

static void CopyOrAddTensor(paddle::experimental::Tensor* tensor,
                            const paddle::experimental::Tensor& t) {
  if (!tensor->defined() || !tensor->initialized()) {
    // Simply copy tensor->impl
    *tensor = t;
  } else {
    // Accumulation
    paddle::imperative::TensorAdd<paddle::experimental::Tensor>(t, tensor);
  }
}

std::vector<std::vector<paddle::experimental::Tensor>> GradNodeAccumulation::
operator()(
    std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
    bool create_graph) {
  VLOG(3) << "Running Eager Backward Node: GradNodeAccumulation";
  PADDLE_ENFORCE(grads.size() == 1,
                 paddle::platform::errors::Fatal(
                     "GradNodeAccumulation should take exactly 1 grad tensor"
                     "However received: %d slot.",
                     grads.size()));
  PADDLE_ENFORCE(grads[0].size() == 1,
                 paddle::platform::errors::Fatal(
                     "GradNodeAccumulation should take exactly 1 grad tensor"
                     "However received: %d in slot %d .",
                     grads[0].size(), 0));
  // Apply Gradient Hooks
  paddle::experimental::Tensor grad_out;
  if (GradientHooksRegistered()) {
    std::vector<std::vector<paddle::experimental::Tensor>> hooked_grads =
        ApplyGradientHooks(grads);
    grad_out = hooked_grads[0][0];
  } else {
    grad_out = grads[0][0];
  }

  if (!weak_grad_.expired() && FLAGS_retain_grad_for_all_tensor) {
    auto grad = weak_grad_.lock();
    CopyOrAddTensor(grad.get(), grad_out);
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }

  return {{grad_out}};
}

void GradNodeAccumulation::RegisterReduceHook(
    std::shared_ptr<TensorVoidHook>&& hook) {
  reduce_hooks_.emplace_back(std::move(hook));
}

void GradNodeAccumulation::ApplyReduceHooks() {
  for (auto& hook : reduce_hooks_) {
    (*hook)();
  }
}
}  // namespace egr
