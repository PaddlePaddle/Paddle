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
#include "paddle/fluid/eager/accumulation/gradient_accumulation.h"
#include "paddle/fluid/eager/eager_tensor.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

static void CopyOrAddTensor(egr::EagerTensor* tensor,
                            const egr::EagerTensor& t) {
  if (!tensor->defined() || !tensor->initialized()) {
    // Simply copy tensor->impl
    *tensor = t;
  } else {
    // Accumulation
    egr::TensorAdd(t, tensor);
  }
}

namespace egr {

void GradNodeAccumulation::RetainGrad(
    const std::function<egr::EagerTensor(const egr::EagerTensor&)>& hook) {
  retain_grad_hook_ = hook;
}

std::vector<std::vector<egr::EagerTensor>> GradNodeAccumulation::operator()(
    const std::vector<std::vector<egr::EagerTensor>>& grads) {
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
  if (GradientHooksRegistered()) {
    std::vector<std::vector<egr::EagerTensor>> hooked_grads =
        ApplyGradientHooks(grads);
    // TODO(jiabin): It's little weird
    CopyOrAddTensor(&accumulated_grad, hooked_grads[0][0]);
  } else {
    CopyOrAddTensor(&accumulated_grad, grads[0][0]);
  }

  if (retain_grad_hook_ != nullptr) {
    retain_grad_hook_(accumulated_grad);
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }

  return {{accumulated_grad}};
}

}  // namespace egr
