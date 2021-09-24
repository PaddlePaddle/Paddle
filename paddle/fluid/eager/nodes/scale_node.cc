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

#include "paddle/fluid/eager/nodes/scale_node.h"
#include "paddle/fluid/eager/function_api.h"

#include "paddle/tcmpt/api/all.h"

#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

void GradNodeScale::SetTensorWrappers_X(
    const std::vector<pt::Tensor>& tensors) {
  // Does nothing for scale
}

void GradNodeScale::SetAttributes_scale(float scale) { scale_ = scale; }

std::vector<std::vector<pt::Tensor>> GradNodeScale::operator()(
    const std::vector<std::vector<pt::Tensor>>& grads) {
  // 1. Check Output Size
  PADDLE_ENFORCE(((grads.size() == 1) && (grads[0].size() == 1)),
                 paddle::platform::errors::Fatal(
                     "ScaleGradNode should take exactly 1 grad tensor"
                     "However received: %d",
                     grads.size()));
  std::vector<std::vector<pt::Tensor>> outs;
  // 2. Create needed out parttern
  pt::Tensor out;
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<pt::tensor>> to apply all hooks?
    std::vector<std::vector<pt::Tensor>> hooked_grads =
        ApplyGradientHooks(grads);
    ScaleAPI(/* slot by slot set */ hooked_grads[0][0], scale_, 0.0 /* bias */,
             true /* bias_after_scale */, &out);
  } else {
    ScaleAPI(grads[0][0], scale_, 0.0 /* bias */, true /* bias_after_scale */,
             &out);
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }

  return {{out}};
}

}  // namespace egr
