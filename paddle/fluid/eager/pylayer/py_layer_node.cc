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

#include "paddle/fluid/eager/pylayer/py_layer_node.h"
#include "paddle/fluid/eager/eager_tensor.h"

#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

std::vector<std::vector<paddle::experimental::Tensor>> GradNodePyLayer::
operator()(
    const std::vector<std::vector<paddle::experimental::Tensor>>& grads) {
  VLOG(3) << "Running Eager Backward Node: GradNodePyLayer";
  paddle::experimental::Tensor grad_out;

  return {{grad_out}};
}

void GradNodePyLayer::RegisterReduceHook(
    std::shared_ptr<TensorVoidHook>&& hook) {
  reduce_hooks_.emplace_back(std::move(hook));
}

void GradNodePyLayer::ApplyReduceHooks() {
  for (auto& hook : reduce_hooks_) {
    (*hook)();
  }
}
}  // namespace egr
