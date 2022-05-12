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

namespace egr {

static void CopyOrAddTensor(paddle::experimental::Tensor* tensor,
                            const paddle::experimental::Tensor& t) {
  if (!tensor->defined() || !tensor->initialized()) {
    // Simply copy tensor->impl
    *tensor = t;
  } else {
    // Accumulation
    if (LIKELY(t.is_dense_tensor())) {
      if (LIKELY(tensor->is_dense_tensor())) {
        paddle::imperative::TensorAdd<paddle::experimental::Tensor>(t, tensor);
      } else {
        // TODO(jiabin): Support Other TensorBase later
        // TODO(zhanlve): Replace SelectedRowsAddTensor with
        // add_dygraph_function once it's supported
        paddle::experimental::Tensor new_buffer(
            std::make_shared<phi::DenseTensor>(), "tmp_accumulator");
        paddle::imperative::SelectedRowsAddTensor(*tensor, t, &new_buffer);
        tensor->set_impl(new_buffer.impl());
      }
    } else {
      // TODO(jiabin): Support Other TensorBase later
      // TODO(zhanlve): Replace SelectedRowsAddTensor with add_dygraph_function
      // once it's supported
      if (tensor->is_dense_tensor()) {
        paddle::imperative::SelectedRowsAddToTensor(t, tensor);
      } else {
        *tensor = std::move(*paddle::imperative::SelectedRowsMerge<
                            paddle::experimental::Tensor>(t, *tensor));
      }
    }
  }
}

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     kSlotSmallVectorSize>
GradNodeAccumulation::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>& grads,  // NOLINT
    bool create_graph,
    bool is_new_grad) {
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
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>
        hooked_grads = ApplyGradientHooks(grads);
    grad_out = hooked_grads[0][0];
  } else {
    grad_out = grads[0][0];
  }

  if (!weak_grad_.expired() && !is_new_grad) {
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
