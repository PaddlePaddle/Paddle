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

#include "glog/logging.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"

namespace egr {

static void CopyOrAddTensor(paddle::experimental::Tensor* tensor,
                            const paddle::experimental::Tensor& t,
                            bool is_fake_empty) {
  if (is_fake_empty) {
    VLOG(3) << "Move Tensor ptr: " << t.impl();
    *tensor = t;
  } else {
    if (!tensor->defined() || !tensor->initialized()) {
      // Simply copy tensor->impl
      VLOG(3) << "Move Tensor ptr: " << t.impl();
      *tensor = t;
    } else {
      VLOG(3) << "Add Tensor ptr: " << t.impl()
              << " with Tensor ptr: " << tensor->impl();
      // Accumulation
      if (LIKELY(t.is_dense_tensor())) {
        if (LIKELY(tensor->is_dense_tensor())) {
          paddle::imperative::TensorAdd<paddle::experimental::Tensor>(t,
                                                                      tensor);
        } else {
          // TODO(jiabin): Support Other TensorBase later
          // TODO(zhanlve): Replace SelectedRowsAddTensor with
          // add_dygraph_function once it's supported
          paddle::experimental::Tensor new_buffer(
              std::make_shared<phi::DenseTensor>(), "tmp_accumulator");
          paddle::imperative::SelectedRowsAddTensor(*tensor, t, &new_buffer);
          tensor->set_impl(new_buffer.impl());
        }
      } else if (LIKELY(t.is_sparse_coo_tensor())) {
        // In fact, the gradient of SparseTensor is still a SparseTensor
        if (LIKELY(tensor->is_sparse_coo_tensor())) {
          auto t_sparse =
              std::dynamic_pointer_cast<phi::SparseCooTensor>(t.impl());
          paddle::experimental::Tensor t_values(
              std::make_shared<phi::DenseTensor>(
                  t_sparse->non_zero_elements()));
          auto tensor_sparse =
              std::dynamic_pointer_cast<phi::SparseCooTensor>(tensor->impl());
          paddle::experimental::Tensor tensor_values(
              std::make_shared<phi::DenseTensor>(
                  tensor_sparse->non_zero_elements()));
          paddle::imperative::TensorAdd<paddle::experimental::Tensor>(
              t_values, &tensor_values);
        }
      } else {
        // TODO(jiabin): Support Other TensorBase later
        // TODO(zhanlve): Replace SelectedRowsAddTensor with
        // add_dygraph_function
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
}

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     kSlotSmallVectorSize>
GradNodeAccumulation::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>& grads,  // NOLINT
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API Grad: GradNodeAccumulation";
  PADDLE_ENFORCE(grads.size() == 1,
                 paddle::platform::errors::Fatal(
                     "GradNodeAccumulation should take exactly 1 grad tensor"
                     "However received: %d slot.",
                     grads.size()));
  PADDLE_ENFORCE(grads[0].size() == 1,
                 paddle::platform::errors::Fatal(
                     "GradNodeAccumulation should take exactly 1 grad tensor"
                     "However received: %d in slot %d .",
                     grads[0].size(),
                     0));
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
    CopyOrAddTensor(grad.get(), grad_out, is_fake_empty_);
    is_fake_empty_ = false;
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }
  VLOG(3) << "Finish AD API Grad: GradNodeAccumulation";
  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s], Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = "(grads[0][0], [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grads[0][0]));
    const char* TENSOR_X_GRAD_TEMPLATE = "(grad_out, [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    output_str += output_x_grad_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }
  return {{grad_out}};
}

void GradNodeAccumulation::RegisterReduceHook(
    std::shared_ptr<VoidHook>&& hook) {
  reduce_hooks_.emplace_back(std::move(hook));
}

void GradNodeAccumulation::ApplyReduceHooks() {
  for (auto& hook : reduce_hooks_) {
    (*hook)();
  }
}
}  // namespace egr
