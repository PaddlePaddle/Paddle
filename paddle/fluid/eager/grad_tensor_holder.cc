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

#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace egr {

void GradTensorHolder::SetBufferSlotRankZeros(size_t slot_id, size_t rank) {
  buffer_[slot_id][rank] =
      paddle::experimental::zeros_like(buffer_[slot_id][rank]);
}

void GradTensorHolder::CopyValueFromTensor(
    size_t slot_id, size_t rank, const paddle::experimental::Tensor& t,
    bool fill_one) {
  // TODO(jiabin): We need to deal with empty input_buffer with slot size not
  // empty;
  PADDLE_ENFORCE(slot_id < buffer_.size(),
                 paddle::platform::errors::Fatal(
                     "Invalid slot_id for GradTensorHolder::add() "
                     "which exceeds size of buffer"));
  VLOG(6) << "Add Tensor for buffer_ slot: " << slot_id
          << ", size: " << buffer_[slot_id].size();
  if (buffer_[slot_id].empty()) {
    VLOG(6) << "Pass add Tensor for buffer_ slot: " << slot_id
            << " since its buffer_ is empty ";
    return;
  }
  PADDLE_ENFORCE(
      rank < buffer_[slot_id].size(),
      paddle::platform::errors::Fatal(
          "Invalid rank for GradTensorHolder::add() which exceeds size "
          "of buffer slot %d, got slot size is: %d rank is: %d",
          slot_id, buffer_[slot_id].size(), rank));
  if (!fill_one) {
    paddle::experimental::Tensor& buffer_tensor = buffer_[slot_id][rank];
    if ((!buffer_tensor.defined() || !buffer_tensor.initialized())) {
      // Perform deep copy here
      buffer_tensor.copy_(t, t.place(), false);
      buffer_tensor.set_autograd_meta(t.mutable_autograd_meta());

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Cannot copy grad_tensors' value to grad tensor holders,"
          "input buffer has already been initialized."));
    }
  } else {
    // Create new tensor->impl and fill it with 1.0
    if (t.defined()) {
      // Fill 1.0, use full to support complex, one_like don't support it.
      buffer_[slot_id][rank] =
          paddle::experimental::full(t.shape(), 1, t.dtype(), t.place());
    }
  }
}

void GradTensorHolder::add(size_t slot_id, size_t rank,
                           const paddle::experimental::Tensor& t,
                           bool create_graph) {
  // TODO(jiabin): We need to deal with empty input_buffer with slot size not
  // empty;
  PADDLE_ENFORCE(slot_id < buffer_.size(),
                 paddle::platform::errors::Fatal(
                     "Invalid slot_id for GradTensorHolder::add() "
                     "which exceeds size of buffer"));
  VLOG(6) << "Add Tensor for buffer_ slot: " << slot_id
          << ", size: " << buffer_[slot_id].size();
  if (buffer_[slot_id].empty()) {
    VLOG(6) << "Pass add Tensor for buffer_ slot: " << slot_id
            << " since its buffer_ is empty ";
    return;
  }
  PADDLE_ENFORCE(
      rank < buffer_[slot_id].size(),
      paddle::platform::errors::Fatal(
          "Invalid rank for GradTensorHolder::add() which exceeds size "
          "of buffer slot %d, got slot size is: %d rank is: %d",
          slot_id, buffer_[slot_id].size(), rank));

  paddle::experimental::Tensor& buffer_tensor = buffer_[slot_id][rank];
  // TODO(jiabin): Code bellow is ugly to divide which inner var we used,
  // remove framework::Variable
  // related code later.
  // This if statement is trying to test neither phi::Tensor nor
  // framework::Variable is initialized.
  if ((!buffer_tensor.defined() || !buffer_tensor.initialized())) {
    // Simply copy tensor->impl
    buffer_tensor = t;
  } else {
    // Accumulation
    PADDLE_ENFORCE_EQ(t.initialized(), true,
                      paddle::platform::errors::Fatal(
                          "We can only accumulate initialized tensor, but we "
                          "got tensor: %s is empty please check you network "
                          "and make sure it creates grads.",
                          t.name()));

    if (t.is_dense_tensor()) {
      if (buffer_tensor.is_dense_tensor()) {
        if (create_graph) {
          buffer_tensor = add_final_state_dygraph_function(t, buffer_tensor);
        } else {
          paddle::imperative::TensorAdd<paddle::experimental::Tensor>(
              t, &buffer_tensor);
        }
      } else {
        // TODO(jiabin): Support Other TensorBase later
        // TODO(zhanlve): Replace SelectedRowsAddTensor with
        // add_dygraph_function once it's supported
        paddle::experimental::Tensor new_buffer(
            std::make_shared<phi::DenseTensor>(), "tmp_accumulator");
        paddle::imperative::SelectedRowsAddTensor(buffer_tensor, t,
                                                  &new_buffer);
        buffer_tensor.set_impl(new_buffer.impl());
      }
    } else {
      // TODO(jiabin): Support Other TensorBase later
      // TODO(zhanlve): Replace SelectedRowsAddTensor with add_dygraph_function
      // once it's supported
      if (buffer_tensor.is_dense_tensor()) {
        paddle::imperative::SelectedRowsAddToTensor(t, &buffer_tensor);
      } else {
        buffer_tensor =
            std::move(*paddle::imperative::SelectedRowsMerge<
                      paddle::experimental::Tensor>(t, buffer_tensor));
      }
    }
  }
}

}  // namespace egr
