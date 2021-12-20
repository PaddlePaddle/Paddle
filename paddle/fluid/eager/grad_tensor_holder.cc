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
#include "paddle/fluid/eager/accumulation/gradient_accumulation.h"

#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace egr {

static void FillUnderlyingVariableWithValue(
    double value, const paddle::framework::DDim& ddim,
    const paddle::platform::Place& place,
    const paddle::framework::proto::VarType::Type& dtype,
    egr::EagerTensor* target) {
  auto* dst_tensor =
      target->MutableVar()->GetMutable<paddle::framework::LoDTensor>();
  auto* dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
  dst_tensor->Resize(ddim);
  // TOOD(jiabin): Ugly fix here we have fwd_data_type_ and data_type, since in
  // grad mission
  // we can't get data_type_ directly. We need to check if we can only use
  // default data_type for now.
  dst_tensor->mutable_data(place, dtype);
  paddle::operators::math::set_constant(*dev_ctx, dst_tensor, value);
}

void GradTensorHolder::add(size_t slot_id, size_t rank,
                           const egr::EagerTensor& t, bool fill_one) {
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
  egr::EagerTensor& buffer_tensor = buffer_[slot_id][rank];
  if (!fill_one) {
    // TODO(jiabin): Code bellow is ugly to divide which inner var we used,
    // remove framework::Variable
    // related code later.
    // This if statement is trying to test neither pten::Tensor nor
    // framework::Variable is initialized.
    if ((!buffer_tensor.defined() || !buffer_tensor.initialized()) &&
        (!buffer_tensor.Var().IsInitialized())) {
      // Simply copy tensor->impl
      buffer_tensor = t;
    } else {
      // Accumulation
      if (t.initialized() && buffer_tensor.initialized()) {
        TensorAdd(t, &buffer_tensor);
      } else if (t.Var().IsInitialized() &&
                 buffer_tensor.Var().IsInitialized()) {
        VariableAdd(t, &buffer_tensor);
      } else if (t.Var().IsInitialized() && buffer_tensor.initialized()) {
        // TODO(jiabin): This can be merge to upper if case.
        buffer_tensor.SyncToVar();
        VariableAdd(t, &buffer_tensor);
      } else if (t.initialized() && buffer_tensor.Var().IsInitialized()) {
        buffer_tensor.SyncToTensor();
        TensorAdd(t, &buffer_tensor);
      } else {
        // Should not happend case
        // 1. both not init
      }
    }
  } else {
    // Create new tensor->impl and fill it with 1.0
    if (t.defined()) {
      // Fill 1.0
      paddle::experimental::Tensor tensor =
          paddle::experimental::ones_like(*t.Tensor().get());
      buffer_tensor.set_tensor(
          std::make_shared<paddle::experimental::Tensor>(tensor));

    } else {
      // TODO(jiabin): Only Support LodTensorForNow
      auto type = paddle::framework::ToVarType(t.Var().Type());
      switch (type) {
        case paddle::framework::proto::VarType::LOD_TENSOR: {
          auto t_ftensor = t.Var().Get<paddle::framework::LoDTensor>();
          FillUnderlyingVariableWithValue(1.0, t_ftensor.dims(),
                                          t_ftensor.place(), t_ftensor.type(),
                                          &buffer_tensor);
          break;
        }
        default: {
          PADDLE_THROW(paddle::platform::errors::NotFound(
              "Cannot found var type: %s in Fill Constant API",
              paddle::framework::ToTypeName(type)));
        }
      }
    }
  }
}

}  // namespace egr
