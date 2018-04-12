//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/broadcast_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

BroadcastOpHandle::BroadcastOpHandle(const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : local_scopes_(local_scopes), places_(places) {}

void BroadcastOpHandle::RunImpl() {
  PADDLE_ENFORCE_EQ(this->inputs_.size(), 1,
                    "The number of input should be one.");
  PADDLE_ENFORCE_EQ(
      this->outputs_.size(), places_.size(),
      "The number of output should equal to the number of places.");

  // Wait input done, this Wait is asynchronous operation
  auto in_var_handle = static_cast<VarHandle *>(this->inputs_[0]);
  auto &in_place = in_var_handle->place_;
  if (inputs_[0]->generated_op_)
    inputs_[0]->generated_op_->Wait(dev_ctxes_[in_place]);

  auto in_scope_idx = in_var_handle->scope_idx_;
  PADDLE_ENFORCE_LT(in_scope_idx, local_scopes_.size(),
                    "The input(%s) is not in the local_scopes.",
                    in_var_handle->name_);
  auto in_var = local_scopes_[in_scope_idx]->FindVar(in_var_handle->name_);

  Tensor *in_tensor = GetTensorFromVar(in_var);
  for (auto *out : outputs_) {
    auto out_handle = static_cast<VarHandle *>(out);
    auto &out_p = out_handle->place_;

    auto out_scope_idx = out_handle->scope_idx_;
    PADDLE_ENFORCE_LT(out_scope_idx, local_scopes_.size(),
                      "%s is not the the local_scopes ", out_handle->name_);
    auto *s = local_scopes_[out_scope_idx];
    auto out_var = s->FindVar(out_handle->name_);
    PADDLE_ENFORCE_EQ(out_p.which(), in_place.which(),
                      "The place of input and output should be the same.");

    if (in_var->IsType<framework::SelectedRows>()) {
      auto &in_sr = in_var->Get<framework::SelectedRows>();
      auto out_sr = out_var->GetMutable<framework::SelectedRows>();
      if (&in_sr == out_sr) continue;
      out_sr->set_height(in_sr.height());
      out_sr->set_rows(in_sr.rows());
      out_sr->mutable_value()->Resize(in_sr.value().dims());
      out_sr->mutable_value()->mutable_data(out_p, in_sr.value().type());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto in_lod = in_var->Get<framework::LoDTensor>();
      auto out_lod = out_var->GetMutable<framework::LoDTensor>();
      if (&in_lod == out_lod) continue;
      out_lod->set_lod(in_lod.lod());
      out_lod->Resize(in_lod.dims());
      out_lod->mutable_data(out_p, in_lod.type());
    } else {
      PADDLE_THROW("Var should be LoDTensor or SelectedRows.");
    }

    Tensor *out_tensor = GetTensorFromVar(out_var);

    paddle::framework::TensorCopy(*in_tensor, out_p, *(dev_ctxes_[in_place]),
                                  out_tensor);
  }
}

std::string BroadcastOpHandle::Name() const { return "broadcast"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
