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

Tensor *GetTensorFromVar(Variable *in_var) {
  if (in_var->IsType<LoDTensor>()) {
    return in_var->GetMutable<LoDTensor>();
  } else if (in_var->IsType<SelectedRows>()) {
    return in_var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
  return nullptr;
}
BroadcastOpHandle::BroadcastOpHandle(const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::ContextMap &ctxs)
    : local_scopes_(local_scopes), places_(places), ctxs_(ctxs) {
  for (auto &p : places_) {
    this->dev_ctxes_[p] = ctxs_.DevCtx(p);
  }
}

void BroadcastOpHandle::RunImpl() {
  PADDLE_ENFORCE_EQ(this->inputs_.size(), 1);
  PADDLE_ENFORCE_EQ(this->outputs_.size(), places_.size());

  // Wait input done, this Wait is asynchronous operation
  auto in_var_handle = static_cast<VarHandle *>(this->inputs_[0]);
  auto &in_place = in_var_handle->place_;
  if (inputs_[0]->generated_op_)
    inputs_[0]->generated_op_->Wait(dev_ctxes_[in_place]);

  auto iter = std::find(places_.begin(), places_.end(), in_place);
  if (iter == places_.end()) {
    PADDLE_THROW("The input of BCast is not in the places_.");
  }

  int offset = iter - places_.begin();
  auto in_var = local_scopes_[offset]->FindVar(in_var_handle->name_);

  Tensor *in_tensor = GetTensorFromVar(in_var);
  for (auto *out : outputs_) {
    auto out_handle = static_cast<VarHandle *>(out);
    auto &out_p = out_handle->place_;

    auto iter = std::find(places_.begin(), places_.end(), out_p);
    if (iter == places_.end()) {
      PADDLE_THROW("The output of BCast is not in the places_.");
    }
    int offset = iter - places_.begin();

    auto *s = local_scopes_[offset];
    auto out_var = s->FindVar(out_handle->name_);

    PADDLE_ENFORCE_EQ(out_var->Type(), in_var->Type(), "");

    if (in_var->IsType<framework::SelectedRows>()) {
      auto in_sr = in_var->GetMutable<framework::SelectedRows>();
      auto out = out_var->GetMutable<framework::SelectedRows>();
      if (in_sr == out) continue;
      out->set_height(in_sr->height());
      out->set_rows(in_sr->rows());
      out->mutable_value()->Resize(in_sr->value().dims());
      out->mutable_value()->mutable_data(out_p, in_sr->value().type());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto in_lod = in_var->GetMutable<framework::LoDTensor>();
      auto out = out_var->GetMutable<framework::LoDTensor>();
      if (in_lod == out) continue;
      out->set_lod(in_lod->lod());
      out->Resize(in_lod->dims());
      out->mutable_data(out_p, in_lod->type());
    } else {
      PADDLE_THROW("Var should be LoDTensor or SelectedRows");
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
