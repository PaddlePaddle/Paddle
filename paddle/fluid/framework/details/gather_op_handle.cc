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

#include "paddle/fluid/framework/details/gather_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

GatherOpHandle::GatherOpHandle(const std::vector<Scope *> &local_scopes,
                               const std::vector<platform::Place> &places)
    : local_scopes_(local_scopes), places_(places) {}

void GatherOpHandle::RunImpl() {
  PADDLE_ENFORCE_EQ(
      this->inputs_.size(), places_.size(),
      "The number of inputs should be equal to the number of place.");
  PADDLE_ENFORCE_EQ(this->outputs_.size(), 1,
                    "The number of output should be one.");
  auto in_0_handle = static_cast<VarHandle *>(inputs_[0]);
  auto pre_in_var =
      local_scopes_[in_0_handle->scope_idx_]->FindVar(in_0_handle->name_);
  PADDLE_ENFORCE(pre_in_var->IsType<framework::SelectedRows>(),
                 "Currently, gather_op only can gather SelectedRows.");
  auto pre_place = in_0_handle->place_;

  // Wait input done, this Wait is asynchronous operation
  for (auto *in : inputs_) {
    if (inputs_[0]->generated_op_) {
      auto &p = static_cast<VarHandle *>(in)->place_;
      in->generated_op_->Wait(dev_ctxes_[p]);
    }
  }

  std::vector<int64_t> out_rows;
  std::vector<Tensor *> in_tensors;
  std::vector<platform::Place> in_places;

  // gather the inputs
  for (auto *in : inputs_) {
    auto in_handle = static_cast<VarHandle *>(in);
    auto in_p = in_handle->place_;
    in_places.push_back(in_p);
    PADDLE_ENFORCE_LT(in_handle->scope_idx_, local_scopes_.size(),
                      "%s is not the the local_scopes ", in_handle->name_);
    PADDLE_ENFORCE_EQ(in_p.which(), pre_place.which(),
                      "The place of input should be the same.");
    auto *s = local_scopes_[in_handle->scope_idx_];
    auto in_var = s->FindVar(in_handle->name_);
    PADDLE_ENFORCE_EQ(in_var->Type(), pre_in_var->Type(),
                      "The type of input is not consistent.");

    if (in_var->IsType<framework::SelectedRows>()) {
      auto &pre_in = pre_in_var->Get<framework::SelectedRows>();
      auto &in_sr = in_var->Get<framework::SelectedRows>();
      auto in_sr_rows = in_sr.rows();
      out_rows.insert(out_rows.begin(), in_sr_rows.begin(), in_sr_rows.end());
      PADDLE_ENFORCE_EQ(pre_in.height(), in_sr.height(),
                        "The height of inputs is not consistent.");
      PADDLE_ENFORCE_EQ(pre_in.GetCompleteDims(), in_sr.GetCompleteDims(), ,
                        "The dims of inputs is not consistent.");
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto &pre_in = pre_in_var->Get<framework::LoDTensor>();
      auto &in_lodtensor = in_var->Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(in_lodtensor.lod(), pre_in.lod(),
                        "The lod of inputs is not consistent.");
      PADDLE_ENFORCE_EQ(in_lodtensor.dims(), pre_in.dims(),
                        "The dims of inputs is not consistent.");
    } else {
      PADDLE_THROW("Var should be LoDTensor or SelectedRows.");
    }
    in_tensors.push_back(GetTensorFromVar(in_var));
    pre_in_var = in_var;
  }

  // write the output
  auto out_handle = static_cast<VarHandle *>(this->outputs_[0]);
  auto &out_place = out_handle->place_;
  auto out_scope_idx = out_handle->scope_idx_;
  auto out_var = local_scopes_[out_scope_idx]->FindVar(out_handle->name_);
  PADDLE_ENFORCE_EQ(out_place.which(), pre_place.which(),
                    "The place of input and output should be the same.");
  if (pre_in_var->IsType<framework::SelectedRows>()) {
    auto &pre_in = pre_in_var->Get<framework::SelectedRows>();
    auto out = out_var->GetMutable<framework::SelectedRows>();
    out->set_height(pre_in.height());
    out->set_rows(out_rows);
    size_t rows = out_rows.size();
    DDim out_dim = pre_in.GetCompleteDims();
    out_dim[0] = static_cast<int64_t>(rows);
    out->mutable_value()->Resize(out_dim);
    out->mutable_value()->mutable_data(out_place, pre_in.value().type());
    auto out_tensor = out->mutable_value();
    // copy
    int s = 0, e = 0;
    for (size_t j = 0; j < in_tensors.size(); ++j) {
      e += in_tensors[j]->dims()[0];
      auto sub_out = out_tensor->Slice(s, e);
      paddle::framework::TensorCopy(*(in_tensors[j]), out_place,
                                    *(dev_ctxes_[in_places[j]]), &sub_out);
      s = e;
    }
  } else if (pre_in_var->IsType<framework::LoDTensor>()) {
    PADDLE_THROW("Currently, Var only can be SelectedRows.");
  } else {
    PADDLE_THROW("Var should be SelectedRows.");
  }
}

std::string GatherOpHandle::Name() const { return "gather"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
