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
  // the input may have dummy var.
  std::vector<VarHandle *> in_var_handles;
  for (auto *in : inputs_) {
    auto *in_handle = dynamic_cast<VarHandle *>(in);
    if (in_handle) {
      in_var_handles.push_back(in_handle);
    }
  }
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  // the output may have dummy var.
  std::vector<VarHandle *> out_var_handles;
  for (auto *out : outputs_) {
    auto *out_handle = dynamic_cast<VarHandle *>(out);
    if (out_handle) {
      out_var_handles.push_back(out_handle);
    }
  }
  PADDLE_ENFORCE_EQ(out_var_handles.size(), 1,
                    "The number of output should be one.");

  auto in_0_handle = static_cast<VarHandle *>(in_var_handles[0]);
  auto pre_in_var =
      local_scopes_[in_0_handle->scope_idx_]->FindVar(in_0_handle->name_);
  auto pre_place = in_0_handle->place_;

  PADDLE_ENFORCE(pre_in_var->IsType<framework::SelectedRows>(),
                 "Currently, gather_op only can gather SelectedRows.");

  PADDLE_ENFORCE_EQ(out_var_handles[0]->place_.which(), pre_place.which(),
                    "The place of input and output should be the same.");

  // Wait input done, this Wait is asynchronous operation
  for (auto *in : in_var_handles) {
    if (in->generated_op_) {
      in->generated_op_->Wait(dev_ctxes_[in->place_]);
    }
  }

  std::vector<int64_t> out_rows;
  std::vector<Tensor> in_tensors;
  std::vector<platform::Place> in_places;

  auto &pre_in = pre_in_var->Get<framework::SelectedRows>();
  // gather the inputs
  for (auto *in : in_var_handles) {
    auto in_handle = static_cast<VarHandle *>(in);
    auto in_p = in_handle->place_;
    in_places.push_back(in_p);
    PADDLE_ENFORCE_EQ(in_p.which(), pre_place.which(),
                      "Places must be all on CPU or all on CUDA.");
    auto in_var =
        local_scopes_.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
    auto &in_sr = in_var->Get<framework::SelectedRows>();

    PADDLE_ENFORCE_EQ(in_sr.value().type(), pre_in.value().type(),
                      "The type of input is not consistent.");
    PADDLE_ENFORCE_EQ(pre_in.height(), in_sr.height(),
                      "The height of inputs is not consistent.");
    PADDLE_ENFORCE_EQ(pre_in.GetCompleteDims(), in_sr.GetCompleteDims(), ,
                      "The dims of inputs is not consistent.");

    auto in_sr_rows = in_sr.rows();
    out_rows.insert(out_rows.end(), in_sr_rows.begin(), in_sr_rows.end());

    in_tensors.emplace_back(in_sr.value());
  }

  // write the output
  auto &out_place = out_var_handles[0]->place_;
  auto out_scope_idx = out_var_handles[0]->scope_idx_;
  auto out_var =
      local_scopes_[out_scope_idx]->FindVar(out_var_handles[0]->name_);

  auto out = out_var->GetMutable<framework::SelectedRows>();
  out->set_height(pre_in.height());
  out->set_rows(out_rows);
  size_t rows = out_rows.size();
  DDim out_dim = pre_in.GetCompleteDims();
  out_dim[0] = static_cast<int64_t>(rows);
  out->mutable_value()->Resize(out_dim);
  out->mutable_value()->mutable_data(out_place, pre_in.value().type());
  Tensor *out_tensor = out->mutable_value();

  // copy
  int s = 0, e = 0;
  for (size_t j = 0; j < in_tensors.size(); ++j) {
    e += in_tensors[j].dims()[0];
    auto sub_out = out_tensor->Slice(s, e);
    paddle::framework::TensorCopy(in_tensors[j], out_place,
                                  *(dev_ctxes_[in_places[j]]), &sub_out);
    s = e;
  }
}

std::string GatherOpHandle::Name() const { return "gather"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
