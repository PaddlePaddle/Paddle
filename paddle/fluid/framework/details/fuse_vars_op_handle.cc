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

#include "paddle/fluid/framework/details/fuse_vars_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

void FuseVarsOpHandle::RunImpl() {
  WaitInputVarGenerated(place_);

  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
  PADDLE_ENFORCE_EQ(inputs_numel_.size(), in_var_handles.size(), "");
  PADDLE_ENFORCE_EQ(out_var_handles.size(), 1);

  int64_t total_numel = 0;
  for (size_t i = 0; i < in_var_handles.size(); ++i) {
    auto in_name = in_var_handles[i]->name_;
    PADDLE_ENFORCE(this->inputs_numel_.count(in_name));
    auto numel = this->inputs_numel_.at(in_name);
    PADDLE_ENFORCE_GT(numel, 0);
    total_numel += numel;
  }
  auto scope = local_scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();

  auto out_var_handle = out_var_handles[0];
  auto out_var = scope->Var(out_var_handle->name_);

  auto out_tensor = out_var->GetMutable<LoDTensor>();
  out_tensor->Resize({total_numel}).mutable_data(this->place_, type_);

  int64_t s = 0;
  for (size_t i = 0; i < in_var_handles.size(); ++i) {
    auto in_name = in_var_handles[i]->name_;
    auto in_t = scope->Var(in_name)->GetMutable<LoDTensor>();
    auto numel = this->inputs_numel_.at(in_name);
    in_t->ShareDataWith(out_tensor->Slice(s, s + numel));
    s += numel;
  }
  this->RunAndRecordEvent([this] {});
}

std::string FuseVarsOpHandle::Name() const { return "fuse vars"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
