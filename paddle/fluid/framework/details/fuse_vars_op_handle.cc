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
  PADDLE_ENFORCE_EQ(in_var_handles.size(), 0);
  PADDLE_ENFORCE_EQ(out_var_handles.size() - 1, inputs_numel_.size(), "");

  auto scope = local_scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();

  auto out_var_handle = out_var_handles[0];
  auto out_var = scope->Var(out_var_handle->name_);

  auto out_tensor = out_var->GetMutable<LoDTensor>();
  out_tensor->Resize({total_numel_}).mutable_data(this->place_, type_);

  int64_t s = 0;
  for (size_t i = 1; i < out_var_handles.size(); ++i) {
    auto out_name = out_var_handles[i]->name_;
    auto out_t = scope->Var(out_name)->GetMutable<LoDTensor>();
    auto numel = this->inputs_numel_.at(out_name);
    out_t->ShareDataWith(out_tensor->Slice(s, s + numel));
    s += numel;
  }
  this->RunAndRecordEvent([] {});
}

std::string FuseVarsOpHandle::Name() const { return "fuse vars"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
