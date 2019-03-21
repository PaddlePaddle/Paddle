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

#include "paddle/fluid/framework/details/fused_broadcast_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

void FusedBroadcastOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  if (places_.size() == 1UL) return;

  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  WaitInputVarGenerated();

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    var_scopes.emplace_back(s->FindVar(kLocalExecScopeName)->Get<Scope *>());
  }

  size_t place_num = places_.size();
  PADDLE_ENFORCE_EQ(in_var_handles.size() * place_num, out_var_handles.size());

  for (size_t i = 0; i < in_var_handles.size(); ++i) {
    BroadcastOneVar(
        *in_var_handles[i],
        std::vector<VarHandle *>(out_var_handles.begin() + i * place_num,
                                 out_var_handles.begin() + (i + 1) * place_num),
        var_scopes);
  }
}

std::string FusedBroadcastOpHandle::Name() const { return "fused_broadcast"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
