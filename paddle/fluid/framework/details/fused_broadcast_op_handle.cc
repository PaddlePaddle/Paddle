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
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {
namespace details {

void FusedBroadcastOpHandle::RunImpl() {
  platform::RecordEvent record_event(
      Name(), platform::TracerEventType::Communication, 1);

  if (places_.size() == 1UL) return;

  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  WaitInputVarGenerated();

  size_t place_num = places_.size();
  PADDLE_ENFORCE_EQ(
      in_var_handles.size() * place_num, out_var_handles.size(),
      platform::errors::PreconditionNotMet(
          "The number of input variable handles plus the number "
          "of places should be equal to the number of output variable handles, "
          "but got the number of input variable handles is %d, the "
          "number of places is %d, and the number of output variable handles "
          "is %d.",
          in_var_handles.size(), place_num, out_var_handles.size()));

  for (size_t i = 0; i < in_var_handles.size(); ++i) {
    BroadcastOneVar(
        *in_var_handles[i],
        std::vector<VarHandle *>(out_var_handles.begin() + i * place_num,
                                 out_var_handles.begin() + (i + 1) * place_num),
        local_exec_scopes_);
  }
}

std::string FusedBroadcastOpHandle::Name() const { return "fused_broadcast"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
