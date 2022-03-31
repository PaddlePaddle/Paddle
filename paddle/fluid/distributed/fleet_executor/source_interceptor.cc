// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/source_interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

SourceInterceptor::SourceInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node), max_run_times_(node->max_run_times()) {
  // prepare the downstream running status
  for (const auto& down : node->downstream()) {
    downstream_step_.emplace(down.first, 0);
  }
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Run(msg); });
}

void SourceInterceptor::SendDataReadyToDownStream(int64_t downstream_id) {
  int64_t micro_step = downstream_step_.at(downstream_id);
  if (micro_step >= max_run_times_) {
    return;
  }
  int64_t scope_idx = micro_step % max_run_times_;
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_READY);
  ready_msg.set_scope_idx(scope_idx);
  Send(downstream_id, ready_msg);
  downstream_step_.at(downstream_id) = micro_step + 1;
}

void SourceInterceptor::Run(const InterceptorMessage& msg) {
  if (msg.message_type() == START) {
    // start run in a new step, reset the previous running status
    for (const auto& down : downstream_step_) {
      downstream_step_.at(down.first) = 0;
      SendDataReadyToDownStream(down.first);
    }
  } else if (msg.message_type() == DATA_IS_USELESS) {
    SendDataReadyToDownStream(msg.src_id());
  }
}

REGISTER_INTERCEPTOR(Source, SourceInterceptor);
}  // namespace distributed
}  // namespace paddle
