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

#include "paddle/fluid/distributed/fleet_executor/sink_interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

SinkInterceptor::SinkInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node), max_run_times_(node->max_run_times()) {
  // prepare the upstream running status
  for (const auto& up : node->upstream()) {
    upstream_step_.emplace(up.first, 0);
  }
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Run(msg); });
}

void SinkInterceptor::StopCarrierIfComplete() {
  bool flag = true;
  for (const auto& up : upstream_step_) {
    flag = flag && (up.second == max_run_times_);
  }
  if (flag) {
    VLOG(3) << "Sink Interceptor is stopping carrier";
    StopCarrier();
    for (const auto& up : upstream_step_) {
      upstream_step_.at(up.first) = 0;
    }
  }
}

void SinkInterceptor::ReplyCompletedToUpStream(int64_t upstream_id) {
  int64_t micro_step = upstream_step_.at(upstream_id);
  int64_t scope_idx = micro_step % max_run_times_;
  InterceptorMessage msg;
  msg.set_message_type(DATA_IS_USELESS);
  msg.set_scope_idx(scope_idx);
  Send(upstream_id, msg);
  upstream_step_.at(upstream_id) = micro_step + 1;
  if (micro_step == max_run_times_ - 1) {
    StopCarrierIfComplete();
  }
}

void SinkInterceptor::Run(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    ReplyCompletedToUpStream(msg.src_id());
  }
}

REGISTER_INTERCEPTOR(Sink, SinkInterceptor);
}  // namespace distributed
}  // namespace paddle
