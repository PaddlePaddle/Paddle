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
#include "paddle/fluid/distributed/fleet_executor/carrier.h"

#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

SourceInterceptor::SourceInterceptor(int64_t interceptor_id, TaskNode* node)
    : ComputeInterceptor(interceptor_id, node),
      max_run_times_(node_->max_run_times()) {
  for (const auto& down : node->downstream()) {
    downstream_flag_.emplace(down.first, false);
  }
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void SourceInterceptor::SendDataReadyToDownStream() {
  for (const auto& down : downstream_flag_) {
    auto downstream_id = down.first;
    InterceptorMessage ready_msg;
    ready_msg.set_message_type(DATA_IS_READY);
    ready_msg.set_src_id(interceptor_id_);
    ready_msg.set_dst_id(downstream_id);
    ready_msg.set_scope_idx(step_);
    for (auto& carrier : multi_carriers_) {
      if (carrier->HasInterceptor(downstream_id)) {
        VLOG(3) << "Carrier send data ready to " << downstream_id;
        carrier->Send(ready_msg);
        break;
      }
    }
  }
}

void SourceInterceptor::Run() {
  if (cur_scope_id_ >= max_run_times_) {
    VLOG(3) << "SourceInterceptor " << interceptor_id_
            << " has run max_run_times=" << max_run_times_;
    return;
  }
  // Read num_of_carrier's data.
  for (size_t i = 0; i < multi_carriers_.size(); ++i) {
    RunOps();
    cur_scope_id_++;
  }
  SendDataReadyToDownStream();
  step_++;
}

bool SourceInterceptor::AllDownsFinished() {
  bool flag = true;
  for (const auto& down : downstream_flag_) {
    flag &= down.second;
  }
  return flag;
}

void SourceInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == START) {
    VLOG(3) << "SourceInterceptor " << interceptor_id_
            << " receiving start message";
    step_ = 0;
    cur_scope_id_ = 0;
    // Run and send data_is_ready for next scope to all downstreams.
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    // Set downstream flag.
    auto src_id = msg.src_id();
    PADDLE_ENFORCE_NE(
        downstream_flag_.find(src_id),
        downstream_flag_.end(),
        platform::errors::NotFound(
            "Cannot find downstream=%lld in downstream_flag_.", src_id));
    downstream_flag_[src_id] = true;
    // Run and send data_is_ready for next scope to all downstreams only if
    // all downstreams have sent the useless message
    if (AllDownsFinished()) {
      Run();
      // Reset downstream flag.
      for (const auto& down : downstream_flag_) {
        downstream_flag_[down.first] = false;
      }
    }
  }
}

REGISTER_INTERCEPTOR(Source, SourceInterceptor);
}  // namespace distributed
}  // namespace paddle
