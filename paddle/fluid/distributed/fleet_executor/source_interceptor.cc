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
    : ComputeInterceptor(interceptor_id, node) {
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void SourceInterceptor::SendDataReadyToDownStream() {
  for (size_t carrier_id = 0; multi_carriers_.size(); ++carrier_id) {
    auto carrier = multi_carriers_[carrier_id];
    for (const auto& down : node_->downstream()) {
      if (carrier->HasInterceptor(down.first)) {
        auto scopes_ready_ = carrier_scope_ids_[carrier_id];
        auto cur_scope_ready_ = scopes_ready_[step_];
        InterceptorMessage ready_msg;
        ready_msg.set_message_type(DATA_IS_READY);
        ready_msg.set_scope_idx(cur_scope_ready_.first);
        ready_msg.set_src_id(interceptor_id_);
        ready_msg.set_dst_id(down.first);
        carrier->Send(ready_msg);
      }
    }
  }
}

void SourceInterceptor::Run() {
  bool flag = true;
  for (const auto scopes_ : carrier_scope_ids_) {
    flag = flag && scopes_[step_].second;
  }

  if (flag) {
    for (const auto scopes_ : carrier_scope_ids_) {
      cur_scope_id_ = scopes_[step_].first;
      VLOG(0) << "cur_scope_id_:" << cur_scope_id_;
      RunOps();
    }
    SendDataReadyToDownStream();
    step_++;
  } else {
    VLOG(3) << "Interceptor " << GetInterceptorId() << " in step " << step_
            << " aren't all ready.";
  }
}

void SourceInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == START) {
    // start run in a new step, reset the previous running status
    VLOG(3) << "SourceInterceptor " << interceptor_id_
            << " receiving start message";
    step_ = 0;
    for (auto& scopes_ : carrier_scope_ids_) {
      scopes_[step_].second = true;
    }
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    VLOG(3) << "Reader interceptor " << interceptor_id_
            << " receive data_is_useless " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    for (size_t carrier_id = 0; multi_carriers_.size(); ++carrier_id) {
      auto carrier = multi_carriers_[carrier_id];
      if (carrier->HasInterceptor(msg.src_id())) {
        auto src_scopes_ready_ = carrier_scope_ids_[carrier_id];
        auto src_scope_ready_ = src_scopes_ready_[step_];
        PADDLE_ENFORCE_EQ(src_scope_ready_.first,
                          msg.scope_idx(),
                          platform::errors::PreconditionNotMet("src_scope"));
        PADDLE_ENFORCE_EQ(src_scope_ready_.second,
                          false,
                          platform::errors::PreconditionNotMet(
                              "Interceptor must destruct with messages empty"));
        src_scope_ready_.second = true;
      }
    }
    Run();
  }
}

REGISTER_INTERCEPTOR(Source, SourceInterceptor);
}  // namespace distributed
}  // namespace paddle
