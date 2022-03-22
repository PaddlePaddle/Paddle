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
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  PADDLE_ENFORCE_GT(node_->max_run_times(), 0,
                    platform::errors::InvalidArgument(
                        "Source interceptor must run at least one "
                        "times, but now max_run_times=%ld",
                        node_->max_run_times()));
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void SourceInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();
  PADDLE_ENFORCE(upstream.empty(),
                 platform::errors::InvalidArgument(
                     "Source interceptor should not have upstream"));
  PADDLE_ENFORCE(!downstream.empty(),
                 platform::errors::InvalidArgument(
                     "Source interceptor should have downstream"));
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }
}

void SourceInterceptor::SendDataReadyToDownStream() {
  for (auto& outs : out_buffs_) {
    auto down_id = outs.first;
    auto max_buff_size = outs.second.first;
    auto used_size = outs.second.second;
    used_size += 1;
    PADDLE_ENFORCE_LE(
        used_size, max_buff_size,
        platform::errors::OutOfRange("downstream=%lld used buff size must <= "
                                     "max_buff_size, but now used_size=%lld, "
                                     "max_buff_size=%lld",
                                     down_id, used_size, max_buff_size));
    outs.second.second = used_size;

    InterceptorMessage ready_msg;
    ready_msg.set_message_type(DATA_IS_READY);
    ready_msg.set_scope_idx(step_ % node_->max_run_times());
    VLOG(3) << "Source Interceptor " << interceptor_id_
            << " Send data_is_ready msg to " << down_id << " step is " << step_;
    Send(down_id, ready_msg);
  }
}

void SourceInterceptor::Run() {
  while (step_ < node_->max_run_times() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId() << " SourceInterceptor running";
    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    step_++;
  }
}

void SourceInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == START) {
    step_ = 0;
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    Run();
  }
}

REGISTER_INTERCEPTOR(Source, SourceInterceptor);

}  // namespace distributed
}  // namespace paddle
