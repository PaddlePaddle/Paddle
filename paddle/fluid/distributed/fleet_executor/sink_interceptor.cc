// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  PADDLE_ENFORCE_GT(node_->max_run_times(), 0,
                    platform::errors::InvalidArgument(
                        "Sink interceptor must run at least one "
                        "times, but now max_run_times=%ld",
                        node_->max_run_times()));
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void SinkInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();
  PADDLE_ENFORCE(!upstream.empty(),
                 platform::errors::InvalidArgument(
                     "Sink interceptor should have upstream"));
  PADDLE_ENFORCE(downstream.empty(),
                 platform::errors::InvalidArgument(
                     "Sink interceptor should not have downstream"));
  for (auto up : upstream) {
    in_readys_.emplace(up.first, std::make_pair(up.second, 0));
  }
}

void SinkInterceptor::ReplyCompletedToUpStream() {
  for (auto& ins : in_readys_) {
    auto up_id = ins.first;
    auto ready_size = ins.second.second;
    ready_size -= 1;
    PADDLE_ENFORCE_GE(
        ready_size, 0,
        platform::errors::OutOfRange(
            "upstream=%lld ready_size must >= 0, but now got %lld", up_id,
            ready_size));
    ins.second.second = ready_size;
    VLOG(3) << "Sink Interceptor " << interceptor_id_
            << " Reply data_is_useless msg to " << up_id << " step is "
            << step_;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
    Send(up_id, reply_msg);
  }
}

void SinkInterceptor::Run() {
  while (IsInputReady()) {
    VLOG(3) << "id=" << GetInterceptorId() << " SinkInterceptor running";
    // send to upstream and decrease buff used
    ReplyCompletedToUpStream();
    step_++;
    if (step_ % node_->max_run_times() == 0) {
      VLOG(3) << "Sink Interceptor is stopping carrier";
      StopCarrier();
    }
  }
}

void SinkInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    IncreaseReady(msg.src_id());
    Run();
  }
}

REGISTER_INTERCEPTOR(Sink, SinkInterceptor);

}  // namespace distributed
}  // namespace paddle
