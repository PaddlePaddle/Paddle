// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/start_interceptor.h"

#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

StartInterceptor::StartInterceptor(int64_t interceptor_id, TaskNode* node)
    : ComputeInterceptor(interceptor_id, node) {}

void StartInterceptor::RunOps() {
  finish_count_++;
  ComputeInterceptor::RunOps();
}

void StartInterceptor::SendDataReadyToDownStream() {
  for (auto& outs : out_buffs_) {
    auto down_id = outs.first;
    auto max_buff_size = outs.second.first;
    auto used_size = outs.second.second;
    used_size += 1;
    if (max_buff_size != INFINITE_BUFFER_SIZE) {
      PADDLE_ENFORCE_LE(
          used_size,
          max_buff_size,
          platform::errors::OutOfRange("downstream=%lld used buff size must <= "
                                       "max_buff_size, but now used_size=%lld, "
                                       "max_buff_size=%lld",
                                       down_id,
                                       used_size,
                                       max_buff_size));
    }
    outs.second.second = used_size;
  }
  const auto& micro_scope_nums = node_->max_run_times();
  if (finish_count_ == micro_scope_nums) {
    for (int64_t i = 0; i < micro_scope_nums; ++i) {
      for (auto& outs : out_buffs_) {
        auto down_id = outs.first;
        InterceptorMessage ready_msg;
        ready_msg.set_message_type(DATA_IS_READY);
        ready_msg.set_scope_idx(i);
        VLOG(3) << "StartInterceptor " << interceptor_id_
                << " Send data_is_ready msg to " << down_id
                << " in scope: " << i;
        Send(down_id, ready_msg);
      }
    }
  }
}

void StartInterceptor::ReplyCompletedToUpStream() {
  ComputeInterceptor::ReplyCompletedToUpStream();
}

void StartInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    VLOG(3) << "Start interceptor " << interceptor_id_
            << " receive data_is_ready " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    IncreaseReady(msg.src_id(), msg.scope_idx());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    finish_count_--;
    if (finish_count_ == 0) {
      const auto& micro_scope_nums = node_->max_run_times();
      for (int64_t i = 0; i < micro_scope_nums; ++i) {
        for (auto& ins : in_readys_) {
          auto up_id = ins.first;
          InterceptorMessage reply_msg;
          reply_msg.set_message_type(DATA_IS_USELESS);
          reply_msg.set_scope_idx(i);
          VLOG(3) << "StartInterceptor " << interceptor_id_
                  << " Send data_is_useless msg to " << up_id
                  << " in scope: " << i;
          Send(up_id, reply_msg);
        }
      }
    }
  }
}

REGISTER_INTERCEPTOR(Start, StartInterceptor);

}  // namespace distributed
}  // namespace paddle
