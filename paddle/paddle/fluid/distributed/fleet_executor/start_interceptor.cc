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

#include "paddle/common/errors.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

StartInterceptor::StartInterceptor(int64_t interceptor_id, TaskNode* node)
    : ComputeInterceptor(interceptor_id, node) {
  auto& downstream = node_->downstream();
  PADDLE_ENFORCE_EQ(
      downstream.size(),
      1,
      platform::errors::OutOfRange(
          "The downstream for StartInterceptor only support 1 for now."));
  for (auto down : downstream) {
    batch_size_ = down.second;
  }
  bool evenly_divisible = ((node_->max_run_times() % batch_size_) == 0);
  PADDLE_ENFORCE(
      evenly_divisible,
      platform::errors::Fatal(
          "Wrong config: Num of step should be divided by batch_size,"
          "num_step=%lld, batch_size=%lld",
          node_->max_run_times(),
          batch_size_));
}

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
  if (finish_count_ == batch_size_) {
    int64_t start_micro_step = step_ % node_->max_run_times();
    for (int64_t i = 0; i < batch_size_; ++i) {
      int64_t scope_id = step_ % node_->max_run_times();
      InterceptorMessage ready_msg;
      ready_msg.set_message_type(DATA_IS_READY);
      ready_msg.set_scope_idx(scope_id);
      ready_msg.set_start_micro_step(start_micro_step);
      ready_msg.set_num_micro_step(batch_size_);
      for (auto& outs : out_buffs_) {
        auto down_id = outs.first;
        VLOG(3) << "StartInterceptor " << interceptor_id_
                << " Send data_is_ready msg to " << down_id
                << " in scope: " << scope_id;
        Send(down_id, ready_msg);
      }
      step_++;
    }
  }
}

void StartInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    VLOG(3) << "Start interceptor " << interceptor_id_
            << " receive data_is_ready " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    IncreaseReady(msg.src_id(), msg.scope_idx());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    VLOG(3) << "Start interceptor receive data_is_useless " << msg.src_id()
            << " " << finish_count_;
    finish_count_--;
    if (finish_count_ == 0) {
      auto end = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end - start_time_);
      VLOG(3) << "Spent "
              << double(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << " seconds.";
      start_time_ = std::chrono::system_clock::now();
      for (int64_t i = 0; i < batch_size_; ++i) {
        for (auto& outs : out_buffs_) {
          auto down_id = outs.first;
          DecreaseBuff(down_id);
        }
      }
      for (int64_t i = 0; i < batch_size_; ++i) {
        Run();
      }
    }
  }
}

REGISTER_INTERCEPTOR(Start, StartInterceptor);

}  // namespace distributed
}  // namespace paddle
