// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

ComputeInterceptor::ComputeInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void ComputeInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();

  for (auto up : upstream) {
    in_readys_.emplace(up.first, std::make_pair(up.second, 0));
  }
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }
}

void ComputeInterceptor::IncreaseReady(int64_t up_id) {
  auto it = in_readys_.find(up_id);
  PADDLE_ENFORCE_NE(it,
                    in_readys_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_readys.", up_id));

  auto max_ready_size = it->second.first;
  auto ready_size = it->second.second;
  ready_size += 1;
  if (max_ready_size != INFINITE_BUFFER_SIZE) {
    PADDLE_ENFORCE_LE(
        ready_size,
        max_ready_size,
        platform::errors::OutOfRange(
            "upstream=%lld ready_size must <= max_ready_size, but "
            "now ready_size=%lld, max_ready_size=%lld",
            up_id,
            ready_size,
            max_ready_size));
  }
  it->second.second = ready_size;
}

void ComputeInterceptor::DecreaseBuff(int64_t down_id) {
  auto it = out_buffs_.find(down_id);
  PADDLE_ENFORCE_NE(it,
                    out_buffs_.end(),
                    platform::errors::NotFound(
                        "Cannot find downstream=%lld in out_buffs.", down_id));
  auto used_size = it->second.second;
  used_size -= 1;
  PADDLE_ENFORCE_GE(
      used_size,
      0,
      platform::errors::OutOfRange(
          "downstream=%lld used buff size must >= 0, but now equal %lld",
          down_id,
          used_size));
  it->second.second = used_size;
}

bool ComputeInterceptor::IsInputReady() {
  for (auto& ins : in_readys_) {
    auto ready_size = ins.second.second;
    // not ready, return false
    if (ready_size == 0) {
      VLOG(3) << "Interceptor " << GetInterceptorId()
              << "'s upstreams aren't all ready.";
      return false;
    }
  }
  return true;
}

bool ComputeInterceptor::CanWriteOutput() {
  for (auto& outs : out_buffs_) {
    auto max_buffer_size = outs.second.first;
    auto used_size = outs.second.second;
    if (max_buffer_size == INFINITE_BUFFER_SIZE) {
      continue;
    }
    // full, return false
    if (used_size == max_buffer_size) {
      VLOG(3) << "Interceptor " << GetInterceptorId()
              << "'s out buffer is full.";
      return false;
    }
  }
  return true;
}

void ComputeInterceptor::SendDataReadyToDownStream() {
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

    InterceptorMessage ready_msg;
    ready_msg.set_message_type(DATA_IS_READY);
    ready_msg.set_scope_idx(cur_scope_id_);
    VLOG(3) << "ComputeInterceptor " << interceptor_id_
            << " Send data_is_ready msg to " << down_id
            << " in scope: " << cur_scope_id_;
    Send(down_id, ready_msg);
  }
}

void ComputeInterceptor::ReplyCompletedToUpStream() {
  for (auto& ins : in_readys_) {
    auto up_id = ins.first;
    auto ready_size = ins.second.second;
    ready_size -= 1;
    PADDLE_ENFORCE_GE(
        ready_size,
        0,
        platform::errors::OutOfRange(
            "upstream=%lld ready_size must >= 0, but now got %lld",
            up_id,
            ready_size));
    ins.second.second = ready_size;

    VLOG(3) << "ComputeInterceptor " << interceptor_id_
            << " Reply data_is_useless msg to " << up_id
            << " in scope: " << cur_scope_id_;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
    reply_msg.set_scope_idx(cur_scope_id_);
    Send(up_id, reply_msg);
  }
}

void ComputeInterceptor::RunOps() {
  for (auto op : node_->ops()) {
    PADDLE_ENFORCE_LT(cur_scope_id_,
                      microbatch_scopes_.size(),
                      platform::errors::InvalidArgument(
                          "Step out of range. There are %ld "
                          "microbatch_scopes, but recevice scope index %ld",
                          microbatch_scopes_.size(),
                          cur_scope_id_));
    op->Run(*microbatch_scopes_[cur_scope_id_], place_);
    if (gc_) {
      framework::DeleteUnusedTensors(*microbatch_scopes_[cur_scope_id_],
                                     op,
                                     node_->unused_vars(),
                                     gc_.get());
    }
  }
}

void ComputeInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId() << " ComputeInterceptor running";

    // get the ready scope id from queue
    cur_scope_id_ = ready_queue_.front();
    ready_queue_.pop();

    RunOps();

    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    // reply to upstream and decrease ready data
    ReplyCompletedToUpStream();
  }
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    IncreaseReady(msg.src_id());
    ready_queue_.push(msg.scope_idx());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    Run();
  }
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
