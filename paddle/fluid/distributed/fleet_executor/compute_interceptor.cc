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
<<<<<<< HEAD
=======
    in_stops_.emplace(up.first, false);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }
<<<<<<< HEAD
=======

  // source compute node, should we add a new SourceInterceptor?
  if (upstream.empty()) {
    is_source_ = true;
    PADDLE_ENFORCE_GT(node_->max_run_times(),
                      0,
                      platform::errors::InvalidArgument(
                          "Source ComputeInterceptor must run at least one "
                          "times, but now max_run_times=%ld",
                          node_->max_run_times()));
    in_readys_.emplace(-1,
                       std::make_pair(std::numeric_limits<int64_t>::max(), 0));
  }

  // If there is no downstream or every downstream is in different rank,
  // then this interceptor is the last one for current rank.
  // This can be get during init, can be cached for later use.
  is_last_ = downstream.empty();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

void ComputeInterceptor::IncreaseReady(int64_t up_id) {
  auto it = in_readys_.find(up_id);
  PADDLE_ENFORCE_NE(it,
                    in_readys_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_readys.", up_id));

<<<<<<< HEAD
=======
  // source node has no upstream, data_is_ready is send by carrier or others
  if (is_source_ && up_id == -1) {
    it->second.second += GetTaskNode()->max_run_times();
    return;
  }

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto max_ready_size = it->second.first;
  auto ready_size = it->second.second;
  ready_size += 1;
  PADDLE_ENFORCE_LE(ready_size,
                    max_ready_size,
                    platform::errors::OutOfRange(
                        "upstream=%lld ready_size must <= max_ready_size, but "
                        "now ready_size=%lld, max_ready_size=%lld",
                        up_id,
                        ready_size,
                        max_ready_size));
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
    PADDLE_ENFORCE_LE(
        used_size,
        max_buff_size,
        platform::errors::OutOfRange("downstream=%lld used buff size must <= "
                                     "max_buff_size, but now used_size=%lld, "
                                     "max_buff_size=%lld",
                                     down_id,
                                     used_size,
                                     max_buff_size));
    outs.second.second = used_size;

    InterceptorMessage ready_msg;
    ready_msg.set_message_type(DATA_IS_READY);
<<<<<<< HEAD
    ready_msg.set_scope_idx(cur_scope_id_);
    VLOG(3) << "ComputeInterceptor " << interceptor_id_
            << " Send data_is_ready msg to " << down_id
            << " in scope: " << cur_scope_id_;
=======
    VLOG(3) << "ComputeInterceptor " << interceptor_id_
            << " Send data_is_ready msg to " << down_id
            << " for step: " << step_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
            << " in scope: " << cur_scope_id_;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
    reply_msg.set_scope_idx(cur_scope_id_);
=======
            << " for step: " << step_;
    if (is_source_ && up_id == -1) return;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    Send(up_id, reply_msg);
  }
}

void ComputeInterceptor::RunOps() {
<<<<<<< HEAD
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
=======
  VLOG(3) << "ComputeInterceptor " << interceptor_id_ << " running ops for the "
          << step_ + 1 << " time.";
  for (auto op : node_->ops()) {
    op->Run(*microbatch_scopes_[step_ % node_->max_run_times()], place_);
    if (gc_) {
      framework::DeleteUnusedTensors(
          *microbatch_scopes_[step_ % node_->max_run_times()],
          op,
          node_->unused_vars(),
          gc_.get());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
  }
}

void ComputeInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId() << " ComputeInterceptor running";

<<<<<<< HEAD
    // get the ready scope id from queue
    cur_scope_id_ = ready_queue_.front();
    ready_queue_.pop();

    RunOps();
=======
    RunOps();
    ++step_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    // reply to upstream and decrease ready data
    ReplyCompletedToUpStream();
<<<<<<< HEAD
  }
=======
    // Try to stop Carrier
    if (is_last_ && (step_ % node_->max_run_times() == 0)) {
      VLOG(3) << "Interceptor " << GetInterceptorId()
              << " is stopping carrier.";
      // FIXME(wangxi): with multi sink interceptor
      StopCarrier();
    }
  }
}

void ComputeInterceptor::ReceivedStop(int64_t up_id) {
  received_stop_ = true;

  // source node has no upstream, stop is send by carrier or others
  if (is_source_ && up_id == -1) return;

  auto it = in_stops_.find(up_id);
  PADDLE_ENFORCE_NE(it,
                    in_stops_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_stops.", up_id));
  PADDLE_ENFORCE_EQ(
      it->second,
      false,
      platform::errors::AlreadyExists("Already received stop from %lld, stop "
                                      "cannot be send more than once."));
  it->second = true;
}

void ComputeInterceptor::TryStop() {
  if (!received_stop_) return;

  // can stop only when all upstream is stop and
  // downstream complete
  for (auto& in_stop : in_stops_) {
    if (!in_stop.second) return;
  }
  for (auto& out_buff : out_buffs_) {
    auto used_size = out_buff.second.second;
    if (used_size != 0) return;
  }

  // send stop to downstream
  for (auto& out : out_buffs_) {
    auto down_id = out.first;
    InterceptorMessage stop;
    stop.set_message_type(STOP);
    Send(down_id, stop);
  }
  stop_ = true;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    IncreaseReady(msg.src_id());
<<<<<<< HEAD
    ready_queue_.push(msg.scope_idx());
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    Run();
<<<<<<< HEAD
  }
=======
  } else if (msg.message_type() == STOP) {
    ReceivedStop(msg.src_id());
  }

  TryStop();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
