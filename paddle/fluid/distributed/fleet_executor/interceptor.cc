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

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/task_loop.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

Interceptor::Interceptor(int64_t interceptor_id, TaskNode* node)
    : interceptor_id_(interceptor_id), node_(node) {}

Interceptor::~Interceptor() {
  // FIXME(wangxi): throw in stop function
  // std::lock_guard<std::mutex> lock(mutex_);
  // PADDLE_ENFORCE_EQ(messages_.empty(), true,
  //                  platform::errors::PreconditionNotMet(
  //                      "Interceptor must destruct with messages empty"));
}

void Interceptor::RegisterMsgHandle(MsgHandle handle) { handle_ = handle; }

void Interceptor::Handle(const InterceptorMessage& msg) {
  PADDLE_ENFORCE_NOT_NULL(handle_, platform::errors::PreconditionNotMet(
                                       "Message handle is not registered."));
  handle_(msg);
}

void Interceptor::LoopOnce() {
  std::deque<InterceptorMessage> tmp_messages;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.swap(tmp_messages);
  }
  PADDLE_ENFORCE_EQ(tmp_messages.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "tmp_messages must not empty in task loop"));

  for (auto& msg : tmp_messages) {
    const MessageType message_type = msg.message_type();
    VLOG(3) << "Interceptor " << interceptor_id_ << " has received a message"
            << " from interceptor " << msg.src_id()
            << " with message: " << message_type << ".";

    Handle(msg);
  }
}

void Interceptor::StopCarrier() {
  PADDLE_ENFORCE_NOT_NULL(carrier_, platform::errors::PreconditionNotMet(
                                        "Carrier is not registered."));
  carrier_->WakeUp();
}

void Interceptor::EnqueueRemoteInterceptorMessage(
    const InterceptorMessage& message) {
  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  VLOG(3) << "Enqueue message: " << message.message_type() << " into "
          << interceptor_id_ << "'s remote mailbox.";

  bool empty = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    empty = messages_.empty();
    messages_.emplace_back(message);
  }
  if (empty) {
    loop_->QueueInLoop([this]() { LoopOnce(); });
  }
}

bool Interceptor::Send(int64_t dst_id, InterceptorMessage& msg) {
  PADDLE_ENFORCE_NOT_NULL(carrier_, platform::errors::PreconditionNotMet(
                                        "Carrier is not registered."));
  msg.set_src_id(interceptor_id_);
  msg.set_dst_id(dst_id);
  return carrier_->Send(msg);
}

void Interceptor::IncreaseReady(int64_t up_id) {
  auto it = in_readys_.find(up_id);
  PADDLE_ENFORCE_NE(it, in_readys_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_readys.", up_id));

  auto max_ready_size = it->second.first;
  auto ready_size = it->second.second;
  ready_size += 1;
  PADDLE_ENFORCE_LE(ready_size, max_ready_size,
                    platform::errors::OutOfRange(
                        "upstream=%lld ready_size must <= max_ready_size, but "
                        "now ready_size=%lld, max_ready_size=%lld",
                        up_id, ready_size, max_ready_size));
  it->second.second = ready_size;
}

void Interceptor::DecreaseBuff(int64_t down_id) {
  auto it = out_buffs_.find(down_id);
  PADDLE_ENFORCE_NE(it, out_buffs_.end(),
                    platform::errors::NotFound(
                        "Cannot find downstream=%lld in out_buffs.", down_id));
  auto used_size = it->second.second;
  used_size -= 1;
  PADDLE_ENFORCE_GE(
      used_size, 0,
      platform::errors::OutOfRange(
          "downstream=%lld used buff size must >= 0, but now equal %lld",
          down_id, used_size));
  it->second.second = used_size;
}

bool Interceptor::IsInputReady() {
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

bool Interceptor::CanWriteOutput() {
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

void Interceptor::SendDataReadyToDownStream() {
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
    ready_msg.set_scope_idx(scope_idx_);
    VLOG(3) << "Interceptor " << interceptor_id_
            << " Send data_is_ready msg to " << down_id << " scope index "
            << scope_idx_;
    Send(down_id, ready_msg);
  }
}

void Interceptor::ReplyCompletedToUpStream() {
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

    VLOG(3) << "Interceptor " << interceptor_id_
            << " Reply data_is_useless msg to " << up_id;
    if (up_id == -1) return;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
    Send(up_id, reply_msg);
  }
}

static InterceptorFactory::CreateInterceptorMap& GetInterceptorMap() {
  static InterceptorFactory::CreateInterceptorMap interceptorMap;
  return interceptorMap;
}

std::unique_ptr<Interceptor> InterceptorFactory::Create(const std::string& type,
                                                        int64_t id,
                                                        TaskNode* node) {
  auto& interceptor_map = GetInterceptorMap();
  auto iter = interceptor_map.find(type);
  PADDLE_ENFORCE_NE(
      iter, interceptor_map.end(),
      platform::errors::NotFound("interceptor %s is not register", type));
  return iter->second(id, node);
}

void InterceptorFactory::Register(
    const std::string& type, InterceptorFactory::CreateInterceptorFunc func) {
  auto& interceptor_map = GetInterceptorMap();
  interceptor_map.emplace(type, func);
}

}  // namespace distributed
}  // namespace paddle
