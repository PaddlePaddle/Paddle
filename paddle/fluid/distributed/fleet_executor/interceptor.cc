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
