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

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

void Carrier::Init(
    const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node) {
  PADDLE_ENFORCE_EQ(is_init_, false, platform::errors::AlreadyExists(
                                         "Carrier is already init."));
  interceptor_id_to_node_ = interceptor_id_to_node;
  CreateInterceptors();
  is_init_ = true;
}

bool Carrier::EnqueueInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // enqueue message to interceptor
  if (interceptor_message.ctrl_message()) {
    // handle control message
    return true;
  } else {
    if (creating_interceptors_) {
      // Cannot handle the message to interceptor since interceptors
      // are still under creating. Will enqueue into a tmp stack.
      VLOG(3) << "Receiving message while creating interceptors.";
      message_tmp_.emplace_back(interceptor_message);
      return true;
    }
    int64_t dst_id = interceptor_message.dst_id();
    Interceptor* dst_interceptor = GetInterceptor(dst_id);
    bool rst =
        dst_interceptor->EnqueueRemoteInterceptorMessage(interceptor_message);
    if (rst) {
      std::condition_variable& interceptor_cond_var =
          dst_interceptor->GetCondVar();
      interceptor_cond_var.notify_all();
    }
    return rst;
  }
}

Interceptor* Carrier::GetInterceptor(int64_t interceptor_id) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_NE(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find interceptor instance for interceptor "
                        "id %lld. Wrong dst? Call before init?",
                        interceptor_id));
  return iter->second.get();
}

void Carrier::Start() {
  // TODO(fleet_executor dev): this start is a faked one, need replace
  for (const auto& pair : interceptor_idx_to_interceptor_) {
    VLOG(3) << "Fake run is sending start to interceptor " << pair.first << ".";
    InterceptorMessage tmp_msg;
    tmp_msg.set_src_id(pair.first);
    tmp_msg.set_dst_id(pair.first);
    tmp_msg.set_message_type(DATA_IS_READY);
    MessageBus& message_bus_instance = MessageBus::Instance();
    PADDLE_ENFORCE_EQ(message_bus_instance.IsInit(), true,
                      platform::errors::PreconditionNotMet(
                          "Message bus has not been initialized."));
    message_bus_instance.Send(tmp_msg);
  }
}

bool Carrier::IsInit() const { return is_init_; }

Interceptor* Carrier::SetInterceptor(int64_t interceptor_id,
                                     std::unique_ptr<Interceptor> interceptor) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_EQ(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::AlreadyExists(
                        "The interceptor id %lld has already been created! "
                        "The interceptor id should be unique.",
                        interceptor_id));
  auto* ptr = interceptor.get();
  interceptor_idx_to_interceptor_.insert(
      std::make_pair(interceptor_id, std::move(interceptor)));
  return ptr;
}

void Carrier::SetCreatingFlag(bool flag) {
  // set the creating flag
  VLOG(3) << "Carrier is set the creating flag from " << creating_interceptors_
          << " to " << flag << ".";
  creating_interceptors_ = flag;
  if (!flag) {
    // finish create interceptors outside, handle tmp messsages
    HandleTmpMessages();
  }
}

void Carrier::HandleTmpMessages() {
  VLOG(3) << "Carrier has received " << message_tmp_.size()
          << " messages during creating interceptors.";
  for (const auto& msg : message_tmp_) {
    EnqueueInterceptorMessage(msg);
  }
  message_tmp_.clear();
}

void Carrier::CreateInterceptors() {
  // create each Interceptor
  if (!interceptor_id_to_node_.empty()) {
    // no auto init since there is no config
    for (const auto& item : interceptor_id_to_node_) {
      int64_t interceptor_id = item.first;
      TaskNode* task_node = item.second;

      // TODO(wangxi): use node_type to select different Interceptor
      auto interceptor =
          std::make_unique<Interceptor>(interceptor_id, task_node);
      SetInterceptor(interceptor_id, std::move(interceptor));
      VLOG(3) << "Create Interceptor with interceptor id: " << interceptor_id
              << ".";
    }
    // The carrier will be always waiting for outside initializer
    // since there is no interceptor has been created during auto init
    creating_interceptors_ = false;
    HandleTmpMessages();
  }
}

}  // namespace distributed
}  // namespace paddle
