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
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

Interceptor::Interceptor(int64_t interceptor_id, TaskNode* node)
    : interceptor_id_(interceptor_id), node_(node) {
  // TODO(Yuang Liu) init number_of_micro_steps_ and number_of_slot_ from node_
  // current init is faking 1F1B schedule
  number_of_micro_steps_ = 8;
  number_of_slot_ = 4;
  for (int64_t i = 0; i < number_of_micro_steps_; ++i) {
    std::unordered_map<int64_t, bool> scope_flag;
    for (auto& upstream_id : node_->upstream()) {
      scope_flag.insert({upstream_id, false});
    }
    upstream_flag_.insert({i, scope_flag});
  }
  interceptor_thread_ = std::thread([this]() {
    VLOG(3) << "Start pooling local mailbox's thread.";
    PoolTheMailbox();
  });
}

Interceptor::~Interceptor() { interceptor_thread_.join(); }

void Interceptor::RegisterMsgHandle(MsgHandle handle) { handle_ = handle; }

void Interceptor::Handle(const InterceptorMessage& msg) {
  if (handle_) {
    handle_(msg);
  } else {
    // default handler, for fake run only
    VLOG(3) << "Interceptor is using default msg handler, the default msg"
               " handler is only used for test purpose.";
    switch (msg.message_type()) {
      case DATA_IS_READY: {
        if (current_step_ == number_of_micro_steps_) {
          PADDLE_THROW(platform::errors::PreconditionNotMet(
              "Current interceptor has already finish all micro steps, but "
              "still got a DATA_IS_READY msg."));
        }
        if (slot_has_been_used_ == number_of_slot_) {
          Carrier& carrier_instance = Carrier::Instance();
          carrier_instance.EnqueueInterceptorMessage(msg);
        } else {
          upstream_flag_[msg.scope_id()][msg.src_id()] = true;
          bool all_set = true;
          for (const auto& pair : upstream_flag_[msg.scope_id()]) {
            all_set = all_set && pair.second;
          }
          if (all_set) {
            MessageBus& message_bus_instance = MessageBus::Instance();
            InterceptorMessage new_msg;
            new_msg.set_src_id(interceptor_id_);
            new_msg.set_message_type(DATA_IS_READY);
            new_msg.set_scope_id(current_step_);
            for (int64_t dst_id : node_->downstream()) {
              msg.set_dst_id(dst_id);
              message_bus_instance.Send(new_msg);
            }
            ++current_step_;
            ++slot_has_been_used_;
          }
        }
        break;
      }
      case DATE_IS_USELESS: {
        --slot_has_been_used_;
        bool all_set = true;
        for (const auto& pair : upstream_flag_[msg.scope_id()]) {
          all_set = all_set && pair.second;
        }
        if (all_set) {
          MessageBus& message_bus_instance = MessageBus::Instance();
          InterceptorMessage new_msg;
          new_msg.set_src_id(interceptor_id_);
          new_msg.set_message_type(DATA_IS_READY);
          new_msg.set_scope_id(current_step_);
          for (int64_t dst_id : node_->downstream()) {
            msg.set_dst_id(dst_id);
            message_bus_instance.Send(new_msg);
          }
          ++current_step_;
          ++slot_has_been_used_;
        }
        break;
      }
      default:
        VLOG(3) << "Default msg handler will only handle DATA_IS_READY and "
                   "DATA_IS_USELESS msg";
        break;
    }
  }
}

std::condition_variable& Interceptor::GetCondVar() {
  // get the conditional var
  return cond_var_;
}

int64_t Interceptor::GetInterceptorId() const {
  // return the interceptor id
  return interceptor_id_;
}

bool Interceptor::EnqueueRemoteInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  VLOG(3) << "Enqueue message: " << interceptor_message.message_type()
          << " into " << interceptor_id_ << "'s remote mailbox.";
  std::unique_lock<std::mutex> lock(remote_mailbox_mutex_);
  remote_mailbox_.push(interceptor_message);
  return true;
}

bool Interceptor::Send(int64_t dst_id, InterceptorMessage& msg) {
  msg.set_src_id(interceptor_id_);
  msg.set_dst_id(dst_id);
  return MessageBus::Instance().Send(msg);
}

void Interceptor::PoolTheMailbox() {
  // pool the local mailbox, parse the Message
  while (true) {
    if (local_mailbox_.empty()) {
      // local mailbox is empty, fetch the remote mailbox
      VLOG(3) << interceptor_id_ << "'s local mailbox is empty. "
              << "Fetch the remote mailbox.";
      PADDLE_ENFORCE_EQ(FetchRemoteMailbox(), true,
                        platform::errors::InvalidArgument(
                            "Error encountered when fetch remote mailbox."));
    }
    const InterceptorMessage interceptor_message = local_mailbox_.front();
    local_mailbox_.pop();
    const MessageType message_type = interceptor_message.message_type();
    VLOG(3) << "Interceptor " << interceptor_id_ << " has received a message"
            << " from interceptor " << interceptor_message.src_id()
            << " with message: " << message_type << ".";
    if (message_type == STOP) {
      // break the pooling thread
      VLOG(3) << "Interceptor " << interceptor_id_ << " is quiting.";
      break;
    }

    Handle(interceptor_message);
  }
}

bool Interceptor::FetchRemoteMailbox() {
  // fetch all Message from remote mailbox to local mailbox
  // return true if remote mailbox not empty, otherwise return false
  std::unique_lock<std::mutex> lock(remote_mailbox_mutex_);
  cond_var_.wait(lock, [this]() { return !remote_mailbox_.empty(); });
  if (remote_mailbox_.empty()) {
    // the thread has been unblocked accidentally
    return false;
  }
  while (!remote_mailbox_.empty()) {
    local_mailbox_.push(std::move(remote_mailbox_.front()));
    remote_mailbox_.pop();
  }
  return true;
}

}  // namespace distributed
}  // namespace paddle
