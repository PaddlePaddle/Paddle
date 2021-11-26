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

#include <chrono>
#include <thread>

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

Interceptor::Interceptor(int64_t interceptor_id, TaskNode* node)
    : interceptor_id_(interceptor_id), node_(node) {
  LOG(INFO) << "Interceptor " << interceptor_id_
            << "'s task role is: " << node_->role() << ".";
  LOG(INFO) << "Interceptor " << interceptor_id_
            << "'s max_run_times is: " << node_->max_run_times() << ".";
  LOG(INFO) << "Interceptor " << interceptor_id_
            << "'s max_slot_nums is: " << node_->max_slot_nums() << ".";
  for (int64_t upstream : node_->upstream()) {
    LOG(INFO) << "Interceptor " << interceptor_id_
              << "'s upstream has: " << upstream << ".";
    std::vector<bool> tmp_value(node_->max_run_times(), false);
    ready_flags_.insert({upstream, tmp_value});
  }
  for (int64_t downstream : node_->downstream()) {
    LOG(INFO) << "Interceptor " << interceptor_id_
              << "'s downstream has: " << downstream << ".";
  }
  interceptor_thread_ = std::thread([this]() {
    VLOG(3) << "Interceptor " << interceptor_id_
            << " starts the thread pooling it's local mailbox.";
    PoolTheMailbox();
  });
}

Interceptor::~Interceptor() { Join(); }

void Interceptor::Join() {
  if (interceptor_thread_.joinable()) {
    interceptor_thread_.join();
  }
}

void Interceptor::RegisterMsgHandle(MsgHandle handle) { handle_ = handle; }

void Interceptor::Handle(const InterceptorMessage& msg) {
  if (handle_) {
    handle_(msg);
  } else {
    // This default handler is faking 1F1B scheduler from section worker.
    LOG(WARNING)
        << "Interceptor is using default message handler. This handler is "
           "only used for test purpose. Check whether you init interceptor "
           "in the proper way. All log will be shown under this handler.";
    if (msg.message_type() == RESET) {
      LOG(INFO) << "Fake handler is resetting the interceptor's status.";
      already_run_times_ = 0;
      used_slot_nums_ = 0;
      for (int64_t upstream : node_->upstream()) {
        for (int i = 0; i < node_->max_run_times(); ++i) {
          ready_flags_.at(upstream)[i] = false;
        }
      }
    } else if (msg.message_type() == DATA_IS_READY) {
      LOG(INFO) << "Interceptor " << interceptor_id_
                << " receives DATA_IS_READY from " << msg.src_id() << ".";
      if (node_->upstream().size() != 0) {
        for (int i = 0; i < node_->max_run_times(); ++i) {
          if (ready_flags_.at(msg.src_id())[i] == false) {
            ready_flags_.at(msg.src_id())[i] = true;
            if (msg.src_id() % 4 != 0) {
              break;
            }
          }
        }
      }
      bool can_run = true;
      if (node_->upstream().size() != 0) {
        for (int64_t upstream : node_->upstream()) {
          can_run = can_run && ready_flags_.at(upstream)[already_run_times_];
        }
      }
      if (!can_run) {
        LOG(INFO) << "Interceptor " << interceptor_id_
                  << " doesn't have empty slot or not every upstream is ready,"
                     " it will store the DATA_IS_READY info.";
      }
      while (can_run &&
             (node_->role() == 1 || used_slot_nums_ < node_->max_slot_nums()) &&
             already_run_times_ < node_->max_run_times()) {
        used_slot_nums_++;
        already_run_times_++;
        LOG(INFO) << "Interceptor " << interceptor_id_
                  << " is running op for the " << already_run_times_
                  << " times.";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        for (int64_t downstream : node_->downstream()) {
          if (node_->role() == 1 &&
              already_run_times_ < node_->max_run_times() &&
              downstream == interceptor_id_ + 1) {
            LOG(INFO) << "Interceptor " << interceptor_id_
                      << " won't send DATA_IS_READY to it's downstream: "
                      << downstream << " since it's not the time for opt.";
            continue;
          }
          LOG(INFO) << "Interceptor " << interceptor_id_
                    << " sends DATA_IS_READY to it's downstream: " << downstream
                    << ".";
          InterceptorMessage tmp_msg;
          tmp_msg.set_message_type(DATA_IS_READY);
          tmp_msg.set_dst_id(downstream);
          while (!Send(downstream, tmp_msg)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
          }
        }
        if (node_->role() == 1) {
          for (int64_t upstream : node_->upstream()) {
            LOG(INFO) << "Interceptor " << interceptor_id_
                      << " sends DATE_IS_USELESS to it's upstream: " << upstream
                      << ".";
            InterceptorMessage tmp_msg;
            tmp_msg.set_message_type(DATE_IS_USELESS);
            tmp_msg.set_dst_id(upstream);
            while (!Send(upstream, tmp_msg)) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
          }
        } else if (node_->role() == 2) {
          Carrier& carrier_instance = Carrier::Instance();
          //          carrier_instance.status = true;
          std::mutex& mu = carrier_instance.mu;
          mu.lock();
          std::condition_variable& cond_var = carrier_instance.cond_var;
          cond_var.notify_all();
          mu.unlock();
        }
        if (node_->upstream().size() != 0) {
          for (int64_t upstream : node_->upstream()) {
            can_run = can_run && ready_flags_.at(upstream)[already_run_times_];
          }
        }
        if (!can_run) {
          LOG(INFO) << "Interceptor " << interceptor_id_
                    << " will stop since max slot or max run time is reached.";
        }
      }
    } else if (msg.message_type() == DATE_IS_USELESS) {
      LOG(INFO) << "Interceptor " << interceptor_id_
                << " receives DATA_IS_USELESS from " << msg.src_id() << ".";
      used_slot_nums_ -= 1;
      bool can_run = true;
      if (node_->upstream().size() != 0) {
        for (int64_t upstream : node_->upstream()) {
          can_run = can_run && ready_flags_.at(upstream)[already_run_times_];
        }
      }
      if (!can_run) {
        LOG(INFO) << "Interceptor " << interceptor_id_
                  << " doesn't have empty slot or not every upstream is ready,"
                     " it will store the DATA_IS_READY info.";
      }
      while (can_run &&
             (node_->role() == 1 || used_slot_nums_ < node_->max_slot_nums()) &&
             already_run_times_ < node_->max_run_times()) {
        already_run_times_++;
        used_slot_nums_++;
        LOG(INFO) << "Interceptor " << interceptor_id_ << " runs for the "
                  << already_run_times_
                  << " times since all upstreams are ready.";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        for (int64_t downstream : node_->downstream()) {
          if (node_->role() == 1 &&
              already_run_times_ < node_->max_run_times() &&
              downstream == interceptor_id_ + 1) {
            LOG(INFO) << "Interceptor " << interceptor_id_
                      << " won't send DATA_IS_READY to it's downstream: "
                      << downstream << " since it's not the time for opt.";
            continue;
          }
          LOG(INFO) << "Interceptor " << interceptor_id_
                    << " sends DATA_IS_READY to it's downstream: " << downstream
                    << ".";
          InterceptorMessage tmp_msg;
          tmp_msg.set_message_type(DATA_IS_READY);
          tmp_msg.set_dst_id(downstream);
          while (!Send(downstream, tmp_msg)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
          }
        }
        if (node_->role() == 1) {
          for (int64_t upstream : node_->upstream()) {
            LOG(INFO) << "Interceptor " << interceptor_id_
                      << " sends DATE_IS_USELESS to it's upstream: " << upstream
                      << ".";
            InterceptorMessage tmp_msg;
            tmp_msg.set_message_type(DATE_IS_USELESS);
            tmp_msg.set_dst_id(upstream);
            while (!Send(upstream, tmp_msg)) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
          }
        } else if (node_->role() == 2) {
          Carrier& carrier_instance = Carrier::Instance();
          //          carrier_instance.status = true;
          std::mutex& mu = carrier_instance.mu;
          mu.lock();
          std::condition_variable& cond_var = carrier_instance.cond_var;
          cond_var.notify_all();
          mu.unlock();
        }
        if (node_->upstream().size() != 0) {
          for (int64_t upstream : node_->upstream()) {
            can_run = can_run && ready_flags_.at(upstream)[already_run_times_];
          }
        }
        if (!can_run) {
          LOG(INFO) << "Interceptor " << interceptor_id_
                    << " will stop since max slot or max run time is reached.";
        }
      }
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

// maybe need a better method for interceptor base
void Interceptor::HandleStop(const InterceptorMessage& msg) { stop_ = true; }

void Interceptor::PoolTheMailbox() {
  // pool the local mailbox, parse the Message
  for (;;) {
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
      HandleStop(interceptor_message);
    } else {
      Handle(interceptor_message);
    }

    if (stop_) {
      // break the pooling thread
      VLOG(3) << "Interceptor " << interceptor_id_ << " is quiting.";
      break;
    }
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
