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

#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

class TaskNode;

class Interceptor {
 public:
  using MsgHandle = std::function<void(const InterceptorMessage&)>;

 public:
  Interceptor() = delete;

  Interceptor(int64_t interceptor_id, TaskNode* node);

  virtual ~Interceptor();

  // register interceptor handle
  void RegisterMsgHandle(MsgHandle handle);

  void Handle(const InterceptorMessage& msg);

  // return the interceptor id
  int64_t GetInterceptorId() const;

  // return the conditional var
  std::condition_variable& GetCondVar();

  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  bool EnqueueRemoteInterceptorMessage(
      const InterceptorMessage& interceptor_message);

  bool Send(int64_t dst_id, InterceptorMessage& msg);  // NOLINT

  DISABLE_COPY_AND_ASSIGN(Interceptor);

 private:
  // pool the local mailbox, parse the Message
  void PoolTheMailbox();

  // fetch all Message from remote mailbox to local mailbox
  // return true if remote mailbox not empty, otherwise return false
  bool FetchRemoteMailbox();

  // interceptor id, handed from above layer
  int64_t interceptor_id_;

  // node need to be handled by this interceptor
  TaskNode* node_;

  // interceptor handle which process message
  MsgHandle handle_{nullptr};

  // mutex to control read/write conflict for remote mailbox
  std::mutex remote_mailbox_mutex_;

  // interceptor runs PoolTheMailbox() function to poll local mailbox
  std::thread interceptor_thread_;

  // conditional variable for blocking the thread when
  // fetch an empty remote mailbox
  std::condition_variable cond_var_;

  // remote mailbox, written by EnqueueRemoteMessage()
  // read by FetchRemoteMailbox()
  std::queue<InterceptorMessage> remote_mailbox_;

  // local mailbox, written by FetchRemoteMailbox()
  // read by PoolTheMailbox()
  std::queue<InterceptorMessage> local_mailbox_;
};

using CreateInterceptorFunc = std::unique_ptr<Interceptor>(int64_t, TaskNode*);
using CreateInterceptorMap =
    std::unordered_map<std::string, CreateInterceptorFunc>;

CreateInterceptorMap& GetCreateInterceptorMap() {
  static CreateInterceptorMap create_interceptor_map;
  return create_interceptor_map;
}

std::unique_ptr<Interceptor> CreateInterceptor(const std::string& type,
                                               int64_t id, TaskNode* node) {
  auto& map = GetCreateInterceptorMap();
  auto iter = map.find(type);
  PADDLE_ENFORCE_NE(
      iter, map.end(),
      platform::errors::NotFound("interceptor %s is not register", type));
  return iter->second(id, node);
}

#define REGISTER_INTERCEPTOR(interceptor_type, interceptor_class)          \
  namespace {                                                              \
  std::unique_ptr<Interceptor> CreatorInterceptor_##interceptor_type(      \
      int64_t id, TaskNode* node) {                                        \
    return std::make_unique<interceptor_class>(id, node);                  \
  }                                                                        \
  class __RegisterInterceptor_##interceptor_type {                         \
   public:                                                                 \
    __RegisterInterceptor_##interceptor_type() {                           \
      auto& map = GetCreateInterceptorMap();                               \
      auto iter = map.find(#interceptor_type);                             \
      PADDLE_ENFORCE_EQ(                                                   \
          iter, map.end(),                                                 \
          platform::errors::AlreadyExists(                                 \
              "interceptor %s is already registered", #interceptor_type)); \
      map[#interceptor_type] = &CreatorInterceptor_##interceptor_type;     \
    }                                                                      \
  };                                                                       \
  __RegisterInterceptor_##interceptor_type g_register_##interceptor_type;  \
  }  // namespace

}  // namespace distributed
}  // namespace paddle
