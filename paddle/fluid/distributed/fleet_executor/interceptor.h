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
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace framework {
class Scope;
class GarbageCollector;
}  // namespace framework
namespace distributed {

class TaskNode;
class Carrier;
class TaskLoop;

using InterpreterCore = framework::InterpreterCore;

constexpr int64_t SOURCE_ID = -1;
constexpr int64_t SINK_ID = -2;

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
  int64_t GetInterceptorId() const { return interceptor_id_; }

  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  void EnqueueRemoteInterceptorMessage(
      const InterceptorMessage& interceptor_message);

  bool Send(int64_t dst_id, InterceptorMessage& msg);  // NOLINT

  void SetPlace(const phi::Place& place) { place_ = place; }

  void SetRootScope(framework::Scope* scope) { root_scope_ = scope; }
  void SetMiniBatchScope(framework::Scope* scope) { minibatch_scope_ = scope; }
  void SetMicroBatchScope(const std::vector<framework::Scope*>& scopes) {
    microbatch_scopes_ = scopes;
  }
  void SetInterpreterCore(
      const std::vector<std::shared_ptr<InterpreterCore>> cores) {
    cores_ = cores;
  }
  void SetGC(const std::shared_ptr<framework::GarbageCollector>& gc) {
    gc_ = gc;
  }
  void RegisterCarrier(Carrier* carrier) { carrier_ = carrier; }
  void RegisterTaskLoop(TaskLoop* loop) { loop_ = loop; }

  TaskNode* GetTaskNode() const { return node_; }

  DISABLE_COPY_AND_ASSIGN(Interceptor);

 protected:
  // interceptor id, handed from above layer
  int64_t interceptor_id_;

  // node need to be handled by this interceptor
  TaskNode* node_;

  // for stop
  void StopCarrier();

  // for runtime
  phi::Place place_;
  framework::Scope* root_scope_{nullptr};
  framework::Scope* minibatch_scope_{nullptr};
  std::vector<framework::Scope*> microbatch_scopes_{};
  std::vector<std::shared_ptr<InterpreterCore>> cores_{};
  std::shared_ptr<framework::GarbageCollector> gc_{nullptr};

  Carrier* carrier_;
  TaskLoop* loop_;

 private:
  void LoopOnce();

  // interceptor handle which process message
  MsgHandle handle_{nullptr};

  std::mutex mutex_;
  std::deque<InterceptorMessage> messages_;
};

class InterceptorFactory {
 public:
  using CreateInterceptorFunc = std::unique_ptr<Interceptor> (*)(int64_t,
                                                                 TaskNode*);
  using CreateInterceptorMap =
      std::unordered_map<std::string, CreateInterceptorFunc>;

  static void Register(const std::string& type, CreateInterceptorFunc func);

  static std::unique_ptr<Interceptor> Create(const std::string& type,
                                             int64_t id,
                                             TaskNode* node);
};

template <typename InterceptorClass>
std::unique_ptr<Interceptor> CreatorInterceptor(int64_t id, TaskNode* node) {
  return std::make_unique<InterceptorClass>(id, node);
}

#define REGISTER_INTERCEPTOR(interceptor_type, interceptor_class)          \
  class __RegisterInterceptor_##interceptor_type {                         \
   public:                                                                 \
    __RegisterInterceptor_##interceptor_type() {                           \
      InterceptorFactory::Register(#interceptor_type,                      \
                                   CreatorInterceptor<interceptor_class>); \
    }                                                                      \
    void Touch() {}                                                        \
  };                                                                       \
  __RegisterInterceptor_##interceptor_type g_register_##interceptor_type;  \
  int TouchRegisterInterceptor_##interceptor_type() {                      \
    g_register_##interceptor_type.Touch();                                 \
    return 0;                                                              \
  }

#define USE_INTERCEPTOR(interceptor_type)                   \
  extern int TouchRegisterInterceptor_##interceptor_type(); \
  UNUSED static int use_interceptor_##interceptor_type =    \
      TouchRegisterInterceptor_##interceptor_type();

}  // namespace distributed
}  // namespace paddle
