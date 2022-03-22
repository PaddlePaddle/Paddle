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
#include <queue>
#include <thread>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Scope;
class GarbageCollector;
}
namespace distributed {

class TaskNode;
class Carrier;
class TaskLoop;

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

  void SetPlace(const platform::Place& place) { place_ = place; }

  void SetRootScope(framework::Scope* scope) { root_scope_ = scope; }
  void SetMiniBatchScope(framework::Scope* scope) { minibatch_scope_ = scope; }
  void SetMicroBatchScope(const std::vector<framework::Scope*>& scopes) {
    microbatch_scopes_ = scopes;
  }
  void SetGC(const std::shared_ptr<framework::GarbageCollector>& gc) {
    gc_ = gc;
  }
  void RegisterCarrier(Carrier* carrier) { carrier_ = carrier; }
  void RegisterTaskLoop(TaskLoop* loop) { loop_ = loop; }

  TaskNode* GetTaskNode() const { return node_; }

  DISABLE_COPY_AND_ASSIGN(Interceptor);

 protected:
  // prepare in_readys_ and out_buffs_ for each interceptor
  // only describe the dependence in current block
  virtual void PrepareDeps() {}
  // when receive data_is_ready, increase the value in in_readys_
  void IncreaseReady(int64_t up_id);
  // when receive data_is_useless, decrease the value in out_buffs_
  void DecreaseBuff(int64_t down_id);
  // check the input needed for run once is ready
  bool IsInputReady();
  // check whether there has available output buffer
  bool CanWriteOutput();
  // send data_is_ready to all downstreams
  virtual void SendDataReadyToDownStream();
  // send data_is_useless to all upstreams
  virtual void ReplyCompletedToUpStream();
  // Notify carrier to tear down
  void StopCarrier();

  // interceptor id, handed from above layer
  int64_t interceptor_id_;
  // upstream_id-->(max_ready_size, ready_size)
  std::map<int64_t, std::pair<int64_t, int64_t>> in_readys_{};
  std::queue<int64_t> ready_scope_idxs_{};
  // downstream_id-->(max_buffer_size, used_size)
  std::map<int64_t, std::pair<int64_t, int64_t>> out_buffs_{};
  int64_t scope_idx_{-1};
  // node need to be handled by this interceptor
  TaskNode* node_;
  // for runtime
  platform::Place place_;
  framework::Scope* root_scope_{nullptr};
  framework::Scope* minibatch_scope_{nullptr};
  std::vector<framework::Scope*> microbatch_scopes_{};
  std::shared_ptr<framework::GarbageCollector> gc_{nullptr};

  Carrier* carrier_;
  TaskLoop* loop_;
  bool stop_{false};  // stop_ is only used by pingpang test case

 private:
  void LoopOnce();

  // interceptor handle which process message
  MsgHandle handle_{nullptr};

  std::mutex mutex_;
  std::deque<InterceptorMessage> messages_;

  int64_t already_run_times_{0};
  int64_t used_slot_nums_{0};
};

class InterceptorFactory {
 public:
  using CreateInterceptorFunc = std::unique_ptr<Interceptor> (*)(int64_t,
                                                                 TaskNode*);
  using CreateInterceptorMap =
      std::unordered_map<std::string, CreateInterceptorFunc>;

  static void Register(const std::string& type, CreateInterceptorFunc func);

  static std::unique_ptr<Interceptor> Create(const std::string& type,
                                             int64_t id, TaskNode* node);
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
