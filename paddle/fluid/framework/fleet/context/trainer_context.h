/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/message_passing_interface.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/variable_helper.h"

namespace paddle {
namespace framework {

class TrainerBase;
class DeviceWorker;
class TrainerContextInterface;

class ContextCallBackInterface {
 public:
  virtual ~ContextCallBackInterface() {}
  virtual const std::string& TypeName() = 0;
  virtual void Clean() {}
};

template <class Derived, class... Args>
class ContextCallBack : public ContextCallBackInterface {
 public:
  virtual ~ContextCallBack() {}
  virtual const std::string& TypeName() {
    static std::string type_name = typeid(Derived).name();
    return type_name;
  }
  virtual void callback(Args...) = 0;
};

// 单流程回调插件申明
// 继承对应流程插件后，实现callback函数，则会在对应阶段获得回调
#define DECLARE_TRAINER_CALLBACK(CallBackType, ...) \
  class CallBackType : public ContextCallBack<CallBackType, ##__VA_ARGS__> {};

// 插件 调用时机：训练阶段开始（比如join/update)
DECLARE_TRAINER_CALLBACK(TrainerBeginCallBack, TrainerContextInterface*)
// 插件 调用时机：训练阶段结束（比如join/update)
DECLARE_TRAINER_CALLBACK(TrainerEndCallBack, TrainerContextInterface*)
// 插件 调用时机：训练阶段内多线程训练，每个线程启动时各调用一次
DECLARE_TRAINER_CALLBACK(ThreadBeginCallBack, TrainerContextInterface*,
                         DeviceWorker*)
// 插件 调用时机：训练阶段内多线程训练，每个线程结束时各调用一次
DECLARE_TRAINER_CALLBACK(ThreadEndCallBack, TrainerContextInterface*,
                         DeviceWorker*)
// 插件 调用时机：训练阶段内多线程训练，每个线程的每个Batch pull参数完成后回调
DECLARE_TRAINER_CALLBACK(ThreadPulledCallBack, TrainerContextInterface*,
                         DeviceWorker*)
// 插件 调用时机：训练阶段内多线程训练，每个线程的每个Batch OP run完成后回调
DECLARE_TRAINER_CALLBACK(ThreadOpDoneCallBack, TrainerContextInterface*,
                         DeviceWorker*)

#undef DECLARE_TRAINER_CALLBACK

// 多流程回调插件
// 继承插件后，重载所需的流程回调函数，所有回调函数都将获得调用
// 用于实现多阶段的复杂需求：如需要分Batch统计，最终再汇聚所有线程&所有节点的Batch统计结果
// 则需要同时实现thread_op_done_callback、trainer_end_callback
// 通过自定义成员变量，可维护多流程间的中间变量或状态
class ContextCallBackGroup : public ContextCallBackInterface {
 public:
  virtual ~ContextCallBackGroup() {}
  virtual const std::string& TypeName() {
    static std::string type_name = "ContextCallBackGroup";
    return type_name;
  }
  virtual void trainer_begin_callback(TrainerContextInterface*) {}
  virtual void trainer_end_callback(TrainerContextInterface*) {}
  virtual void thread_begin_callback(TrainerContextInterface*, DeviceWorker*) {}
  virtual void thread_end_callback(TrainerContextInterface*, DeviceWorker*) {}
  virtual void thread_pulled_callback(TrainerContextInterface*, DeviceWorker*) {
  }
  virtual void thread_op_done_callback(TrainerContextInterface*,
                                       DeviceWorker*) {}
};

// 可以注册一个单独的Callback
// 也可以注册一个对应多流程回调的Callback组合
// callback实例生命周期与TrainerContext相同
// 会被多线程调用，需保证线程安全
// 每个训练Job(如join/update)结束后,会统一调用Clean 可在此做必要后置处理
#define REGIST_CONTEXT_CALLBACK(CallBackClass)                          \
  {                                                                     \
    auto callback =                                                     \
        std::shared_ptr<ContextCallBackInterface>(new CallBackClass()); \
    regist_callback(#CallBackClass, callback);                          \
  }

class TrainerContextInterface {
 public:
  virtual ~TrainerContextInterface() {}
  static std::string ContextVarName() { return "trainer_context"; }
  virtual int Initialize(CommRole role_comm);
  virtual int Finalize();
  // model memory are hosted in root_scope
  virtual Scope* GetScope() { return root_scope_; }
  virtual void SetTrainer(TrainerBase* trainer, const TrainerDesc& desc);

  template <class DataSetType>
  DataSetType* GetDataset() {
    return static_cast<DataSetType*>(dataset_ptr_);
  }

  // call before a train job begin
  virtual void on_trainer_run_begin();
  // call after a train job  end
  virtual void on_trainer_run_end();

  // call before train in every train_thread
  virtual void on_thread_train_end(DeviceWorker* worker);
  // call after train in every train_thread
  virtual void on_thread_train_begin(DeviceWorker* worker);

  // call after pulled sparse/dense param in every train_thread
  virtual void on_thread_pull_done(DeviceWorker* worker);
  // call after dnn(forward && backward) in every train_thread
  virtual void on_thread_op_done(DeviceWorker* worker);

 public:
  CommRole comm_;
  uint32_t thread_num_ = 0;
  TrainerDesc trainer_desc_;
  Scope* root_scope_ = nullptr;
  Dataset* dataset_ptr_ = nullptr;
  TrainerBase* trainer_ = nullptr;
  std::unordered_map<
      std::string,
      std::map<std::string, std::shared_ptr<ContextCallBackInterface>>>
      callback_map_;

 private:
  virtual void regist_callback(
      const std::string& callback_name,
      std::shared_ptr<ContextCallBackInterface> callback_ptr) {
    auto type_name = callback_ptr->TypeName();
    if (callback_map_.find(type_name) == callback_map_.end()) {
      auto useless = callback_map_[type_name];
    }
    auto& type_callbacks = callback_map_[type_name];
    type_callbacks[callback_name] = callback_ptr;
  }
};

template <class Derived>
class TrainerContext : public TrainerContextInterface {
 public:
  static Derived* get_context(Scope* root_scope) {
    auto var = root_scope->FindVar("trainer_context");
    if (var == nullptr) {
      return nullptr;
    }
    long ptr = var->Get<long>();  // NOLINT
    return (Derived*)ptr;         // NOLINT
  }
};

// TODO(paddle-dev) support Registery
class FeedTrainerContext : public TrainerContext<FeedTrainerContext> {
 public:
  virtual int Initialize(CommRole role_comm);
  inline CommonMPI* MPI() { return mpi_.get(); }

 public:
  std::shared_ptr<CommonMPI> mpi_;
};

template <typename T>
void rank0_print(T& args, TrainerContextInterface* ctx) {  // NOLINT
  if (ctx == NULL ||
      ((FeedTrainerContext*)ctx)->MPI()->Rank(ctx->comm_) != 0) {  // NOLINT
    return;
  }
  std::cout << args << std::flush;
}

}  // namespace framework
}  // namespace paddle
