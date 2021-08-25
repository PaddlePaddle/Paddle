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

#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/context_callback.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/trainer_context.h"

namespace paddle {
namespace framework {

int TrainerContextInterface::Initialize(CommRole comm_role) {
  comm_ = comm_role;
  // REGIST_CALLBACK
  REGIST_CONTEXT_CALLBACK(FeedAucCalculator)
  REGIST_CONTEXT_CALLBACK(FeedWuaucCalculator);
  REGIST_CONTEXT_CALLBACK(FeedPnCalculator);
  // REGIST_CONTEXT_CALLBACK(FeedInstanceDumper);
  return 0;
}

int TrainerContextInterface::Finalize() {
  for (auto callback_group : callback_map_) {
    for (auto callback_item : callback_group.second) {
      callback_item.second->Clean();
    }
  }
  return 0;
}

void TrainerContextInterface::SetTrainer(TrainerBase* trainer,
                                         const TrainerDesc& desc) {
  trainer_ = trainer;
  trainer_desc_ = desc;
  root_scope_ = trainer->root_scope_;
  thread_num_ = trainer->thread_num_;
  dataset_ptr_ = trainer->dataset_ptr_;
}

#define DEFINE_CALLBACK_IMPL(callback_type, callback_func, ...)                \
  static std::string callback_name = typeid(callback_type).name();             \
  if (callback_map_.count(callback_name)) {                                    \
    auto& callbacks = callback_map_[callback_name];                            \
    for (auto& callback_item : callbacks) {                                    \
      auto* callback =                                                         \
          reinterpret_cast<callback_type*>(callback_item.second.get());        \
      callback->callback(this, ##__VA_ARGS__);                                 \
    }                                                                          \
  }                                                                            \
  if (callback_map_.count("ContextCallBackGroup")) {                           \
    auto& callbacks = callback_map_["ContextCallBackGroup"];                   \
    for (auto& callback_item : callbacks) {                                    \
      auto* callback =                                                         \
          reinterpret_cast<ContextCallBackGroup*>(callback_item.second.get()); \
      callback->callback_func(this, ##__VA_ARGS__);                            \
    }                                                                          \
  }

// call before a train job begin
void TrainerContextInterface::on_trainer_run_begin() {
  DEFINE_CALLBACK_IMPL(TrainerBeginCallBack, trainer_begin_callback);
}

// call after a train job  end
void TrainerContextInterface::on_trainer_run_end() {
  DEFINE_CALLBACK_IMPL(TrainerEndCallBack, trainer_end_callback);
}

// call before train in every train_thread
void TrainerContextInterface::on_thread_train_end(DeviceWorker* worker) {
  DEFINE_CALLBACK_IMPL(ThreadBeginCallBack, thread_begin_callback, worker);
}
// call after train in every train_thread
void TrainerContextInterface::on_thread_train_begin(DeviceWorker* worker) {
  DEFINE_CALLBACK_IMPL(ThreadEndCallBack, thread_end_callback, worker);
}

// call after pulled sparse/dense param in every train_thread
void TrainerContextInterface::on_thread_pull_done(DeviceWorker* worker) {
  DEFINE_CALLBACK_IMPL(ThreadPulledCallBack, thread_pulled_callback, worker);
}
// call after dnn(forward && backward) in every train_thread
void TrainerContextInterface::on_thread_op_done(DeviceWorker* worker) {
  DEFINE_CALLBACK_IMPL(ThreadOpDoneCallBack, thread_op_done_callback, worker);
}

int FeedTrainerContext::Initialize(CommRole comm_role) {
  int ret = TrainerContextInterface::Initialize(comm_role);
  if (ret == 0) {
    int mpi_argc = 0;  // argc置为0， 避免MPI重复初始化
    char** mpi_argv = NULL;
    mpi_ = std::make_shared<CommonMPI>();
    mpi_->Initialize(mpi_argc, mpi_argv);
  }
  return ret;
}

}  // namespace framework
}  // namespace paddle
