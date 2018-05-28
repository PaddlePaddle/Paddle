// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <time.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

typedef std::pair<std::string, std::shared_ptr<VariableResponse>>
    ReceivedMessage;

class RPCProcessorCtx {
 public:
  RPCProcessorCtx()
      : sync_mode_(true),
        scope_(nullptr),
        dev_ctx_(nullptr),
        program_(nullptr),
        executor_(nullptr) {}
  virtual ~RPCProcessorCtx() {}

  void SetSyncMode(bool sync_mode) { sync_mode_ = sync_mode; }
  void SetScope(framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }
  void SetExecutor(framework::Executor* executor) { executor_ = executor; }
  void SetPrefetchPreparedCtx(
      std::unique_ptr<framework::ExecutorPrepareContext> prepared) {
    prefetch_ctx_.reset(prepared.release());
  }

  // Used for async
  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    grad_to_prepared_ctx_ = g;
  }

  bool sync_mode() { return sync_mode_; }
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ExecutorPrepareContext* prefetch_ctx() {
    return prefetch_ctx_.get();
  }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }

 protected:
  bool sync_mode_;
  framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;

  std::unique_ptr<framework::ExecutorPrepareContext> prefetch_ctx_;
  framework::ProgramDesc* program_;
  framework::Executor* executor_;

  // Used for async.
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      grad_to_prepared_ctx_;
};

class GRPCProcessorCtx : public RPCProcessorCtx {
 public:
  GRPCProcessorCtx() : exit_flag_(false), fan_in_(-1) { clear_to_init(); }
  virtual ~GRPCProcessorCtx() {}

  bool ProcessSendImpl(const std::string& msg_name, framework::Variable* var,
                       framework::Scope* scope = nullptr);

  bool ProcessGetImpl(const std::string& msg_name, framework::Variable** var);

  bool ProcessPrefetchImpl(const std::string& msg_name, framework::Scope* scope,
                           framework::Variable** var);

  void SetFanIn(int fan_in) { fan_in_ = fan_in; }

  void WaitFanInOfSend() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    condition_send_.wait(lock, [=] {
      return (this->batch_barrier_send_ >= fan_in_ || exit_flag_);
    });

    VLOG(3) << "batch_barrier_send_:" << batch_barrier_send_;
  }

  void WaitFanInOfGet() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    condition_get_.wait(lock, [=] {
      return (this->batch_barrier_get_ >= fan_in_ || exit_flag_);
    });

    VLOG(3) << "batch_barrier_get_:" << batch_barrier_get_;
  }

  void clear_to_init() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    batch_barrier_send_ = 0;

    sparse_vars_.clear();
    batch_barrier_get_ = 0;
  }

  void SetExit();

  bool IsExit() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    return exit_flag_;
  }

  std::vector<framework::Variable*>& sparse_vars() { return sparse_vars_; }

 private:
  void IncreaseBatchBarrierGet();
  void IncreaseBatchBarrierSend();
  // void IncreaseRecvVarCnt();

 private:
  // status
  bool exit_flag_;
  int fan_in_;
  std::mutex mutex_;

  // send
  std::condition_variable condition_send_;
  int batch_barrier_send_;

  // get
  std::condition_variable condition_get_;
  int batch_barrier_get_;
  // Record received sparse variables, so that
  // we could reset those after execute optimize program
  std::vector<framework::Variable*> sparse_vars_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
