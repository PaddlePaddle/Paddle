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
#include <condition_variable>  // NOLINT

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace distributed {

constexpr char kRequestSend[] = "RequestSend";
constexpr char kRequestGet[] = "RequestGet";
constexpr char kRequestGetMonomerVariable[] = "RequestGetMonomerVariable";
constexpr char kRequestGetMonomerBarrier[] = "RequestGetMonomerBarrier";
constexpr char kRequestPrefetch[] = "RequestPrefetch";
constexpr char kRequestCheckpoint[] = "RequestCheckpoint";
constexpr char kRequestPassBarrier[] = "RequestPassBarrier";
constexpr char kRequestGetNoBarrier[] = "GetVariableNoBarrier";
constexpr char kRequestNotify[] = "RequestNotify";

constexpr char kSendRPC[] = "SendRPC";
constexpr char kGetRPC[] = "GetRPC";
constexpr char kGetNoBarrierRPC[] = "GetNoBarrierRPC";
constexpr char kGetMonomerRPC[] = "GetMonomerRPC";
constexpr char kPrefetchRPC[] = "PrefetchRPC";
constexpr char kBatchBarrierRPC[] = "BatchBarrierRPC";
constexpr char kFetchBarrierRPC[] = "FetchBarrierRPC";
constexpr char kSendMonomerFetchBarrierRPC[] = "SendMonomerFetchBarrierRPC";
constexpr char kSendCompleteRPC[] = "SendCompleteRPC";
constexpr char kCheckPointNotifyRPC[] = "CheckPointNotifyRPC";

#define LISTEN_TERMINATE_MESSAGE "TERMINATE@RECV"
#define BATCH_BARRIER_MESSAGE "BATCH_BARRIER@RECV"
#define FETCH_BARRIER_MESSAGE "FETCH_BARRIER@RECV"
#define COMPLETE_MESSAGE "COMPLETE@RECV"
#define WITHOUT_BARRIER_MESSAGE "@WITHOUT_BARRIER@RECV"
#define LEARNING_RATE_DECAY_COUNTER "@LR_DECAY_COUNTER@"

#define CHECKPOINT_SAVE_MESSAGE "SAVE@CHECKPOINTNOTIFY"
#define CHECKPOINT_LOAD_MESSAGE "LOAD@CHECKPOINTNOTIFY"

enum DistributedMode { kSync = 0, kAsync = 1, kHalfAsync = 2, kGeo = 3 };

class RPCServer;

class VarHandle {
 public:
  VarHandle(const std::string ep, const std::string& method,
            const std::string& name,
            const platform::DeviceContext* p_ctx = nullptr,
            const framework::Scope* p_scope = nullptr)
      : status_(kDefaultState) {
    ep_ = ep;
    ctx_ = p_ctx;
    scope_ = p_scope;
    name_ = name;
    method_ = method;
  }

  virtual ~VarHandle() {}

 public:
  bool should_retry = false;

  bool Wait() {
    int ret = kDefaultState;
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      wait_cond_.wait(lk, [this] { return status_ != kDefaultState; });
      ret = status_;
    }
    VLOG(7) << "VarHandle wait:" << ret;
    return ret != kErrorState;
  }

  void Finish(bool ok) {
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      status_ = ok ? kFinishState : kErrorState;
    }
    VLOG(7) << "VarHandle finish:" << ok;
    wait_cond_.notify_all();
  }

  std::string String() const {
    std::ostringstream s;
    s << method_ << " name:[" << name_ << "], ep:[" << ep_ << "], status:["
      << status_ << "]";
    return s.str();
  }

  std::string ep() const { return ep_; }
  const platform::DeviceContext* ctx() const { return ctx_; }
  const framework::Scope* scope() const { return scope_; }
  std::string name() const { return name_; }
  std::string method() const { return method_; }

 protected:
  // RPC endpoint.
  std::string ep_;
  const platform::DeviceContext* ctx_;
  const framework::Scope* scope_;
  // Variable name.
  std::string name_;
  // RPC method name.
  std::string method_;

 protected:
  std::mutex sync_mutex_;
  std::condition_variable wait_cond_;

  enum VarHandleStatus {
    kDefaultState = -1,
    kErrorState = 0,
    kFinishState = 1,
  };
  VarHandleStatus status_;

 private:
  DISABLE_COPY_AND_ASSIGN(VarHandle);
};

typedef std::shared_ptr<VarHandle> VarHandlePtr;

class RequestHandler {
 public:
  explicit RequestHandler(int distributed_mode)
      : distributed_mode_(distributed_mode),
        dev_ctx_(nullptr),
        executor_(nullptr),
        scope_(nullptr),
        program_(nullptr),
        rpc_server_(nullptr) {}

  virtual ~RequestHandler() {}

  // Set attributes.
  void SetScope(framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }
  void SetExecutor(framework::Executor* executor) { executor_ = executor; }

  // Used for dist lookup table prefetch
  void SetPrefetchPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    prefetch_var_name_to_prepared_ctx_ = g;
  }

  void SetCheckpointNotifyPreparedCtx(
      std::shared_ptr<framework::ExecutorPrepareContext> g) {
    checkpoint_prepared_ctx_ = g;
  }

  // Used for async.
  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    grad_to_prepared_ctx_ = g;
  }

  void SetSparseGradToParam(std::unordered_map<std::string, std::string>* g) {
    sparse_grad_to_param_ = g;
  }

  void SetLrDecayPreparedCtx(
      std::shared_ptr<framework::ExecutorPrepareContext> g) {
    lr_decay_prepared_ctx_ = g;
  }

  void SetRPCServer(RPCServer* rpc_server) { rpc_server_ = rpc_server; }

  // Get attributes.
  int distributed_mode() { return distributed_mode_; }
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }

  // This function processes user's rpc request.
  // The implemention is in request_handler_impl.
  // example:
  //    std::string varname = request_.varname();
  //
  //    auto scope = request_handler_->scope();
  //    auto invar = scope->FindVar(varname);
  //    framework::Variable* outvar = nullptr;
  //
  //    request_handler_->Handle(varname, scope, invar, &outvar);
  //    if (outvar) {
  //        SerializeToByteBuffer(varname, outvar,
  //           *request_handler_->dev_ctx(), &reply_);
  //    }
  virtual bool Handle(const std::string& varname, framework::Scope* scope,
                      framework::Variable* var, framework::Variable** outvar,
                      const int trainer_id,
                      const std::string& out_var_name = "",
                      const std::string& table_name = "") = 0;

 protected:
  const int distributed_mode_;

  const platform::DeviceContext* dev_ctx_;
  framework::Executor* executor_;
  framework::Scope* scope_;
  framework::ProgramDesc* program_;

  // used for distribute lookup table prefetch
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      prefetch_var_name_to_prepared_ctx_;
  // used for checkpoint notify
  std::shared_ptr<framework::ExecutorPrepareContext> checkpoint_prepared_ctx_;

  // Used for async.
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      grad_to_prepared_ctx_;
  std::unordered_map<std::string, std::string>* sparse_grad_to_param_;

  // used for lr decay
  std::shared_ptr<framework::ExecutorPrepareContext> lr_decay_prepared_ctx_;
  RPCServer* rpc_server_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
