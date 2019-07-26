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
#include "paddle/fluid/operators/distributed/request.h"
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

#define CHECKPOINT_SAVE_MESSAGE "SAVE@CHECKPOINTNOTIFY"
#define CHECKPOINT_LOAD_MESSAGE "LOAD@CHECKPOINTNOTIFY"

class RPCServer;

class RequestHandler {
 public:
  RequestHandler()
      : dev_ctx_(nullptr),
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

  // Used for send/get/prefetch handlers, so put it here.
  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    grad_to_prepared_ctx_ = g;
  }

  void SetRPCServer(RPCServer* rpc_server) { rpc_server_ = rpc_server; }

  // Get attributes.
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }

  // Handle should call start/finish as following order
  // GetOrCreateRequestVar should return the variable pointer to store
  // incomming.
  // In Handle, process the request, set out_var as response.
  virtual bool Handle(RPCRequest* request) = 0;
  virtual framework::Variable* GetOrCreateRequestVar(const std::string& varname,
                                                     RPCRequest* request) = 0;

 protected:
  const platform::DeviceContext* dev_ctx_;
  framework::Executor* executor_;
  framework::Scope* scope_;
  framework::ProgramDesc* program_;

  RPCServer* rpc_server_;

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      grad_to_prepared_ctx_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
