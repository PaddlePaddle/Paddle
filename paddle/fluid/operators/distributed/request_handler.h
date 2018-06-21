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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {
namespace distributed {

constexpr char kRequestSend[] = "RequestSend";
constexpr char kRequestGet[] = "RequestGet";
constexpr char kRequestPrefetch[] = "RequestPrefetch";

#define LISTEN_TERMINATE_MESSAGE "TERMINATE@RECV"
#define BATCH_BARRIER_MESSAGE "BATCH_BARRIER@RECV"
#define FETCH_BARRIER_MESSAGE "FETCH_BARRIER@RECV"
#define COMPLETE_MESSAGE "COMPLETE@RECV"

class RPCServer;

class RequestHandler {
 public:
  explicit RequestHandler(bool sync_mode)
      : sync_mode_(sync_mode),
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

  // Used for async.
  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    grad_to_prepared_ctx_ = g;
  }

  void SetRPCServer(RPCServer* rpc_server) { rpc_server_ = rpc_server; }

  // Get attributes.
  bool sync_mode() { return sync_mode_; }
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
                      const std::string& out_var_name = "") = 0;

 protected:
  const bool sync_mode_;

  const platform::DeviceContext* dev_ctx_;
  framework::Executor* executor_;
  framework::Scope* scope_;
  framework::ProgramDesc* program_;

  // used for distribute lookup table prefetch
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      prefetch_var_name_to_prepared_ctx_;

  // Used for async.
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      grad_to_prepared_ctx_;

  RPCServer* rpc_server_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
