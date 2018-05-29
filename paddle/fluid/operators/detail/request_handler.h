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

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

constexpr char kRequestSend[] = "RequestSend";
constexpr char kRequestGet[] = "RequestGet";
constexpr char kRequestPrefetch[] = "RequestPrefetch";

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

  // set attribute
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

  void SetRPCServer(RPCServer* rpc_server) { rpc_server_ = rpc_server; }

  // get attribute
  bool sync_mode() { return sync_mode_; }
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ExecutorPrepareContext* prefetch_ctx() {
    return prefetch_ctx_.get();
  }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }
  std::vector<framework::Variable*>& sparse_vars() { return sparse_vars_; }

  // request handler
  virtual bool Handle(void* input, void* output) = 0;

 protected:
  const bool sync_mode_;

  const platform::DeviceContext* dev_ctx_;
  framework::Executor* executor_;
  framework::Scope* scope_;
  framework::ProgramDesc* program_;
  std::unique_ptr<framework::ExecutorPrepareContext> prefetch_ctx_;

  // Used for async.
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      grad_to_prepared_ctx_;

  // get
  // Record received sparse variables, so that
  // we could reset those after execute optimize program
  std::vector<framework::Variable*> sparse_vars_;
  RPCServer* rpc_server_;

  std::mutex sparse_var_mutex_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
