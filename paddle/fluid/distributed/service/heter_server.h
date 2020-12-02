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
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/service/heter_serde.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace distributed {

using MultiVarMsg = ::paddle::MultiVariableMessage;
using VarMsg = ::paddle::VariableMessage;

typedef std::function<void(void*)> HeterRpcCallbackFunc;
typedef std::function<int(const MultiVarMsg*, MultiVarMsg*, brpc::Controller*)>
    HeterServiceHandler;

class HeterService : public ::paddle::PsService {
 public:
  HeterService() {}
  virtual ~HeterService() {}
  void SendAndRecvVariable(::google::protobuf::RpcController* controller,
                           const MultiVarMsg* request, MultiVarMsg* response,
                           ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    int ret = 0;
    std::string message_name = request->message_name();
    auto itr = handler_map_.find(message_name);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    PADDLE_ENFORCE_NE(
        itr, handler_map_.end(),
        platform::errors::InvalidArgument(
            "HeterService::SendAndRecvVariable Get illegal message_name: %s "
            "which is not in HeterService::handler_map_",
            message_name));
    ret = itr->second(request, response, cntl);
  }

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func) {
    handler_map_[message_name] = func;
  }

 private:
  std::unordered_map<std::string, HeterServiceHandler> handler_map_;
};

class HeterServer {
 public:
  virtual ~HeterServer() {
    server_.Stop(1000);
    server_.Join();
  }

  void Stop() {
    server_.Stop(1000);
    server_.Join();
  }

  HeterServer() {}

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func);

  void StartHeterService();

  void SetEndPoint(std::string& endpoint);

  // HeterWrapper singleton
  static std::shared_ptr<HeterServer> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new HeterServer());
    }
    return s_instance_;
  }

  void WaitServerReady();

 private:
  static std::shared_ptr<HeterServer> s_instance_;
  std::string endpoint_;

 protected:
  brpc::Server server_;
  HeterService service_;
  DISABLE_COPY_AND_ASSIGN(HeterServer);
  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;
  int ready_;
};

class HeterRequestHandler {
 public:
  explicit HeterRequestHandler()
      : dev_ctx_(nullptr),
        executor_(nullptr),
        scope_(nullptr),
        program_(nullptr) {}

  virtual ~HeterRequestHandler() {}

  void SetScope(framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }
  void SetExecutor(framework::Executor* executor) { executor_ = executor; }

  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    message_to_prepared_ctx_ = g;
  }

  virtual int Handle(const MultiVarMsg* request, MultiVarMsg* response,
                     brpc::Controller* cntl) = 0;

 protected:
  const platform::DeviceContext* dev_ctx_;
  framework::Executor* executor_;
  framework::Scope* scope_;
  framework::ProgramDesc* program_;

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      message_to_prepared_ctx_;
};

class RequestSendAndRecvHandler final : public HeterRequestHandler {
 public:
  explicit RequestSendAndRecvHandler() {}
  virtual ~RequestSendAndRecvHandler() {}
  int Handle(const MultiVarMsg* request, MultiVarMsg* response,
             brpc::Controller* cntl) override {
    auto& local_scope = scope_->NewScope();
    auto message_name = request->message_name();
    auto& request_io_buffer = cntl->request_attachment();
    distributed::DeserializeFromMultiVarMsgAndIOBuf(
        *request, &request_io_buffer, *dev_ctx_, &local_scope);
    executor_->RunPreparedContext(
        (*message_to_prepared_ctx_)[message_name].get(), &local_scope, false);

    auto response_var_nums = request->recv_var_names_size();
    std::vector<std::string> response_var_names(response_var_nums),
        empty_var_names{};

    for (int var_idx = 0; var_idx < response_var_nums; ++var_idx) {
      response_var_names[var_idx] = request->recv_var_names(var_idx);
    }
    auto& response_io_buffer = cntl->response_attachment();
    distributed::SerializeToMultiVarMsgAndIOBuf(
        message_name, response_var_names, empty_var_names, *dev_ctx_,
        &local_scope, response, &response_io_buffer);
    scope_->DeleteScope(&local_scope);
    return 0;
  }
};

}  // end namespace distributed
}  // end namespace paddle
