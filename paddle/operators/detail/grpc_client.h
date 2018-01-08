/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include <time.h>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "paddle/framework/data_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/operators/detail/simple_block_queue.h"
#include "paddle/operators/detail/utils.h"

namespace paddle {
namespace operators {
namespace detail {

struct VarHandle {
  std::string ep;
  const platform::DeviceContext* ctx;
  const framework::Scope* scope;
  std::string name;

  std::string String() const {
    std::ostringstream s;
    s << "name:[" << name << "] ep:[" << ep << "]";
    return s.str();
  }
};

void ProcSendResponse(const VarHandle&, const sendrecv::VoidMessage& msg);

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& msg);

struct ClientBase {
  explicit ClientBase(std::shared_ptr<grpc::Channel> ch) {
    stub = sendrecv::SendRecvService::NewStub(ch);
    context = NULL;
  }

  virtual ~ClientBase() {}

  virtual void Prepare(const VarHandle& var_info, int64_t time_out) {
    context.reset(new grpc::ClientContext());
    var_h = var_info;

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context->set_deadline(deadline);
  }

  virtual void Proceed() = 0;

  std::unique_ptr<sendrecv::SendRecvService::Stub> stub;
  std::unique_ptr<grpc::ClientContext> context;
  grpc::Status status;
  VarHandle var_h;
};

typedef std::function<void(const VarHandle&, const sendrecv::VoidMessage&)>
    RequestSendCallBack;

struct SendProcessor : public ClientBase {
  explicit SendProcessor(std::shared_ptr<grpc::Channel> ch) : ClientBase(ch) {}

  virtual ~SendProcessor() {}

  virtual void Proceed() { response_call_back(var_h, reply); }

  sendrecv::VoidMessage reply;
  RequestSendCallBack response_call_back = ProcSendResponse;
};

typedef std::function<void(const VarHandle&, const sendrecv::VariableMessage&)>
    RequestGetCallBack;

struct GetProcessor : public ClientBase {
  explicit GetProcessor(std::shared_ptr<grpc::Channel> ch) : ClientBase(ch) {}

  virtual ~GetProcessor() {}

  virtual void Proceed() { response_call_back(var_h, reply); }

  sendrecv::VariableMessage reply;
  RequestGetCallBack response_call_back = ProcGetResponse;
};

class RPCClient {
 public:
  bool AsyncSendVariable(const std::string& ep,
                         const platform::DeviceContext& ctx,
                         const framework::Scope* scope,
                         const std::string& var_name,
                         int64_t time_out = 600 * 1000);

  bool AsyncGetVariable(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope* scope,
                        const std::string& var_name,
                        int64_t time_out = 600 * 1000);
  bool wait();

 private:
  bool Proceed();
  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);

 private:
  grpc::CompletionQueue cq_;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels_;
  int64_t count_ = 0;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
