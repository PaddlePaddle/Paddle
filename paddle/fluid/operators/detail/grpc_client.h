/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/simple_block_queue.h"

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

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& msg);

class ClientBase {
 public:
  explicit ClientBase(std::shared_ptr<grpc::Channel> ch) {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
    context_ = NULL;
  }

  virtual ~ClientBase() {}

  virtual void Prepare(const VarHandle& var_info, int64_t time_out) {
    context_.reset(new grpc::ClientContext());
    var_h_ = var_info;

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context_->set_deadline(deadline);
  }

  virtual void Prepare(int64_t time_out) {
    context_.reset(new grpc::ClientContext());

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context_->set_deadline(deadline);
  }

  virtual void Process() = 0;

  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> context_;
  grpc::Status status_;
  VarHandle var_h_;
};

typedef std::function<void(const VarHandle&, const sendrecv::VoidMessage&)>
    RequestSendCallBack;

class SendProcessor : public ClientBase {
 public:
  explicit SendProcessor(std::shared_ptr<grpc::Channel> ch) : ClientBase(ch) {}

  virtual ~SendProcessor() {}

  virtual void Process() {
    if (response_call_back_) {
      response_call_back_(var_h_, reply_);
    }
  }

  sendrecv::VoidMessage reply_;
  RequestSendCallBack response_call_back_ = NULL;
};

typedef std::function<void(const VarHandle&, const sendrecv::VariableMessage&)>
    RequestGetCallBack;

class GetProcessor : public ClientBase {
 public:
  explicit GetProcessor(std::shared_ptr<grpc::Channel> ch) : ClientBase(ch) {}

  virtual ~GetProcessor() {}

  virtual void Process() {
    if (response_call_back_) {
      response_call_back_(var_h_, reply_);
    }
  }

  sendrecv::VariableMessage reply_;
  RequestGetCallBack response_call_back_ = ProcGetResponse;
};

class BatchBarrierProcessor : public ClientBase {
 public:
  explicit BatchBarrierProcessor(std::shared_ptr<grpc::Channel> ch)
      : ClientBase(ch) {}

  virtual ~BatchBarrierProcessor() {}

  virtual void Process() {}
  sendrecv::VoidMessage reply_;
};

class RPCClient {
 public:
  bool AsyncSendVariable(const std::string& ep,
                         const platform::DeviceContext& ctx,
                         const framework::Scope& scope,
                         const std::string& var_name,
                         int64_t time_out = 600 * 1000);

  bool AsyncGetVariable(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& var_name,
                        int64_t time_out = 600 * 1000);

  bool AsyncSendBatchBarrier(const std::string& ep,
                             int64_t time_out = 600 * 1000);

  bool Wait();

 private:
  bool Proceed();
  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);

 private:
  grpc::CompletionQueue cq_;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels_;
  int64_t req_count_ = 0;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
