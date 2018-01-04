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

#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

namespace paddle {
namespace operators {
namespace detail {

struct VarHandle {
  std::string ep;
  std::string name;
  std::string String() const {
    std::ostringstream s;
    s << "name:" << name << " ep:" << ep;
    return s.str();
  }
};

struct SendStatus {
  std::string error;
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  VarHandle var;

  std::string String() const {
    std::time_t t0 = std::chrono::system_clock::to_time_t(start);
    std::time_t t1 = std::chrono::system_clock::to_time_t(end);
    std::ostringstream s;
    s << var.String() << " start_time:" << std::ctime(&t0)
      << " end_time:" << std::ctime(&t1);
    return s.str();
  }
};

class ClientBase {
 public:
  virtual bool Call(const framework::Scope* scope, const std::string& name,
                    int64_t time_out) = 0;
  virtual bool ProcRet() = 0;
  virtual grpc::Status* GetStatus() = 0;
  virtual SendStatus GetSendStatus() = 0;

  virtual ~ClientBase() = 0;
};

class SendClient : public ClientBase {
 public:
  SendClient(std::shared_ptr<grpc::CompletionQueue> cq,
             std::shared_ptr<grpc::Channel> ch, std::string ep) {
    cq_ = cq;
    ch_ = ch;
    ep_ = ep;
    stub_ = sendrecv::SendRecvService::NewStub(ch);
    context_.reset(new grpc::ClientContext());
    status_.reset(new grpc::Status());
    reply_.reset(new sendrecv::VoidMessage());
  }

  virtual ~SendClient() {}

  typedef std::chrono::system_clock::time_point time_point;

  bool Call(const framework::Scope* scope, const std::string& name,
            int64_t time_out) {
    // record send status
    send_status_.start = std::chrono::system_clock::now();
    send_status_.end = send_status_.start;
    send_status_.var.ep = ep_;
    send_status_.var.name = name;

    const time_point deadline =
        send_status_.start + std::chrono::milliseconds(time_out);
    context_->set_deadline(deadline);

    sendrecv::VariableMessage req;
    _Request(scope, name, &req);

    rpc_ = stub_->AsyncSendVariable(context_.get(), req, cq_.get());
    rpc_->Finish(reply_.get(), status_.get(), (void*)this);

    return true;
  }

  bool ProcRet() { return true; }

  grpc::Status* GetStatus() { return status_.get(); }

  SendStatus GetSendStatus() { return send_status_; }

 private:
  void _Request(const framework::Scope* scope, const std::string name,
                sendrecv::VariableMessage* msg) {
    // FIXME(gongwb): pass device context to here.
    auto ctx = platform::CPUDeviceContext();
    auto* var = scope->FindVar(name);
    PADDLE_ENFORCE(var);
    // TODO(gongwb): support SelectedRows
    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "Only support LoDTensor, %s has wrong type", name);

    const framework::LoDTensor& tensor = var->Get<framework::LoDTensor>();
    std::ostringstream oss;
    framework::SerializeToStream(oss, tensor, ctx);
    msg->set_varname(name);
    msg->set_serialized(oss.str());
  }

 private:
  std::shared_ptr<grpc::CompletionQueue> cq_;
  std::shared_ptr<grpc::Channel> ch_;
  std::string ep_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> context_;
  std::unique_ptr<grpc::Status> status_;
  std::unique_ptr<sendrecv::VoidMessage> reply_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<sendrecv::VoidMessage>> rpc_;
  SendStatus send_status_;
};

class GetClient : public ClientBase {
 public:
  GetClient(std::shared_ptr<grpc::CompletionQueue> cq,
            std::shared_ptr<grpc::Channel> ch, std::string ep) {
    cq_ = cq;
    ch_ = ch;
    ep_ = ep;
    stub_ = sendrecv::SendRecvService::NewStub(ch);
    context_.reset(new grpc::ClientContext());
    status_.reset(new grpc::Status());
    reply_.reset(new sendrecv::VoidMessage());
  }

  virtual ~GetClient() {}

  bool Call(const framework::Scope* scope, const std::string& name,
            int64_t time_out) {
    return true;
  }

  bool ProcRet() { return true; }

  grpc::Status* GetStatus() { return status_.get(); }

  SendStatus GetSendStatus() { return send_status_; }

 private:
  std::shared_ptr<grpc::CompletionQueue> cq_;
  std::shared_ptr<grpc::Channel> ch_;
  std::string ep_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> context_;
  std::unique_ptr<grpc::Status> status_;
  std::unique_ptr<sendrecv::VoidMessage> reply_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<sendrecv::VoidMessage>> rpc_;
  SendStatus send_status_;
};

class RPCClients {
 public:
  bool AsyncSendVariable(const std::string& ep, const framework::Scope* scope,
                         const std::string& var_name, int64_t time_out) {
    auto ch = _Get(ep);

    auto c = std::shared_ptr<ClientBase>(new SendClient(cq_, ch, ep));
    clients_[c.get()] = c;

    return c->Call(scope, var_name, time_out);
  }
  bool AsyncGetVariable(const std::string& ep, const framework::Scope* scope,
                        const std::string& var_name, int64_t time_out) {
    auto ch = _Get(ep);

    auto c = std::shared_ptr<ClientBase>(new GetClient(cq_, ch, ep));
    clients_[c.get()] = c;

    return c->Call(scope, var_name, time_out);
  }

  bool Proceed(SendStatus& s) {
    void* tag = NULL;
    bool ok = false;

    if (!cq_->Next(&tag, &ok)) {
      return false;
    }
    GPR_ASSERT(ok);
    PADDLE_ENFORCE(tag);

    auto cls = static_cast<ClientBase*>(tag);
    clients_.erase(tag);
    const grpc::Status& status = *cls->GetStatus();
    s = cls->GetSendStatus();

    s.end = std::chrono::system_clock::now();
    if (status.ok()) {
      cls->ProcRet();
      return true;
    }

    if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
      s.error = "rpc timed out";
    } else {
      std::ostringstream stringStream;
      stringStream << "rpc failed because:" << status.error_code();
      s.error = stringStream.str();
    }

    return true;
  }

 private:
  std::shared_ptr<grpc::Channel> _Get(const std::string& ep) {
    auto it = channels_.find(ep);
    if (it != channels_.end()) {
      return it->second;
    }

    auto ch = std::shared_ptr<grpc::Channel>(
        grpc::CreateChannel(ep, grpc::InsecureChannelCredentials()));

    channels_[ep] = ch;
    return ch;
  }

 private:
  std::shared_ptr<grpc::CompletionQueue> cq_;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels_;

  // even if user don't call proceed,
  // the ClientBase will release automaticly
  std::map<void*, std::shared_ptr<ClientBase>> clients_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
