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
    s << "name:[" << name << "] ep:[" << ep << "]";
    return s.str();
  }
};

struct SendStatus {
  std::string error;
  VarHandle var;

  std::string String() const {
    std::ostringstream s;

    s << var.String();

    if (error != "") {
      s << " error_info:[" << error << "]";
    }

    return s.str();
  }
};

class ClientProcessor {
 public:
  ClientProcessor(grpc::CompletionQueue* cq, std::shared_ptr<grpc::Channel> ch,
                  std::string ep) {
    cq_ = cq;
    ch_ = ch;
    ep_ = ep;
    stub_ = sendrecv::SendRecvService::NewStub(ch);
    context_.reset(new grpc::ClientContext());
    status_.reset(new grpc::Status());
  }

  virtual ~ClientProcessor() {}

  virtual bool Call(const framework::Scope* scope, const std::string& name,
                    int64_t time_out) = 0;

  virtual bool ProcRet() { return true; }

  virtual grpc::Status* GetStatus() { return status_.get(); }

  virtual SendStatus GetSendStatus() { return send_status_; }

 protected:
  typedef std::chrono::system_clock::time_point time_point;

  grpc::CompletionQueue* cq_;
  std::shared_ptr<grpc::Channel> ch_;
  std::string ep_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> context_;
  std::unique_ptr<grpc::Status> status_;
  SendStatus send_status_;
};

class SendProcessor : public ClientProcessor {
 public:
  SendProcessor(grpc::CompletionQueue* cq, std::shared_ptr<grpc::Channel> ch,
                std::string ep)
      : ClientProcessor(cq, ch, ep) {}

  virtual ~SendProcessor() {}

  virtual bool Call(const framework::Scope* scope, const std::string& name,
                    int64_t time_out) {
    // record send status
    auto start = std::chrono::system_clock::now();
    send_status_.var.ep = ep_;
    send_status_.var.name = name;

    const time_point deadline = start + std::chrono::milliseconds(time_out);
    context_->set_deadline(deadline);

    sendrecv::VariableMessage req;
    Request(scope, name, &req);

    reply_.reset(new sendrecv::VoidMessage());
    auto rpc = stub_->AsyncSendVariable(context_.get(), req, cq_);
    rpc->Finish(reply_.get(), status_.get(), (void*)this);

    return true;
  }

 private:
  void Request(const framework::Scope* scope, const std::string name,
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

 protected:
  std::unique_ptr<sendrecv::VoidMessage> reply_;
};

class GetProcessor : public ClientProcessor {
 public:
  GetProcessor(grpc::CompletionQueue* cq, std::shared_ptr<grpc::Channel> ch,
               std::string ep)
      : ClientProcessor(cq, ch, ep) {
    reply_.reset(new sendrecv::VoidMessage());
  }

  virtual ~GetProcessor() {}

  virtual bool Call(const framework::Scope* scope, const std::string& name,
                    int64_t time_out) {
    return true;
  }

  virtual bool ProcRet() { return true; }

 protected:
  std::unique_ptr<sendrecv::VoidMessage> reply_;
};

class RPCClient {
 public:
  bool AsyncSendVariable(const std::string& ep, const framework::Scope* scope,
                         const std::string& var_name, int64_t time_out) {
    auto ch = GetChannel(ep);

    auto c = std::shared_ptr<ClientProcessor>(new SendProcessor(&cq_, ch, ep));
    clients_[c.get()] = c;

    count_++;

    return c->Call(scope, var_name, time_out);
  }
  bool AsyncGetVariable(const std::string& ep, const framework::Scope* scope,
                        const std::string& var_name, int64_t time_out) {
    auto ch = GetChannel(ep);

    auto c = std::shared_ptr<ClientProcessor>(new GetProcessor(&cq_, ch, ep));
    clients_[c.get()] = c;

    count_++;

    return c->Call(scope, var_name, time_out);
  }

  bool wait() {
    bool ok = true;

    while (true) {
      if (count_ <= 0) {
        break;
      }

      detail::SendStatus s;
      if (!Proceed(s)) {
        LOG(ERROR) << "Get meets CompletionQueue error";
        return false;
      }

      // TODO(gongwb): add more retry?
      if (s.error != "") {
        ok = false;
        LOG(ERROR) << "sync update variable error:" << s.String();
        continue;
      }
      VLOG(3) << "sync update variable ok:" << s.String();
    }

    return ok;
  }

  bool Proceed(SendStatus& s) {
    void* tag = NULL;
    bool ok = false;

    if (!cq_.Next(&tag, &ok)) {
      return false;
    }
    count_--;
    GPR_ASSERT(ok);
    PADDLE_ENFORCE(tag);

    ClientProcessor* cls = (ClientProcessor*)tag;
    const grpc::Status& status = *cls->GetStatus();
    s = cls->GetSendStatus();

    if (status.ok()) {
      cls->ProcRet();
      clients_.erase(tag);
      return true;
    }

    if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
      s.error = "rpc timed out";
    } else {
      std::ostringstream stringStream;
      stringStream << "rpc failed because:" << status.error_code();
      s.error = stringStream.str();
    }

    clients_.erase(tag);
    return true;
  }

 private:
  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep) {
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
  grpc::CompletionQueue cq_;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels_;

  // even if user don't call proceed,
  // the ClientProcessor will release automaticly
  std::map<void*, std::shared_ptr<ClientProcessor>> clients_;

  int64_t count_ = 0;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
