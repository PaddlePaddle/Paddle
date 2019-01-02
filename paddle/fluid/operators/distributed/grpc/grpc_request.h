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

#include <string>
#include "grpc++/grpc++.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace distributed {

enum CallStatus { PROCESS = 0, FINISH };

// NOTE: We use GRPCRequestBase base class for all the rpc call types.
class GRPCRequestBase {
 public:
  explicit RequestBase(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : responder_(&ctx_),
        service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    PADDLE_ENFORCE(cq_);
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx(),
                                            !request_handler->sync_mode()));
    int method_id = static_cast<int>(distributed::GrpcMethod::kSendVariable);
    // GRPC request is initialized in here:
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestBase() {}
  virtual void Process() = 0;

  std::string Status2String(const std::string& method) {
    std::string status = "Process";
    if (status_ == FINISH) {
      status = "Finish";
    }

    std::ostringstream s;
    s << method << " name:[" << GetReqName() << "]"
      << ", ep:[" << ctx_.peer() << "]"
      << " " << status << " using req_id:" << req_id_;
    return s.str();
  }

  CallStatus Status() const {
    std::lock_guard<std::mutex> l(status_mu_);
    return status_;
  }

  template <typename T>
  void Finish(const T& reply, ServerAsyncResponseWriter<T>* responder) {
    std::lock_guard<std::mutex> l(status_mu_);
    status_ = FINISH;
    responder->Finish(reply, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  // FIXME(typhoonzero): to unique_ptr
  std::shared_ptr<GRPCVariableResponse> request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;

  mutable std::mutex status_mu_;
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  RequestHandler* request_handler_;
  int req_id_;
};

class GRPCRequestSend final : public GRPCRequestBase {
 public:
  void Process() override {
    std::string varname = request_->Varname();
    VLOG(4) << "RequestSend var_name:" << varname;

    auto scope = request_->GetMutableLocalScope();
    auto invar = request_->GetVar();
    int trainer_id = request_->GetTrainerId();
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar, trainer_id);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGet final : public GRPCRequestBase {
 public:
  void Process() override {
    std::string varname = request_->Varname();
    int trainer_id = request_.trainer_id();
    VLOG(4) << "RequestGet " << varname;

    auto scope = request_handler_->scope();
    auto invar = scope->FindVar(varname);
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar, trainer_id);

    if (outvar) {
      SerializeToByteBuffer(varname, outvar, *request_handler_->dev_ctx(),
                            &reply_);
    }
    Finish(reply_, &responder_);
  }
};

class RPCRequestPrefetch final : public RPCRequestBase {
 public:
  void Process() override {
    std::string in_var_name = request_->Varname();
    std::string out_var_name = request_->OutVarname();
    std::string table_name = request_->TableName();
    int trainer_id = request_->GetTrainerId();
    VLOG(4) << "RequestPrefetch, in_var_name: " << in_var_name
            << " out_var_name: " << out_var_name;

    auto scope = request_->GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    framework::Variable* outvar = scope->Var(out_var_name);

    request_handler_->Handle(in_var_name, scope, invar, &outvar, trainer_id,
                             out_var_name, table_name);

    SerializeToByteBuffer(out_var_name, outvar, *request_handler_->dev_ctx(),
                          &reply_);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestCheckpointNotify final : public GRPCRequestBase {
 public:
  void Process() override {
    auto scope = request_->GetMutableLocalScope();

    std::string checkpoint_notify = request_->Varname();
    std::string checkpoint_dir = request_->OutVarname();
    int trainer_id = request_->GetTrainerId();

    VLOG(4) << "RequestCheckpointNotify notify: " << checkpoint_notify
            << ", dir: " << checkpoint_dir;

    request_handler_->Handle(checkpoint_notify, scope, nullptr, nullptr,
                             trainer_id, checkpoint_dir);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGetMonomer final : public RequestBase {
 public:
  void Process() override {
    std::string varname = request_->Varname();

    // rpc_server_->WaitVarCond(varname);
    // MonomerHandle h = rpc_server_->GetMonomer(varname);
    // auto scope = h.scope_;
    auto scope = request_->GetMutableLocalScope();
    auto invar = scope->FindVar(varname);
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar,
                             request_.trainer_id());

    if (outvar) {
      SerializeToByteBuffer(varname, outvar, *h.dev_ctx_, &reply_);
    }
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGetMonomerBarrier final : public GRPCRequestBase {
 public:
  void Process() override {
    std::string varname = request_->Varname();
    VLOG(4) << "RequestGetMonomerBarrier " << varname;

    // rpc_server_->WaitVarCond(varname);
    // MonomerHandle h = rpc_server_->GetMonomer(varname);

    framework::Scope* scope = nullptr;
    framework::Variable* invar = nullptr;
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar,
                             request_.trainer_id());

    Finish(reply_, &responder_);
  }
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
