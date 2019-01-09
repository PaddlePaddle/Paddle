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
#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_service.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace distributed {

using ::grpc::ServerAsyncResponseWriter;

enum CallStatus { PROCESS = 0, FINISH };

// GRPCRequestBase is used for fetch out incomming requests from
// GRPC clients. It's quite different to general RPCRequest class
// for:
//
// GRPCRequestBase: implementation for GRPC, call handlers when a
//                  request arrives.
// RPCRequest:      decoupled from rpc implementations, is the input
//                  of handlers.

class GRPCRequestBase {
 public:
  explicit GRPCRequestBase(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           RequestHandler* request_handler, int req_id,
                           int method_id)
      : responder_(&ctx_),
        service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    PADDLE_ENFORCE(cq_);
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx()));
    // GRPC request is initialized in here:
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~GRPCRequestBase() {}
  virtual void Process() = 0;

  std::string Status2String(RequestType req_type) {
    std::string status = "Process";
    if (status_ == FINISH) {
      status = "Finish";
    }

    std::ostringstream s;
    s << req_type << " name:[" << request_->Varname() << "]"
      << ", ep:[" << ctx_.peer() << "]"
      << " " << status << " using req_id:" << req_id_;
    return s.str();
  }

  CallStatus Status() const {
    std::lock_guard<std::mutex> l(status_mu_);
    return status_;
  }

  void Finish(const ::grpc::ByteBuffer& reply,
              ServerAsyncResponseWriter<::grpc::ByteBuffer>* responder) {
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
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    std::string varname = request_->Varname();
    VLOG(4) << "RequestSend var_name:" << varname;

    auto scope = request_->GetMutableLocalScope();
    auto invar = request_->GetVar();
    int trainer_id = request_->GetTrainerId();
    framework::Variable* outvar = nullptr;

    RPCRequest req(varname, invar, &outvar, trainer_id, RequestType::SEND);
    request_handler_->Handle(&req, scope);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGet final : public GRPCRequestBase {
 public:
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    std::string varname = request_->Varname();
    int trainer_id = request_->GetTrainerId();
    VLOG(4) << "RequestGet " << varname;

    auto scope = request_handler_->scope();
    auto invar = scope->FindVar(varname);
    framework::Variable* outvar = nullptr;

    RPCRequest req(varname, invar, &outvar, trainer_id, RequestType::RECV);
    request_handler_->Handle(&req, scope);

    if (outvar) {
      SerializeToByteBuffer(varname, outvar, *request_handler_->dev_ctx(),
                            &reply_);
    }
    Finish(reply_, &responder_);
  }
};

class GRPCRequestPrefetch final : public GRPCRequestBase {
 public:
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    std::string in_var_name = request_->Varname();
    std::string out_var_name = request_->OutVarname();
    // std::string table_name = request_->TableName();
    int trainer_id = request_->GetTrainerId();
    VLOG(4) << "RequestPrefetch, in_var_name: " << in_var_name
            << " out_var_name: " << out_var_name;

    auto scope = request_->GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    framework::Variable* outvar = scope->Var(out_var_name);

    RPCRequest req(in_var_name, invar, &outvar, trainer_id,
                   RequestType::PREFETCH);
    req.out_var_name_ = out_var_name;  // FIXME(typhoonzer): copy
    req.table_name_ = request_->TableName();
    request_handler_->Handle(&req, scope);

    SerializeToByteBuffer(out_var_name, outvar, *request_handler_->dev_ctx(),
                          &reply_);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestCheckpointNotify final : public GRPCRequestBase {
 public:
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    auto scope = request_->GetMutableLocalScope();

    std::string checkpoint_notify = request_->Varname();
    std::string checkpoint_dir = request_->OutVarname();
    int trainer_id = request_->GetTrainerId();

    VLOG(4) << "RequestCheckpointNotify notify: " << checkpoint_notify
            << ", dir: " << checkpoint_dir;

    RPCRequest req(request_->Varname(), nullptr, nullptr, trainer_id,
                   RequestType::CHECKPOINT);
    req.out_var_name_ = checkpoint_notify;
    request_handler_->Handle(&req, scope);
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGetMonomer final : public GRPCRequestBase {
 public:
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    std::string varname = request_->Varname();

    auto scope = request_handler_->scope();
    auto invar = scope->FindVar(varname);
    framework::Variable* outvar = nullptr;

    RPCRequest req(varname, invar, &outvar, request_->GetTrainerId(),
                   RequestType::GET_MONOMER);
    request_handler_->Handle(&req, scope);

    if (outvar) {
      auto* dev_ctx = request_handler_->dev_ctx();
      SerializeToByteBuffer(varname, outvar, *dev_ctx, &reply_);
    }
    Finish(reply_, &responder_);
  }
};

class GRPCRequestGetMonomerBarrier final : public GRPCRequestBase {
 public:
  using GRPCRequestBase::GRPCRequestBase;
  void Process() override {
    std::string varname = request_->Varname();
    VLOG(4) << "RequestGetMonomerBarrier " << varname;

    // FIXME(typhoonzero): wait var ready in handler
    RPCRequest req(varname, nullptr, nullptr, request_->GetTrainerId(),
                   RequestType::GET_MONOMER_BARRIER);
    request_handler_->Handle(&req, nullptr);

    Finish(reply_, &responder_);
  }
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
