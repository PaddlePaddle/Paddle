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

#include <memory>
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

// GRPCRequest is used for fetch out incomming requests from
// GRPC clients. It's quite different to general RPCRequest class
// for:
//
// GRPCRequest: implementation for GRPC, generate requests, call handlers
//              then return the response to client side.
// RPCRequest:  decoupled from rpc implementations, is the input
//              of handlers.

class GRPCRequest {
 public:
  explicit GRPCRequest(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id,
                       RequestType req_type)
      : responder_(&ctx_),
        req_type_(req_type),
        service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    PADDLE_ENFORCE(cq_);
    rpc_request_.reset(new RPCRequest());
    auto get_var_callback =
        std::bind(&RequestHandler::GetOrCreateRequestVar, request_handler_,
                  std::placeholders::_1, rpc_request_.get());
    request_.reset(new GRPCVariableResponse(get_var_callback,
                                            request_handler_->dev_ctx()));
    service_->RequestAsyncUnary(
        static_cast<int>(req_type_), &ctx_, request_.get(), &responder_, cq_,
        cq_, reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }
  virtual ~GRPCRequest() {}

  void Process() {
    rpc_request_->Prepare(request_->Varname(), request_->GetVar(),
                          request_->OutVarname(), request_->TableName(),
                          request_->GetTrainerId(), req_type_);
    request_handler_->Handle(rpc_request_.get());
    Finish();
  }

  void Finish() {
    ::grpc::ByteBuffer reply;
    // NOTE: response is in out_var_ and out_var_name_
    if (rpc_request_->out_var_ != nullptr) {
      SerializeToByteBuffer(rpc_request_->out_var_name_, rpc_request_->out_var_,
                            *request_handler_->dev_ctx(), &reply);
    }
    if (rpc_request_->scope_) {
      request_handler_->scope()->DeleteScope(rpc_request_->scope_);
    }
    {
      std::lock_guard<std::mutex> l(status_mu_);
      status_ = FINISH;
    }
    responder_.Finish(reply, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

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

 protected:
  // FIXME(typhoonzero): to unique_ptr
  std::shared_ptr<GRPCVariableResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;

  std::unique_ptr<RPCRequest> rpc_request_;
  RequestType req_type_;

  mutable std::mutex status_mu_;
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  RequestHandler* request_handler_;
  int req_id_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
