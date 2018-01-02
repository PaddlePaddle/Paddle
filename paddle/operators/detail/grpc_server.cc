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

#include "grpc_server.h"

using grpc::ServerAsyncResponseWriter;

namespace paddle {
namespace operators {
namespace detail {

enum CallStatus { CREATE, PROCESS, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  virtual void Proceed() = 0;
};

class RequestSend final : public RequestBase {
 public:
  RequestSend(SendRecvService::AsyncService* service, ServerCompletionQueue* cq)
      : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
    Proceed();
  }

  void Proceed() {
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestSendVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                    this);
    } else if (status_ == PROCESS) {
      new RequestSend(service_, cq_);

      // The actual processing.
      // std::string prefix("Hello ");
      // reply_.set_message(prefix + request_.name());

      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 private:
  VariableMessage request_;
  VoidMessage reply_;
  ServerContext ctx_;

  SendRecvService::AsyncService* service_;
  ServerCompletionQueue* cq_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
  CallStatus status_;
};

class RequestGet final : public RequestBase {
 public:
  RequestGet(SendRecvService::AsyncService* service, ServerCompletionQueue* cq)
      : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
    Proceed();
  }

  void Proceed() {
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestGetVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                   this);
    } else if (status_ == PROCESS) {
      new RequestGet(service_, cq_);

      // The actual processing.
      // std::string prefix("Hello ");
      // reply_.set_message(prefix + request_.name());

      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 private:
  VariableMessage request_;
  VariableMessage reply_;
  ServerContext ctx_;

  SendRecvService::AsyncService* service_;
  ServerCompletionQueue* cq_;
  ServerAsyncResponseWriter<VariableMessage> responder_;
  CallStatus status_;
};

// There is no shutdown handling in this code.
void AsyncGRPCServer::Run() {
  ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());

  builder.RegisterService(&service_);
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  // std::cout << "Server listening on " << server_address << std::endl;

  HandleRpcs();
}

// This can be run in multiple threads if needed.
void AsyncGRPCServer::HandleRpcs() {
  new RequestSend(&service_, cq_.get());
  new RequestGet(&service_, cq_.get());

  void* tag = NULL;
  bool ok = false;
  while (true) {
    GPR_ASSERT(cq_->Next(&tag, &ok));
    GPR_ASSERT(ok);
    static_cast<RequestBase*>(tag)->Proceed();
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
