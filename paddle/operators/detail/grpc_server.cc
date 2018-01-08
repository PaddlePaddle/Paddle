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
#include <future>

using grpc::ServerAsyncResponseWriter;

namespace paddle {
namespace operators {
namespace detail {

enum CallStatus { CREATE = 0, PROCESS, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  explicit RequestBase(sendrecv::SendRecvService::AsyncService* service,
                       grpc::ServerCompletionQueue* cq)
      : service_(service), cq_(cq) {}
  virtual ~RequestBase() {}
  virtual void Proceed() = 0;

  CallStatus Status() { return status_; }

 protected:
  grpc::ServerContext ctx_;
  sendrecv::SendRecvService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
};

typedef std::pair<std::string, sendrecv::VariableMessage> MessageWithName;

class RequestSend final : public RequestBase {
 public:
  explicit RequestSend(sendrecv::SendRecvService::AsyncService* service,
                       grpc::ServerCompletionQueue* cq,
                       SimpleBlockQueue<MessageWithName>* queue)
      : RequestBase(service, cq), queue_(queue), responder_(&ctx_) {
    Proceed();
  }

  virtual ~RequestSend() {}

  void SendVariable() {
    MessageWithName msg_with_name =
        std::make_pair(request_.varname(), std::move(request_));
    queue_->Push(std::move(msg_with_name));
  }

  virtual void Proceed() {
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestSendVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                    this);
      VLOG(4) << "create RequestSend" << std::endl;
    } else if (status_ == PROCESS) {
      new RequestSend(service_, cq_, queue_);

      // The actual processing.
      SendVariable();

      status_ = FINISH;
      responder_.Finish(reply_, grpc::Status::OK, this);
      VLOG(4) << "Process RequestSend" << std::endl;
    } else {
      VLOG(4) << "delete RequestSend" << std::endl;
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 protected:
  sendrecv::VariableMessage request_;
  sendrecv::VoidMessage reply_;
  SimpleBlockQueue<MessageWithName>* queue_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(sendrecv::SendRecvService::AsyncService* service,
                      grpc::ServerCompletionQueue* cq, framework::Scope* scope)
      : RequestBase(service, cq), responder_(&ctx_), scope_(scope) {
    Proceed();
  }

  virtual ~RequestGet() {}

  void GetVariable() {
    std::string var_name = request_.varname();
    auto* var = scope_->FindVar(var_name);

    // FIXME(gongwb): device context?
    SerializeToMessage(var_name, var, platform::CPUDeviceContext(), &reply_);
  }

  virtual void Proceed() {
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestGetVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                   this);
    } else if (status_ == PROCESS) {
      new RequestGet(service_, cq_, scope_);

      // The actual processing.
      status_ = FINISH;
      responder_.Finish(reply_, grpc::Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 protected:
  sendrecv::VariableMessage request_;
  sendrecv::VariableMessage reply_;
  ServerAsyncResponseWriter<sendrecv::VariableMessage> responder_;
  framework::Scope* scope_;
};

void AsyncGRPCServer::RunSyncUpdate() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);

  cq_send_ = builder.AddCompletionQueue();
  cq_get_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << address_ << std::endl;

  t_send_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleReqSend, this)));
  t_get_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleReqGet, this, true)));

  // wait server
  server_->Wait();
}

void AsyncGRPCServer::ShutDown() {
  cq_send_->Shutdown();
  t_send_->join();

  cq_get_->Shutdown();
  t_get_->join();

  server_->Shutdown();
}

void AsyncGRPCServer::HandleReqSend() {
  RequestSend* req_send =
      new RequestSend(&service_, cq_send_.get(), &var_recv_queue_);
  VLOG(4) << "RequestSend status" << req_send->Status();
  void* tag = NULL;
  bool ok = false;
  while (true) {
    if (!cq_send_->Next(&tag, &ok)) {
      LOG(ERROR) << "HandleReqSend meets CompletionQueue errors" << std::endl;
      break;
    }
    if (!ok) {
      LOG(WARNING) << "HandleReqSend meets not handled events" << std::endl;
      continue;
    }
    static_cast<RequestBase*>(tag)->Proceed();
  }
}

void AsyncGRPCServer::HandleReqGet(bool wait) {
  RequestGet* req_get = new RequestGet(&service_, cq_get_.get(), scope_);
  VLOG(4) << "RequestSend status:" << req_get->Status();

  void* tag = NULL;
  bool ok = false;
  while (true) {
    if (!cq_get_->Next(&tag, &ok)) {
      LOG(ERROR) << "HandleReqSend meets CompletionQueue errors" << std::endl;
      break;
    }
    if (!ok) {
      LOG(WARNING) << "HandleReqSend meets not handled events" << std::endl;
      continue;
    }

    if (wait && !done_) {
      Wait();
    }
    static_cast<RequestBase*>(tag)->Proceed();
  }
}

void AsyncGRPCServer::Wait() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  condition_.wait(lock, [=] { return this->done_ == true; });
}

void AsyncGRPCServer::Reset() {
  std::lock_guard<std::mutex> lock(this->mutex_);
  done_ = false;
}

void AsyncGRPCServer::Done() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    done_ = true;
  }
  condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
