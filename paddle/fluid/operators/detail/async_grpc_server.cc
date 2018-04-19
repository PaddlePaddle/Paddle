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

#include "paddle/fluid/operators/detail/async_grpc_server.h"

#include <limits>
#include <string>

using ::grpc::ServerAsyncResponseWriter;

namespace paddle {
namespace operators {
namespace detail {

enum CallStatus { PROCESS = 0, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  explicit RequestBase(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       const platform::DeviceContext* dev_ctx)
      : service_(service), cq_(cq), status_(PROCESS), dev_ctx_(dev_ctx) {
    PADDLE_ENFORCE(cq_);
  }
  virtual ~RequestBase() {}
  virtual void Process() { assert(false); }

  CallStatus Status() { return status_; }
  void SetStatus(CallStatus status) { status_ = status; }
  virtual std::string GetReqName() {
    assert(false);
    return "";
  }

 protected:
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  const platform::DeviceContext* dev_ctx_;
};

class RequestSend final : public RequestBase {
 public:
  explicit RequestSend(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       framework::Scope* scope, ReceivedQueue* queue,
                       const platform::DeviceContext* dev_ctx)
      : RequestBase(service, cq, dev_ctx), queue_(queue), responder_(&ctx_) {
    request_.reset(new VariableResponse(scope, dev_ctx_));
    int method_id = static_cast<int>(detail::GrpcMethod::kSendVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_.get(), &responder_,
                                cq_, cq_, this);
  }

  virtual ~RequestSend() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    queue_->Push(std::make_pair(request_->Varname(), request_));

    sendrecv::VoidMessage reply;
    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ReceivedQueue* queue_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(GrpcService::AsyncService* service,
                      ::grpc::ServerCompletionQueue* cq,
                      framework::Scope* scope,
                      const platform::DeviceContext* dev_ctx)
      : RequestBase(service, cq, dev_ctx), responder_(&ctx_), scope_(scope) {
    auto method_id = static_cast<int>(detail::GrpcMethod::kGetVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, &request_, &responder_, cq_,
                                cq_, this);
  }

  virtual ~RequestGet() {}

  virtual std::string GetReqName() { return request_.varname(); }

  virtual void Process() {
    // proc request.
    std::string var_name = request_.varname();
    auto* var = scope_->FindVar(var_name);

    ::grpc::ByteBuffer reply;
    SerializeToByteBuffer(var_name, var, *dev_ctx_, &reply);

    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  sendrecv::VariableMessage request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::Scope* scope_;
};

class RequestPrefetch final : public RequestBase {
 public:
  explicit RequestPrefetch(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           framework::Scope* scope,
                           const platform::DeviceContext* dev_ctx,
                           framework::Executor* executor,
                           framework::ProgramDesc* program,
                           framework::ExecutorPrepareContext* prefetch_ctx)
      : RequestBase(service, cq, dev_ctx),
        responder_(&ctx_),
        scope_(scope),
        executor_(executor),
        program_(program),
        prefetch_ctx_(prefetch_ctx) {
    request_.reset(new VariableResponse(scope, dev_ctx_));
    int method_id = static_cast<int>(detail::GrpcMethod::kPrefetchVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_.get(), &responder_,
                                cq_, cq_, this);
  }

  virtual ~RequestPrefetch() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    // prefetch process...
    ::grpc::ByteBuffer reply;

    std::string var_name = request_->OutVarname();
    VLOG(3) << "prefetch var " << var_name;
    auto var_desc = program_->Block(0).FindVar(var_name);
    framework::Scope* local_scope = &scope_->NewScope();
    auto* var = local_scope->FindVar(var_name);
    InitializeVariable(var, var_desc->GetType());
    executor_->RunPreparedContext(prefetch_ctx_, scope_, false, false);

    SerializeToByteBuffer(var_name, var, *dev_ctx_, &reply);

    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::Scope* scope_;
  framework::Executor* executor_;
  framework::ProgramDesc* program_;
  framework::ExecutorPrepareContext* prefetch_ctx_;
};

void SyncGRPCServer::RunAsyncUpdate() {
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, ::grpc::InsecureServerCredentials(),
                           &selected_port_);
  builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  builder.RegisterService(&service_);

  cq_send_ = builder.AddCompletionQueue();
  cq_get_ = builder.AddCompletionQueue();
  cq_prefetch_ = builder.AddCompletionQueue();

  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << address_
            << " selected port: " << selected_port_;

  std::function<void()> send_register =
      std::bind(&SyncGRPCServer::TryToRegisterNewSendOne, this);
  std::function<void()> get_register =
      std::bind(&SyncGRPCServer::TryToRegisterNewGetOne, this);
  std::function<void()> prefetch_register =
      std::bind(&SyncGRPCServer::TryToRegisterNewPrefetchOne, this);

  // TODO(wuyi): Run these "HandleRequest" in thread pool
  t_send_.reset(
      new std::thread(std::bind(&SyncGRPCServer::HandleRequest, this,
                                cq_send_.get(), "cq_send", send_register)));
  t_get_.reset(
      new std::thread(std::bind(&SyncGRPCServer::HandleRequest, this,
                                cq_get_.get(), "cq_get", get_register)));
  t_prefetch_.reset(new std::thread(
      std::bind(&SyncGRPCServer::HandleRequest, this, cq_prefetch_.get(),
                "cq_prefetch", prefetch_register)));
  // wait server
  server_->Wait();
  t_send_->join();
  t_get_->join();
  t_prefetch_->join();
}

void SyncGRPCServer::ShutdownQueue() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  cq_send_->Shutdown();
  cq_get_->Shutdown();
  cq_prefetch_->Shutdown();
}

// This URL explains why shutdown is complicate:
void SyncGRPCServer::ShutDown() {
  is_shut_down_ = true;
  ShutdownQueue();
  server_->Shutdown();
}

void SyncGRPCServer::TryToRegisterNewSendOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }
  RequestSend* send = new RequestSend(&service_, cq_send_.get(), scope_,
                                      &var_recv_queue_, dev_ctx_);
  VLOG(4) << "Create RequestSend status:" << send->Status();
}

void SyncGRPCServer::TryToRegisterNewGetOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewGetOne";
    return;
  }
  RequestGet* get = new RequestGet(&service_, cq_get_.get(), scope_, dev_ctx_);
  VLOG(4) << "Create RequestGet status:" << get->Status();
}

void SyncGRPCServer::TryToRegisterNewPrefetchOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewPrefetchOne";
    return;
  }
  RequestPrefetch* prefetch =
      new RequestPrefetch(&service_, cq_prefetch_.get(), scope_, dev_ctx_,
                          executor_, program_, prefetch_ctx_);

  VLOG(4) << "Create RequestPrefetch status:" << prefetch->Status();
}

void SyncGRPCServer::HandleRequest(::grpc::ServerCompletionQueue* cq,
                                   const std::string& cq_name,
                                   std::function<void()> TryToRegisterNewOne) {
  TryToRegisterNewOne();

  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(3) << "HandleRequest for " << cq_name << " while in";
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << cq_name << " CompletionQueue shutdown!";
      break;
    }
    VLOG(3) << "HandleRequest for " << cq_name << " while after Next";

    PADDLE_ENFORCE(tag);

    RequestBase* base = reinterpret_cast<RequestBase*>(tag);
    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      LOG(WARNING) << cq_name << " recv no regular event:argument name["
                   << base->GetReqName() << "]";
      TryToRegisterNewOne();
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        VLOG(4) << cq_name << " status:" << base->Status();
        TryToRegisterNewOne();
        base->Process();
        break;
      }
      case FINISH: {
        VLOG(4) << cq_name << " status:" << base->Status();
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
