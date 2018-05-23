/*Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detail/grpc_server.h"

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
                       GRPCProcessorCtx* rpc_processor)
      : service_(service),
        cq_(cq),
        status_(PROCESS),
        rpc_processor_(rpc_processor) {
    PADDLE_ENFORCE(cq_);
  }
  virtual ~RequestBase() {}
  virtual void Process() { assert(false); }

  CallStatus Status() { return status_; }
  void SetStatus(CallStatus status) { status_ = status; }
  virtual std::string GetReqName() = 0;

 protected:
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  GRPCProcessorCtx* rpc_processor_;
};

class RequestSend final : public RequestBase {
 public:
  explicit RequestSend(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       GRPCProcessorCtx* rpc_processor)
      : RequestBase(service, cq, rpc_processor), responder_(&ctx_) {
    if (rpc_processor->sync_mode()) {
      request_.reset(
          new VariableResponse(scope, rpc_processor->dev_ctx(), false));
    } else {
      request_.reset(
          new VariableResponse(scope, rpc_processor->dev_ctx(), true));
    }
    int method_id = static_cast<int>(detail::GrpcMethod::kSendVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_.get(), &responder_,
                                cq_, cq_, this);
  }

  virtual ~RequestSend() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    std::string var_name = GetReqName();
    VLOG(3) << "RequestSend " << var_name;

    rpc_processor_->RequestSend(request_.get());

    sendrecv::VoidMessage reply;
    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(GrpcService::AsyncService* service,
                      ::grpc::ServerCompletionQueue* cq,
                      framework::BlockingQueue<MessageWithName>* queue,
                      GRPCProcessorCtx* rpc_processor)
      : RequestBase(service, cq, rpc_processor),
        responder_(&ctx_),
        queue_(queue) {
    auto method_id = static_cast<int>(detail::GrpcMethod::kGetVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, &request_, &responder_, cq_,
                                cq_, this);
  }

  virtual ~RequestGet() {}

  virtual std::string GetReqName() { return request_.varname(); }

  virtual void Process() {
    // proc request.
    std::string var_name = request_.varname();
    VLOG(3) << "RequestGet " << var_name;

    ::grpc::ByteBuffer reply;
    if (var_name != FETCH_BARRIER_MESSAGE) {
      rpc_processor_->RequestGet(request_.get(), &reply);
    }

    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;

    if (var_name == FETCH_BARRIER_MESSAGE) {
      sendrecv::VariableMessage msg;
      MessageWithName msg_with_name = std::make_pair(var_name, msg);
      queue_->Push(msg_with_name);
    }
  }

 protected:
  sendrecv::VariableMessage request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::BlockingQueue<MessageWithName>* queue_;
};

class RequestPrefetch final : public RequestBase {
 public:
  explicit RequestPrefetch(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           GRPCProcessorCtx* rpc_processor)
      : RequestBase(service, cq, rpc_processor), responder_(&ctx_) {
    if (rpc_processor->sync_mode()) {
      request_.reset(
          new VariableResponse(scope, rpc_processor->dev_ctx(), false));
    } else {
      request_.reset(
          new VariableResponse(scope, rpc_processor->dev_ctx(), true));
    }
    int method_id = static_cast<int>(detail::GrpcMethod::kPrefetchVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_.get(), &responder_,
                                cq_, cq_, this);
  }

  virtual ~RequestPrefetch() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    // prefetch process...
    ::grpc::ByteBuffer reply;

    rpc_processor->RequestPrefetch(request_.get(), &reply);

    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

void AsyncGRPCServer::WaitClientGet(int count) {
  int fetch_barriers = 0;
  while (fetch_barriers < count) {
    auto msg = var_get_queue_.Pop();
    if (msg.first == FETCH_BARRIER_MESSAGE) {
      fetch_barriers++;
    }
  }
}

void AsyncGRPCServer::WaitServerReady() {
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
}

void AsyncGRPCServer::RunSyncUpdate() {
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
      std::bind(&AsyncGRPCServer::TryToRegisterNewSendOne, this);
  std::function<void()> get_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewGetOne, this);
  std::function<void()> prefetch_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewPrefetchOne, this);

  // TODO(wuyi): Run these "HandleRequest" in thread pool
  t_send_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleRequest, this,
                                cq_send_.get(), "cq_send", send_register)));
  t_get_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleRequest, this,
                                cq_get_.get(), "cq_get", get_register)));
  t_prefetch_.reset(new std::thread(
      std::bind(&AsyncGRPCServer::HandleRequest, this, cq_prefetch_.get(),
                "cq_prefetch", prefetch_register)));

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();
  // wait server
  server_->Wait();
  t_send_->join();
  t_get_->join();
  t_prefetch_->join();
}

void AsyncGRPCServer::ShutdownQueue() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  cq_send_->Shutdown();
  cq_get_->Shutdown();
  cq_prefetch_->Shutdown();
}

// This URL explains why shutdown is complicate:
void AsyncGRPCServer::ShutDown() {
  is_shut_down_ = true;
  ShutdownQueue();
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewSendOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }
  RequestSend* send =
      new RequestSend(&service_, cq_send_.get(), rpc_processor_);
  VLOG(4) << "Create RequestSend status:" << send->Status();
}

void AsyncGRPCServer::TryToRegisterNewGetOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewGetOne";
    return;
  }
  RequestGet* get =
      new RequestGet(&service_, cq_get_.get(), &var_get_queue_, rpc_processor_);
  VLOG(4) << "Create RequestGet status:" << get->Status();
}

void AsyncGRPCServer::TryToRegisterNewPrefetchOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewPrefetchOne";
    return;
  }
  RequestPrefetch* prefetch =
      new RequestPrefetch(&service_, cq_prefetch_.get(), rpc_processor_);

  VLOG(4) << "Create RequestPrefetch status:" << prefetch->Status();
}

// FIXME(typhoonzero): change cq_name to enum.
void AsyncGRPCServer::HandleRequest(::grpc::ServerCompletionQueue* cq,
                                    const std::string& cq_name,
                                    std::function<void()> TryToRegisterNewOne) {
  TryToRegisterNewOne();

  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(3) << "HandleRequest for " << cq_name << " wait Next";
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << cq_name << " CompletionQueue shutdown!";
      break;
    }
    VLOG(3) << "HandleRequest for " << cq_name << " get Next";

    PADDLE_ENFORCE(tag);

    if (sync_mode_) {
      // FIXME(typhoonzero): de-couple the barriers with recv_op
      if (!is_shut_down_ && cq_name == "cq_get") WaitCond(1);
      if (!is_shut_down_ && cq_name == "cq_send") WaitCond(0);
      VLOG(3) << "HandleRequest for " << cq_name << " after WaitCond";
    }

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
        TryToRegisterNewOne();
        base->Process();
        VLOG(4) << cq_name << " PROCESS status:" << base->Status();
        break;
      }
      case FINISH: {
        VLOG(4) << cq_name << " FINISH status:" << base->Status();
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

void AsyncGRPCServer::WaitCond(int cond) {
  std::unique_lock<std::mutex> lock(this->barrier_mutex_);
  barrier_condition_.wait(lock,
                          [=] { return this->barrier_cond_step_ == cond; });
}

void AsyncGRPCServer::SetCond(int cond) {
  {
    std::lock_guard<std::mutex> lock(this->barrier_mutex_);
    barrier_cond_step_ = cond;
  }
  barrier_condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
