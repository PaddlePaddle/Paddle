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

DEFINE_int32(rpc_server_handle_send_threads, 20,
             "Number of threads used to handle send at rpc server.");
DEFINE_int32(rpc_server_handle_get_threads, 20,
             "Number of threads used to handle get at rpc server.");
DEFINE_int32(rpc_server_handle_prefetch_threads, 1,
             "Number of threads used to handle prefetch at rpc server.");

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
                       GRPCProcessorCtx* rpc_processor, int req_id)
      : RequestBase(service, cq, rpc_processor), responder_(&ctx_) {
    request_.reset(new VariableResponse(rpc_processor->scope(),
                                        rpc_processor->dev_ctx(),
                                        !rpc_processor->sync_mode()));
    int method_id = static_cast<int>(detail::GrpcMethod::kSendVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestSend() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    std::string var_name = GetReqName();
    VLOG(3) << "RequestSend " << var_name;

    rpc_processor_->RequestSend(request_);

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  sendrecv::VoidMessage reply_;
  std::shared_ptr<VariableResponse> request_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
  int req_id_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(GrpcService::AsyncService* service,
                      ::grpc::ServerCompletionQueue* cq,
                      framework::BlockingQueue<MessageWithName>* queue,
                      GRPCProcessorCtx* rpc_processor, int req_id)
      : RequestBase(service, cq, rpc_processor),
        responder_(&ctx_),
        queue_(queue) {
    auto method_id = static_cast<int>(detail::GrpcMethod::kGetVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

  virtual ~RequestGet() {}

  virtual std::string GetReqName() { return request_.varname(); }

  virtual void Process() {
    // proc request.
    std::string var_name = request_.varname();
    VLOG(3) << "RequestGet " << var_name;

    if (var_name != FETCH_BARRIER_MESSAGE) {
      rpc_processor_->RequestGet(&request_, &reply_);
    }

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));

    if (var_name == FETCH_BARRIER_MESSAGE) {
      sendrecv::VariableMessage msg;
      MessageWithName msg_with_name = std::make_pair(var_name, msg);
      queue_->Push(msg_with_name);
    }
  }

 protected:
  sendrecv::VariableMessage request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::BlockingQueue<MessageWithName>* queue_;
  int req_id_;
};

class RequestPrefetch final : public RequestBase {
 public:
  explicit RequestPrefetch(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           GRPCProcessorCtx* rpc_processor, int req_id)
      : RequestBase(service, cq, rpc_processor), responder_(&ctx_) {
    request_.reset(new VariableResponse(rpc_processor->scope(),
                                        rpc_processor->dev_ctx(),
                                        !rpc_processor->sync_mode()));

    int method_id = static_cast<int>(detail::GrpcMethod::kPrefetchVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

  virtual ~RequestPrefetch() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    // prefetch process...

    rpc_processor_->RequestPrefetch(request_.get(), &reply_);

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;

  int req_id_;
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

  std::function<void(int)> send_register = std::bind(
      &AsyncGRPCServer::TryToRegisterNewSendOne, this, std::placeholders::_1);
  std::function<void(int)> get_register = std::bind(
      &AsyncGRPCServer::TryToRegisterNewGetOne, this, std::placeholders::_1);
  std::function<void(int)> prefetch_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewPrefetchOne, this,
                std::placeholders::_1);

  for (int i = 0; i < kSendReqsBufSize; ++i) {
    TryToRegisterNewSendOne(i);
  }
  for (int i = 0; i < kGetReqsBufSize; ++i) {
    TryToRegisterNewGetOne(i);
  }
  for (int i = 0; i < kPrefetchReqsBufSize; ++i) {
    TryToRegisterNewPrefetchOne(i);
  }

  for (int i = 0; i < FLAGS_rpc_server_handle_send_threads; ++i) {
    t_sends_.emplace_back(new std::thread(
        std::bind(&AsyncGRPCServer::HandleRequest, this, cq_send_.get(),
                  static_cast<int>(GrpcMethod::kSendVariable), send_register)));
  }
  for (int i = 0; i < FLAGS_rpc_server_handle_get_threads; ++i) {
    t_gets_.emplace_back(new std::thread(
        std::bind(&AsyncGRPCServer::HandleRequest, this, cq_get_.get(),
                  static_cast<int>(GrpcMethod::kGetVariable), get_register)));
  }
  for (int i = 0; i < FLAGS_rpc_server_handle_prefetch_threads; ++i) {
    t_prefetchs_.emplace_back(new std::thread(std::bind(
        &AsyncGRPCServer::HandleRequest, this, cq_prefetch_.get(),
        static_cast<int>(GrpcMethod::kPrefetchVariable), prefetch_register)));
  }
  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();
  // wait server
  server_->Wait();
  for (int i = 0; i < FLAGS_rpc_server_handle_send_threads; ++i) {
    t_sends_[i]->join();
  }
  for (int i = 0; i < FLAGS_rpc_server_handle_get_threads; ++i) {
    t_gets_[i]->join();
  }
  for (int i = 0; i < FLAGS_rpc_server_handle_prefetch_threads; ++i) {
    t_prefetchs_[i]->join();
  }
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

void AsyncGRPCServer::TryToRegisterNewSendOne(int i) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }
  RequestSend* send =
      new RequestSend(&service_, cq_send_.get(), rpc_processor_, i);
  send_reqs_[i] = static_cast<RequestBase*>(send);
  VLOG(4) << "Create RequestSend status:" << send->Status();
}

void AsyncGRPCServer::TryToRegisterNewGetOne(int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewGetOne";
    return;
  }
  RequestGet* get = new RequestGet(&service_, cq_get_.get(), &var_get_queue_,
                                   rpc_processor_, req_id);
  get_reqs_[req_id] = static_cast<RequestBase*>(get);
  VLOG(4) << "Create RequestGet status:" << get->Status();
}

void AsyncGRPCServer::TryToRegisterNewPrefetchOne(int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewPrefetchOne";
    return;
  }

  RequestPrefetch* prefetch = new RequestPrefetch(&service_, cq_prefetch_.get(),
                                                  rpc_processor_, req_id);

  prefetch_reqs_[req_id] = static_cast<RequestBase*>(prefetch);

  VLOG(4) << "Create RequestPrefetch status:" << prefetch->Status();
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, int rpc_id,
    std::function<void(int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(3) << "HandleRequest for " << rpc_id << " wait Next";
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << rpc_id << " CompletionQueue shutdown!";
      break;
    }

    int req_id = reinterpret_cast<intptr_t>(tag);

    VLOG(3) << "HandleRequest for " << rpc_id << " get Next";

    RequestBase* base = nullptr;
    {
      if (rpc_id == static_cast<int>(GrpcMethod::kGetVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kGetReqsBufSize);
        base = get_reqs_[req_id];
      } else if (rpc_id == static_cast<int>(GrpcMethod::kSendVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kSendReqsBufSize);
        base = send_reqs_[req_id];
      } else if (rpc_id == static_cast<int>(GrpcMethod::kPrefetchVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kPrefetchReqsBufSize);
        base = prefetch_reqs_[req_id];
      } else {
        PADDLE_ENFORCE(false, "not surpported rpc_id");
      }
    }

    if (rpc_processor_->sync_mode() && !is_shut_down_) {
      auto it = barrier_.find(rpc_id);
      if (it != barrier_.end()) {
        WaitCond(rpc_id);
      }

      VLOG(3) << "HandleRequest for " << rpc_id << " sync_mode";
    }

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      LOG(WARNING) << rpc_id << " recv no regular event:argument name["
                   << base->GetReqName() << "]";
      TryToRegisterNewOne(req_id);
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        base->Process();
        VLOG(4) << rpc_id << " PROCESS status:" << base->Status();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(req_id);
        VLOG(4) << req_id << " FINISH status:" << base->Status();
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

void AsyncGRPCServer::RegisterBarrier(int rpc_id) {
  std::unique_lock<std::mutex> lock(this->barrier_mutex_);
  assert(barrier_.find(rpc_id) == barrier_.end());
  barrier_.insert(rpc_id);
}

void AsyncGRPCServer::WaitCond(int cond) {
  std::unique_lock<std::mutex> lock(this->barrier_mutex_);
  barrier_condition_.wait(lock,
                          [=] { return this->barrier_cond_step_ == cond; });
}

void AsyncGRPCServer::SetCond(int rpc_id) {
  {
    std::lock_guard<std::mutex> lock(this->barrier_mutex_);
    barrier_cond_step_ = rpc_id;
  }
  barrier_condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
