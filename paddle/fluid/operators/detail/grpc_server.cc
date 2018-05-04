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

#include "paddle/fluid/operators/detail/grpc_server.h"

#include <limits>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include "paddle/fluid/operators/detail/mpi_utils.h"

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
                      const platform::DeviceContext* dev_ctx,
                      framework::BlockingQueue<MessageWithName>* queue)
      : RequestBase(service, cq, dev_ctx),
        responder_(&ctx_),
        scope_(scope),
        queue_(queue) {
    int method_id = static_cast<int>(detail::GrpcMethod::kGetVariable);
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
    if (var_name != FETCH_BARRIER_MESSAGE) {
      SerializeToByteBuffer(var_name, var, *dev_ctx_, &reply);
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
  framework::Scope* scope_;
  framework::BlockingQueue<MessageWithName>* queue_;
};

class RequestMPISend final : public RequestBase {
 public:
  explicit RequestMPISend(GrpcService::AsyncService* service,
                          ::grpc::ServerCompletionQueue* cq,
                          framework::Scope* scope, ReceivedQueue* queue,
                          const platform::DeviceContext* dev_ctx, int mpi_dst)
      : RequestBase(service, cq, dev_ctx), queue_(queue), responder_(&ctx_) {
    this->mpi_dst = mpi_dst;
    int method_id = static_cast<int>(detail::GrpcMethod::kMPISendVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_, &responder_, cq_,
                                cq_, this);
  }

  virtual ~RequestMPISend() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    framework::Async([&dev_ctx, &scope, &request_, &queue]() {
      vector<string> indexs;
      boost::split(indexs, request_.var_slice(), boost::is_any_of(":"));

      std::vector<Slice> slices;
      std::vector<std::future<void>> fs;

      for (int i = 0; i < indexs.length; i++) {
        int slice_len = indexs[i];
        Slice slice;
        slices.push_back(slice);
        fs.push_back(framework::Async(
            [&var_req, &grpc_res, &request_, &slice_len, &slice]() {
              MPIIrecvProcess(&meta_buffer, slice_len, request_.src(),
                              request_.tag());
            }));
      }
      for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();

      grpc::ByteBuffer meta_buffer(&slices[0], indexs.length);
      framework::Variable* outvar = nullptr;
      DeserializeFromByteBuffer(meta_buffer, dev_ctx_, &scope, &outvar);
      queue.push(request_.Varname(), outvar);
    });

    // Async MpiIrecv to receive data
    // need to add reply content
    sendrecv::VariableMeta reply;

    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  sendrecv::VariableMeta request_;
  ReceivedQueue* queue_;
  ServerAsyncResponseWriter<sendrecv::VariableMeta> responder_;

  // For MPI
  int mpi_dst;
};

class RequestMPIGet final : public RequestBase {
 public:
  explicit RequestMPIGet(GrpcService::AsyncService* service,
                         ::grpc::ServerCompletionQueue* cq,
                         framework::Scope* scope, ReceivedQueue* queue,
                         const platform::DeviceContext* dev_ctx)
      : RequestBase(service, cq, dev_ctx), queue_(queue), responder_(&ctx_) {
    request_.reset(new VariableResponse(scope, dev_ctx_));
    int method_id = static_cast<int>(detail::GrpcMethod::kMPIGetVariable);
    service_->RequestAsyncUnary(method_id, &ctx_, request_.get(), &responder_,
                                cq_, cq_, this);
  }

  virtual ~RequestMPIGet() {}

  virtual std::string GetReqName() { return request_->Varname(); }

  virtual void Process() {
    // proc request.
    std::string var_name = request_.varname();
    auto* var = scope_->FindVar(var_name);

    ::grpc::ByteBuffer mpi_reply;
    SerializeToByteBuffer(var_name, var, *dev_ctx_, &mpi_reply);

    framework::Async([&dev_ctx, &scope, &request_, &queue, &mpi_reply]() {
      MPIIsendProcess(&mpi_reply, request_.length(), request_.src(),
                      request_.tag());
      queue.push(request_.Varname());
    });

    sendrecv::VariableMeta reply;
    responder_.Finish(reply, ::grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  std::shared_ptr<sendrecv::VariableMeta> request_;
  ReceivedQueue* queue_;
  ServerAsyncResponseWriter<sendrecv::VariableMeta> responder_;
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
  int blkid_;
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

  // For MPI
  std::function<void()> mpi_send_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewMPISendOne, this);
  std::function<void()> mpi_get_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewMPIGetOne, this);

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

  // For MPI
  t_mpi_send_.reset(new std::thread(
      std::bind(&AsyncGRPCServer::HandleRequest, this, cq_mpi_send_.get(),
                "cq_mpi_send", mpi_send_register)));
  t_mpi_get_.reset(new std::thread(std::bind(&AsyncGRPCServer::HandleRequest,
                                             this, cq_mpi_get_.get(),
                                             "cq_mpi_get", mpi_get_register)));

  // wait server
  server_->Wait();
  t_send_->join();
  t_get_->join();
  t_prefetch_->join();

  // For MPI
  t_mpi_send_->join();
  t_mpi_get_->join();
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
  RequestSend* send = new RequestSend(&service_, cq_send_.get(), scope_,
                                      &var_recv_queue_, dev_ctx_);
  VLOG(4) << "Create RequestSend status:" << send->Status();
}

void AsyncGRPCServer::TryToRegisterNewGetOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewGetOne";
    return;
  }
  RequestGet* get = new RequestGet(&service_, cq_get_.get(), scope_, dev_ctx_,
                                   &var_get_queue_);
  VLOG(4) << "Create RequestGet status:" << get->Status();
}

void AsyncGRPCServer::TryToRegisterNewPrefetchOne() {
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

void AsyncGRPCServer::TryToRegisterNewMPISendOne() {
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewMPISendOne";
    return;
  }
}

void AsyncGRPCServer::TryToRegisterNewMPIGetOne() {}

// FIXME(typhoonzero): change cq_name to enum.
void AsyncGRPCServer::HandleRequest(::grpc::ServerCompletionQueue* cq,
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
    // FIXME(typhoonzero): de-couple the barriers with recv_op
    if (!is_shut_down_ && cq_name == "cq_get") WaitCond(1);
    if (!is_shut_down_ && cq_name == "cq_send") WaitCond(0);

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
