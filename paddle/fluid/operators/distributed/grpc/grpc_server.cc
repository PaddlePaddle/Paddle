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

#include <limits>
#include <memory>
#include <string>

#include "paddle/fluid/operators/distributed/grpc/grpc_request.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_server.h"

using ::grpc::ServerAsyncResponseWriter;

DECLARE_bool(rpc_disable_reuse_port);

namespace paddle {
namespace operators {
namespace distributed {

void AsyncGRPCServer::WaitServerReady() {
  VLOG(4) << "AsyncGRPCServer is waiting server to become ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(4) << "AsyncGRPCServer WaitSeverReady end.";
}

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
// Come from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

void AsyncGRPCServer::StartServer() {
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(bind_address_, ::grpc::InsecureServerCredentials(),
                           &selected_port_);

  builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  if (FLAGS_rpc_disable_reuse_port) {
    builder.SetOption(
        std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
  }
  builder.RegisterService(&service_);

  for (auto t : rpc_call_map_) {
    rpc_cq_[t.first].reset(builder.AddCompletionQueue().release());
  }

  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << bind_address_
            << " selected port: " << selected_port_;

  std::function<void(const RequestType, int)> f =
      std::bind(&AsyncGRPCServer::TryToRegisterNewOne, this,
                std::placeholders::_1, std::placeholders::_2);

  for (auto& t : rpc_call_map_) {
    auto rpc_type = t.first;
    auto& cq = rpc_cq_[rpc_type];
    auto threadnum = rpc_thread_num_[rpc_type];
    auto& reqs = rpc_reqs_[rpc_type];

    reqs.reserve(kRequestBufSize);

    for (int i = 0; i < kRequestBufSize; i++) {
      TryToRegisterNewOne(rpc_type, i);
    }

    for (int i = 0; i < threadnum; i++) {
      rpc_threads_[rpc_type].emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq.get(), rpc_type, f)));
      VLOG(4) << "created thread for req type " << rpc_type;
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  // wait server
  server_->Wait();

  for (auto& t : rpc_threads_) {
    auto& threads = t.second;
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i]->join();
      VLOG(4) << t.first << " threads ends!";
    }
  }
}

void AsyncGRPCServer::ShutdownQueue() {
  for (auto& t : rpc_cq_) {
    t.second->Shutdown();
    VLOG(4) << t.first << " queue shutdown!";
  }
}

void AsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_shut_down_ = true;
  ShutdownQueue();

  VLOG(4) << "server_ shutdown!";
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewOne(const RequestType rpc_type,
                                          int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(4) << "shutdown, skip TryToRegisterNewOne";
    return;
  }

  VLOG(7) << "TryToRegisterNewOne on RPC type: " << rpc_type
          << " REQ ID: " << req_id;

  auto& reqs = rpc_reqs_[rpc_type];
  auto* handler = rpc_call_map_[rpc_type];
  auto& cq = rpc_cq_[rpc_type];

  reqs[req_id] =
      new GRPCRequest(&service_, cq.get(), handler, req_id, rpc_type);
  VLOG(7) << "TryToRegisterNewOne status:" << reqs[req_id]->Status();
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, const RequestType rpc_type,
    std::function<void(RequestType, int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(4) << "HandleRequest " << rpc_type << " wait next";
    if (!cq->Next(&tag, &ok)) {
      LOG(WARNING) << "CompletionQueue " << rpc_type << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));
    VLOG(4) << "HandleRequest " << rpc_type << ", req_id:" << req_id
            << " get next";

    auto& reqs = rpc_reqs_[rpc_type];
    GRPCRequest* base = nullptr;
    {
      PADDLE_ENFORCE(req_id >= 0 && req_id < kRequestBufSize);
      std::unique_lock<std::mutex> lock(cq_mutex_);
      base = reqs[req_id];
    }

    VLOG(4) << base->Status2String(rpc_type);

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      VLOG(4) << "completion queue:" << rpc_type << " recv no regular event"
              << " context:" << base->Status2String(rpc_type);
      TryToRegisterNewOne(rpc_type, req_id);
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        base->Process();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(rpc_type, req_id);
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
