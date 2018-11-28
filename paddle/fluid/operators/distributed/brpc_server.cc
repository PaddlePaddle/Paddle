// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/distributed/brpc_server.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace sendrecv {

typedef std::unordered_map<std::string,
                           paddle::operators::distributed::RequestHandler*>
    HandlerMap;

class BRPCServiceImpl : public SendRecvService {
 public:
  explicit BRPCServiceImpl(const HandlerMap& rpc_call_map)
      : request_send_h_(nullptr),
        request_get_h_(nullptr),
        request_prefetch_h_(nullptr) {
    auto it = rpc_call_map.find(paddle::operators::distributed::kRequestSend);
    if (it != rpc_call_map.end()) {
      request_send_h_ = it->second;
    }

    it = rpc_call_map.find(paddle::operators::distributed::kRequestSend);
    if (it != rpc_call_map.end()) {
      request_get_h_ = it->second;
    }

    it = rpc_call_map.find(paddle::operators::distributed::kRequestPrefetch);
    if (it != rpc_call_map.end()) {
      request_prefetch_h_ = it->second;
    }
  }

  virtual ~BRPCServiceImpl() {}

  void SendVariable(google::protobuf::RpcController* cntl_butil,
                    const VariableMessage* request, VoidMessage* response,
                    google::protobuf::Closure* done) override {
    PADDLE_ENFORCE(request_send_h_ != nullptr,
                   "RequestSend handler should be registed first!");
    brpc::ClosureGuard done_guard(done);

    paddle::framework::Scope* local_scope = request_send_h_->scope();
    paddle::framework::Variable* outvar = nullptr;
    paddle::framework::Variable* invar = nullptr;

    std::string varname = request->varname();

    if (!request_send_h_->sync_mode()) {
      local_scope = &request_send_h_->scope()->NewScope();
      invar = local_scope->Var(varname);
    } else {
      invar = local_scope->FindVar(varname);
    }

    request_send_h_->Handle(varname, local_scope, invar, &outvar);

    if (!request_send_h_->sync_mode()) {
      request_send_h_->scope()->DeleteScope(local_scope);
    }
  }

  void GetVariable(google::protobuf::RpcController* cntl_butil,
                   const VariableMessage* request, VariableMessage* response,
                   google::protobuf::Closure* done) override {
    PADDLE_ENFORCE(request_get_h_ != nullptr,
                   "RequestGet handler should be registed first!");
  }

  void PrefetchVariable(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request,
                        VariableMessage* response,
                        google::protobuf::Closure* done) override {
    PADDLE_ENFORCE(request_prefetch_h_ != nullptr,
                   "kRequestPrefetch handler should be registed first!");
  }

 private:
  paddle::operators::distributed::RequestHandler* request_send_h_;
  paddle::operators::distributed::RequestHandler* request_get_h_;
  paddle::operators::distributed::RequestHandler* request_prefetch_h_;
};
}  // namespace sendrecv

namespace paddle {
namespace operators {
namespace distributed {

void AsyncBRPCServer::StartServer() {
  // Instance of your service.
  sendrecv::BRPCServiceImpl service_impl(rpc_call_map_);

  // Add the service into server. Notice the second parameter, because the
  // service is put on stack, we don't want server to delete it, otherwise
  // use brpc::SERVER_OWNS_SERVICE.
  if (server_.AddService(&service_impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(FATAL) << "Fail to add service";
    return;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = idle_timeout_s_;
  options.max_concurrency = max_concurrency_;
  if (server_.Start(bind_address_.c_str(), &options) != 0) {
    LOG(FATAL) << "Fail to start EchoServer" << bind_address_;
    return;
  }

  butil::EndPoint ep = server_.listen_address();
  selected_port_ = ep.port;

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  server_.Join();
}

void AsyncBRPCServer::ShutDownImpl() { server_.Stop(1000); }

void AsyncBRPCServer::WaitServerReady() {
  VLOG(3) << "AsyncGRPCServer is wait server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(3) << "AsyncGRPCServer WaitSeverReady";
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
