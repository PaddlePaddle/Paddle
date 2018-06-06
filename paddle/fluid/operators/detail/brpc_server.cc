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

#include "paddle/fluid/operators/detail/brpc_server.h"
#include "glog/logging.h"

namespace sendrecv {
class BRPCServiceImpl : public SendRecvService {
 public:
  BRPCServiceImpl() {}
  virtual ~BRPCServiceImpl() {}
  void SendVariable(google::protobuf::RpcController* cntl_butil,
                    const VariableMessage* request, VoidMessage* response,
                    google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    // Echo request and its attachment
    /*
    response->set_message(request->message());
    if (FLAGS_echo_attachment) {
        cntl->response_attachment().append(cntl->request_attachment());
    }
    */
  }
};
}  // namespace sendrecv

namespace paddle {
namespace operators {
namespace detail {

void AsyncBRPCServer::StartServer() {
  brpc::ServerOptions options;
  options.idle_timeout_sec = idle_timeout_s_;
  options.max_concurrency = max_concurrency_;
  if (server_.Start(bind_address_.c_str(), &options) != 0) {
    LOG(FATAL) << "Fail to start EchoServer" << bind_address_;
    return -1;
  }
}

void AsyncBRPCServer::ShutDown() { server_.Stop(); }

void AsyncBRPCServer::WaitServerReady() {
  while (1) {
    brpc::Server::Status status = server_.Status();
    if (status != brpc::Server::Status::RUNNING &&
        status != brpc::Server::Status::READY) {
      LOG(INFO) << "wait start server on " << bind_address_;
      sleep(5);
    }
  }
}

};  // namespace detail
};  // namespace operators
};  // namespace paddle
