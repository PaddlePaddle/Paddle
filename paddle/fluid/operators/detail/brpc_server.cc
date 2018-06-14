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
#include "paddle/fluid/operators/detail/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/brpc_variable_response.h"
#include "paddle/fluid/operators/detail/request_handler.h"

namespace sendrecv {

typedef std::unordered_map<std::string,
                           paddle::operators::detail::RequestHandler*>
    HandlerMap;

namespace detail = paddle::operators::detail;

class BRPCServiceImpl : public SendRecvService {
 public:
  explicit BRPCServiceImpl(const HandlerMap& rpc_call_map)
      : request_send_h_(nullptr),
        request_get_h_(nullptr),
        request_prefetch_h_(nullptr) {
    auto it = rpc_call_map.find(paddle::operators::detail::kRequestSend);
    if (it != rpc_call_map.end()) {
      request_send_h_ = it->second;
    }

    it = rpc_call_map.find(paddle::operators::detail::kRequestSend);
    if (it != rpc_call_map.end()) {
      request_get_h_ = it->second;
    }

    it = rpc_call_map.find(detail::kRequestPrefetch);
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
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    VLOG(3) << "RequestSend var_name:" << varname;

    detail::BRPCVariableResponse resp(request_send_h_->scope(),
                                      request_send_h_->dev_ctx());
    PADDLE_ENFORCE(resp.Parse(cntl->request_attachment(), *request) == 0,
                   "parse iobuf to tensor error!");

    auto scope = resp.GetMutableLocalScope();
    auto invar = resp.GetVar();
    paddle::framework::Variable* outvar = nullptr;

    request_send_h_->Handle(varname, scope, invar, &outvar);
  }

  void GetVariable(google::protobuf::RpcController* cntl_butil,
                   const VariableMessage* request, VariableMessage* response,
                   google::protobuf::Closure* done) override {
    PADDLE_ENFORCE(request_get_h_ != nullptr,
                   "RequestGet handler should be registed first!");

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    VLOG(3) << "RequestGet " << varname;

    detail::BRPCVariableResponse resp(request_get_h_->scope(),
                                      request_get_h_->dev_ctx());
    PADDLE_ENFORCE(resp.Parse(cntl->request_attachment(), *request) == 0,
                   "parse iobuf to tensor error!");

    auto scope = request_get_h_->scope();
    auto invar = scope->FindVar(varname);
    paddle::framework::Variable* outvar = nullptr;

    request_get_h_->Handle(varname, scope, invar, &outvar);

    if (outvar) {
      detail::SerializeToIOBuf(varname, outvar, *request_get_h_->dev_ctx(),
                               response, &cntl->response_attachment());
    }
  }

  void PrefetchVariable(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request,
                        VariableMessage* response,
                        google::protobuf::Closure* done) override {
    PADDLE_ENFORCE(request_prefetch_h_ != nullptr,
                   "kRequestPrefetch handler should be registed first!");

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    // prefetch process...
    std::string in_var_name = request->varname();
    std::string out_var_name = request->out_varname();
    VLOG(3) << "RequestPrefetch, in_var_name: " << in_var_name
            << " out_var_name: " << out_var_name;

    detail::BRPCVariableResponse resp(request_prefetch_h_->scope(),
                                      request_prefetch_h_->dev_ctx());
    PADDLE_ENFORCE(resp.Parse(cntl->request_attachment(), *request) == 0,
                   "parse iobuf to tensor error!");

    auto scope = resp.GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    paddle::framework::Variable* outvar = scope->FindVar(out_var_name);

    request_prefetch_h_->Handle(in_var_name, scope, invar, &outvar,
                                out_var_name);

    detail::SerializeToIOBuf(out_var_name, outvar,
                             *request_prefetch_h_->dev_ctx(), response,
                             &cntl->response_attachment());
  }

 private:
  detail::RequestHandler* request_send_h_;
  detail::RequestHandler* request_get_h_;
  detail::RequestHandler* request_prefetch_h_;
};
}  // namespace sendrecv

namespace paddle {
namespace operators {
namespace detail {

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

};  // namespace detail
};  // namespace operators
};  // namespace paddle
