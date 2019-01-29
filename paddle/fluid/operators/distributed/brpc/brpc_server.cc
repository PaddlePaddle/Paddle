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

#include "paddle/fluid/operators/distributed/brpc/brpc_server.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_variable_response.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace sendrecv {

namespace distributed = paddle::operators::distributed;

typedef std::unordered_map<distributed::RequestType,
                           distributed::RequestHandler*,
                           distributed::EnumClassHash>
    HandlerMap;

class BRPCServiceImpl : public SendRecvService {
 public:
  explicit BRPCServiceImpl(HandlerMap* rpc_call_map,
                           distributed::RPCServer* rpc_server)
      : rpc_call_map_(rpc_call_map), rpc_server_(rpc_server) {
    VLOG(3) << "BRPCServiceImpl size: " << rpc_call_map->size();
    auto it = rpc_call_map->find(distributed::RequestType::SEND);
    if (it != rpc_call_map->end()) {
      send_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::RequestType::SEND)));
    }

    it = rpc_call_map->find(distributed::RequestType::RECV);
    if (it != rpc_call_map->end()) {
      get_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::RequestType::RECV)));
    }

    it = rpc_call_map->find(distributed::RequestType::RECV_NO_BARRIER);
    if (it != rpc_call_map->end()) {
      getnobarrier_threads_.reset(
          new paddle::framework::ThreadPool(rpc_server_->GetThreadNum(
              distributed::RequestType::RECV_NO_BARRIER)));
    }

    it = rpc_call_map->find(distributed::RequestType::PREFETCH);
    if (it != rpc_call_map->end()) {
      prefetch_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::RequestType::PREFETCH)));
    }

    it = rpc_call_map->find(distributed::RequestType::CHECKPOINT);
    if (it != rpc_call_map->end()) {
      checkpoint_notify_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::RequestType::CHECKPOINT)));
    }
  }

  void HandleRequest(distributed::RequestType req_type,
                     google::protobuf::RpcController* cntl_butil,
                     const VariableMessage* request, VariableMessage* response,
                     google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);
    auto handler = rpc_call_map_->at(req_type);
    distributed::RPCRequest req;
    auto get_var_callback =
        std::bind(&distributed::RequestHandler::GetOrCreateRequestVar, handler,
                  std::placeholders::_1, &req);

    distributed::BRPCVariableResponse resp(get_var_callback,
                                           handler->dev_ctx());
    PADDLE_ENFORCE(resp.Parse(cntl->request_attachment(), *request) == 0,
                   "parse iobuf to tensor error!");
    req.Prepare(request->varname(), resp.GetVar(), "", "",
                request->trainer_id(), req_type);
    handler->Handle(&req);
    bool var_is_stable =
        req_type == distributed::RequestType::PREFETCH ? true : false;
    if (req.out_var_) {
      // respones do not need trainer id and table_name.
      distributed::SerializeToIOBuf(
          req.out_var_name_, req.out_var_, *handler->dev_ctx(), response,
          &cntl->response_attachment(), req.out_var_name_, var_is_stable);
    }
  }

  virtual ~BRPCServiceImpl() {}
  void SendVariable(google::protobuf::RpcController* cntl_butil,
                    const VariableMessage* request, VariableMessage* response,
                    google::protobuf::Closure* done) override {
    send_threads_->Run([=] {
      HandleRequest(distributed::RequestType::SEND, cntl_butil, request,
                    response, done);
    });
  }

  void GetVariable(google::protobuf::RpcController* cntl_butil,
                   const VariableMessage* request, VariableMessage* response,
                   google::protobuf::Closure* done) override {
    get_threads_->Run([=] {
      HandleRequest(distributed::RequestType::RECV, cntl_butil, request,
                    response, done);
    });
  }

  void GetVariableNoBarrier(google::protobuf::RpcController* cntl_butil,
                            const VariableMessage* request,
                            VariableMessage* response,
                            google::protobuf::Closure* done) override {
    getnobarrier_threads_->Run([=] {
      HandleRequest(distributed::RequestType::RECV_NO_BARRIER, cntl_butil,
                    request, response, done);
    });
  }

  void PrefetchVariable(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request,
                        VariableMessage* response,
                        google::protobuf::Closure* done) override {
    prefetch_threads_->Run([=] {
      HandleRequest(distributed::RequestType::PREFETCH, cntl_butil, request,
                    response, done);
    });
  }

  void CheckpointNotify(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request,
                        VariableMessage* response,
                        google::protobuf::Closure* done) override {
    checkpoint_notify_threads_->Run([=] {
      HandleRequest(distributed::RequestType::CHECKPOINT, cntl_butil, request,
                    response, done);
    });
  }

  void GetMonomerVariable(google::protobuf::RpcController* cntl_butil,
                          const VariableMessage* request,
                          VariableMessage* response,
                          google::protobuf::Closure* done) override {
    HandleRequest(distributed::RequestType::GET_MONOMER, cntl_butil, request,
                  response, done);
  }

  void GetMonomerBarrier(google::protobuf::RpcController* cntl_butil,
                         const VariableMessage* request,
                         VariableMessage* response,
                         google::protobuf::Closure* done) override {
    HandleRequest(distributed::RequestType::GET_MONOMER_BARRIER, cntl_butil,
                  request, response, done);
  }

 private:
  HandlerMap* rpc_call_map_{nullptr};
  distributed::RPCServer* rpc_server_{nullptr};

  // FIXME(gongwb): brpc should support process one rpc use one threadpool.
  std::unique_ptr<paddle::framework::ThreadPool> send_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> get_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> getnobarrier_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> prefetch_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> checkpoint_notify_threads_;
};
}  // namespace sendrecv

namespace paddle {
namespace operators {
namespace distributed {

void AsyncBRPCServer::StartServer() {
  // Instance of your service.
  sendrecv::BRPCServiceImpl service_impl(&rpc_call_map_, this);

  // Add the service into server. Notice the second parameter, because the
  // service is put on stack, we don't want server to delete it, otherwise
  // use brpc::SERVER_OWNS_SERVICE.
  if (server_.AddService(&service_impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(FATAL) << "Fail to add service";
    return;
  }

  brpc::ServerOptions options;
#ifdef PADDLE_WITH_BRPC_RDMA
  options.use_rdma = true;
#endif
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
