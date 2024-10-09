// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/service/simple_rpc/baidu_rpc_server.h"
#include <brpc/channel.h>
#include <brpc/server.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/phi/core/enforce.h"

namespace brpc {
DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);
}  // namespace brpc

namespace paddle {
namespace distributed {
namespace simple {

static const int MIN_SERVER_LISTEN_PORT = 20000;
static const int MAX_SERVER_LISTEN_PORT = 65535;
static const int64_t MAX_RPC_BODY_SIZE = 10 * 1024 * 1024 * 1024L;

class BRpcReqService : public RpcService {
 public:
  BRpcReqService(RpcCallback callback, bool simplex)
      : RpcService(callback), _simplex(simplex) {}
  void set_handler(brpc::Controller *cntl,
                   google::protobuf::Closure *done,
                   SimpleRpcResponse *response) {
    _cntl = cntl;
    _response = response;
    _done = done;
  }
  bool is_simplex(void) { return _simplex; }
  butil::IOBuf &response_attachment(void) {
    return _cntl->response_attachment();
  }
  void done(int64_t size) {
    _response->set_archive_size(size);
    _done->Run();
  }

 private:
  bool _simplex = true;
  brpc::Controller *_cntl = nullptr;
  SimpleRpcResponse *_response = nullptr;
  google::protobuf::Closure *_done = nullptr;
};

/**
 * @Brief service 处理
 */
class BRpcServiceImpl : public SimpleRpcService {
 public:
  explicit BRpcServiceImpl(int rank_id) : _rank_id(rank_id) {}
  virtual ~BRpcServiceImpl() {}
  virtual void handle_request(google::protobuf::RpcController *cntl_base,
                              const SimpleRpcRequest *baidu_rpc_request,
                              SimpleRpcResponse *baidu_rpc_response,
                              google::protobuf::Closure *done) {
    brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
    uint64_t size = baidu_rpc_request->archive_size();
    butil::IOBuf &attach = cntl->request_attachment();
    BinaryArchive iar;
    iar.Reserve(size);
    uint64_t attach_size = attach.cutn(iar.Buffer(), size);
    PADDLE_ENFORCE_EQ(
        (attach_size == size),
        true,
        common::errors::PreconditionNotMet("Request size is wrong."));
    iar.AdvanceFinish(size);

    RpcMessageHead head;
    iar.ReadBack(&head, sizeof(RpcMessageHead));
    if (head.message_type == RpcMessageHead::REQUEST) {
      PADDLE_ENFORCE_EQ(
          (head.server_id == _rank_id),
          true,
          common::errors::PreconditionNotMet(
              "Server id %d not equal rank id %d.", head.server_id, _rank_id));
      BRpcReqService *service =
          reinterpret_cast<BRpcReqService *>(head.service);
      service->set_handler(cntl, done, baidu_rpc_response);
      service->callback()(head, iar);
      // 如果只单向由client->server通信，就直接将应答为0
      if (service->is_simplex()) {
        baidu_rpc_response->set_archive_size(0);
        done->Run();
      }
      return;
    }
    if (head.message_type == RpcMessageHead::RESPONSE) {
      PADDLE_ENFORCE_EQ(
          (head.client_id == _rank_id),
          true,
          common::errors::PreconditionNotMet(
              "Client id %d not equal rank id %d.", head.client_id, _rank_id));
      head.request->callback()(head, iar);
      delete head.request;
      PADDLE_ENFORCE_NE(
          head.service,
          nullptr,
          common::errors::PreconditionNotMet("Service should not be nullptr."));
      head.service->decrease_request();
    } else {
      PADDLE_THROW(common::errors::InvalidArgument("Unknown message type"));
    }
    baidu_rpc_response->set_archive_size(0);
    done->Run();
  }

 private:
  int _rank_id = 0;
};

BaiduRpcServer::BaiduRpcServer() : RpcServer(), _server(nullptr) {
  /** 因为RPC这里主要用于pull sparse和data shuffle数据量比较大，
   * 单个pass的key超过几亿，发送数据单包大小是存在超过1G以上的可能，
   * 需要设baidu rpc最大可发送包的大小
   */
  if (brpc::FLAGS_max_body_size < MAX_RPC_BODY_SIZE) {
    brpc::FLAGS_max_body_size = MAX_RPC_BODY_SIZE;
  }
  if (brpc::FLAGS_socket_max_unwritten_bytes < MAX_RPC_BODY_SIZE) {
    brpc::FLAGS_socket_max_unwritten_bytes = MAX_RPC_BODY_SIZE;
  }
  _server.reset(new brpc::Server);
  _ref = 0;
}

BaiduRpcServer::~BaiduRpcServer() {}

/**
 * @brief 初始化服务
 */
void BaiduRpcServer::initialize() {
  if (++_ref > 1) {
    LOG(WARNING) << "already initialize rpc server";
    return;
  }

  PADDLE_ENFORCE_NE(
      _gloo,
      nullptr,
      common::errors::PreconditionNotMet("Gloo not allow nullptr."));
  _gloo->Barrier();
  _server->set_version(google::VersionString());
  brpc::ServerOptions option;
  option.idle_timeout_sec = _connection_idle_timeout_sec;
  option.auth = nullptr;
  option.num_threads = _thread_num;
  _service_impl = std::make_shared<BRpcServiceImpl>(_gloo->Rank());
  int ret =
      _server->AddService(_service_impl.get(), brpc::SERVER_DOESNT_OWN_SERVICE);
  PADDLE_ENFORCE_EQ(
      (ret == 0),
      true,
      common::errors::PreconditionNotMet("Failed to add BRpcServiceImpl."));
  brpc::PortRange range(MIN_SERVER_LISTEN_PORT, MAX_SERVER_LISTEN_PORT);
  auto server_ip = butil::ip2str(butil::int2ip(_ips[_gloo->Rank()]));
  ret = _server->Start(server_ip.c_str(), range, &option);
  PADDLE_ENFORCE_EQ(
      (ret == 0),
      true,
      common::errors::PreconditionNotMet("Fail to start BaiduRpcServer."));
  butil::EndPoint ep = _server->listen_address();
  std::vector<int> ports = _gloo->AllGather(ep.port);
  auto new_channel = [this, &ports](int i) {
    brpc::Channel *channel_ptr = new brpc::Channel();
    brpc::ChannelOptions option;
    option.connection_type = _connection_type;
    option.auth = nullptr;
    option.timeout_ms = _client_timeout_ms;
    option.connect_timeout_ms = _connect_timeout_ms;
    option.max_retry = _max_retry;

    butil::EndPoint cep;
    cep.ip = butil::int2ip(_ips[i]);
    cep.port = ports[i];
    if (channel_ptr->Init(cep, &option) != 0) {
      PADDLE_THROW(common::errors::Fatal("Failed to initialize channel"));
    }
    LOG(INFO) << "connected to " << butil::endpoint2str(cep).c_str();
    return channel_ptr;
  };
  for (int i = 0; i < _gloo->Size(); i++) {
    _senders.emplace_back(new SimpleRpcService_Stub(
        new_channel(i), google::protobuf::Service::STUB_OWNS_CHANNEL));
  }
  _gloo->Barrier();
  LOG(WARNING) << "initialize rpc server : " << butil::endpoint2str(ep).c_str();
}
/**
 * @brief 停止服务
 */
void BaiduRpcServer::finalize() {
  if (--_ref > 0) {
    LOG(WARNING) << "finalize running by other";
    return;
  }
  _gloo->Barrier();
  _server->Stop(60000);
  _server->Join();
  _gloo->Barrier();
  LOG(INFO) << "finalize rpc server";
}

/**
 * @brief 客户端发送回的应答
 */
static void handle_baidu_rpc_response(brpc::Controller *cntl,
                                      SimpleRpcResponse *baidu_rpc_response) {
  size_t size = baidu_rpc_response->archive_size();
  if (size > 0) {
    BinaryArchive iar;
    iar.Reserve(size);
    size_t attach_size = cntl->response_attachment().cutn(iar.Buffer(), size);
    PADDLE_ENFORCE_EQ(
        (attach_size == size),
        true,
        common::errors::PreconditionNotMet("Request size is wrong."));
    iar.AdvanceFinish(size);

    RpcMessageHead head;
    iar.ReadBack(&head, sizeof(RpcMessageHead));
    if (head.message_type == RpcMessageHead::RESPONSE) {
      head.request->callback()(head, iar);
      delete head.request;
      PADDLE_ENFORCE_NE(
          head.service,
          nullptr,
          common::errors::PreconditionNotMet("Service should not be nullptr."));
      head.service->decrease_request();
    } else {
      PADDLE_THROW(common::errors::InvalidArgument("Unknown message type"));
    }
  }
  delete baidu_rpc_response;
  delete cntl;
}

void BaiduRpcServer::send_request(int server_id,
                                  void *service_,
                                  const size_t n,
                                  BinaryArchive *oars,
                                  RpcCallback callback) {
  send_request_ex(server_id, 0, service_, n, oars, callback);
}
void BaiduRpcServer::send_request_ex(int server_id,
                                     int consumer_id,
                                     void *service_,
                                     const size_t n,
                                     BinaryArchive *oars,
                                     RpcCallback callback) {
  RpcService *service = reinterpret_cast<RpcService *>(service_);
  service->increase_request();

  RpcMessageHead head;
  head.service = service->remote_pointer(server_id);
  head.request = new RpcRequest(callback);
  head.client_id = _gloo->Rank();
  head.server_id = server_id;
  head.message_type = RpcMessageHead::REQUEST;
  head.consumer_id = consumer_id;

  send_message(server_id, head, n, oars);
}
void BaiduRpcServer::send_response(RpcMessageHead head,
                                   const size_t n,
                                   BinaryArchive *oars) {
  PADDLE_ENFORCE_EQ(
      (head.server_id == _gloo->Rank()),
      true,
      common::errors::PreconditionNotMet("Server_id not equal rank id."));
  PADDLE_ENFORCE_EQ(
      (head.client_id >= 0 && head.client_id < _gloo->Size()),
      true,
      common::errors::PreconditionNotMet("The client id is error."));
  BRpcReqService *service = reinterpret_cast<BRpcReqService *>(head.service);
  head.service = head.service->remote_pointer(head.client_id);
  head.message_type = RpcMessageHead::RESPONSE;

  // 如果只单向由client->server通信，就统一走数据发送接口
  if (service->is_simplex()) {
    send_message(head.client_id, head, n, oars);
  } else {
    // 这种情况只适合在callback里面直接调用send_response方式
    auto &ar = service->response_attachment();
    for (size_t i = 0; i < n; i++) {
      auto &oar = oars[i];
      if (oar.Length() == 0) {
        continue;
      }
      ar.append(oar.Buffer(), oar.Length());
    }
    ar.append(&head, sizeof(head));
    service->done(ar.length());
  }
}

void BaiduRpcServer::send_message(int send_id,
                                  const RpcMessageHead &head,
                                  const size_t n,
                                  BinaryArchive *oars) {
  brpc::Controller *cntl = new brpc::Controller();
  cntl->ignore_eovercrowded();

  auto &ar = cntl->request_attachment();
  for (size_t i = 0; i < n; i++) {
    auto &oar = oars[i];
    if (oar.Length() == 0) {
      continue;
    }
    ar.append(oar.Buffer(), oar.Length());
  }
  ar.append(&head, sizeof(head));

  SimpleRpcRequest baidu_rpc_request;
  baidu_rpc_request.set_archive_size(ar.length());
  cntl->set_log_id(_gloo->Rank());

  SimpleRpcResponse *baidu_rpc_response = new SimpleRpcResponse();
  google::protobuf::Closure *done = google::protobuf::NewCallback(
      &handle_baidu_rpc_response, cntl, baidu_rpc_response);
  _senders[send_id]->handle_request(
      cntl, &baidu_rpc_request, baidu_rpc_response, done);
}
/**
 * @Brief 主要处理baidu-rpc异步响应
 */
void *BaiduRpcServer::add_service(RpcCallback callback, bool simplex) {
  return new BRpcReqService(std::move(callback), simplex);
}
}  // namespace simple
}  // namespace distributed
}  // namespace paddle
#endif
