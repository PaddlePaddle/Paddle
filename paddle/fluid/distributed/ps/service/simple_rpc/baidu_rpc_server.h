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
#pragma once
#include <memory>  // std::unique_ptr
#include <string>  // std::string
#include <vector>  // std::vector

#include "paddle/fluid/distributed/ps/service/simple_brpc.pb.h"  // RpcRequest
#include "paddle/fluid/distributed/ps/service/simple_rpc/rpc_server.h"  // RpcServerCallBack

namespace brpc {
class Channel;
class Controller;
class Server;
}  // namespace brpc
namespace google {
namespace protobuf {
class Closure;
class RpcController;
}  // namespace protobuf
}  // namespace google

namespace paddle {
namespace distributed {
namespace simple {
/**
 * @Brief service 处理
 */
class BRpcServiceImpl;
/**
 * @brief baidu rpc
 */
class BaiduRpcServer : public RpcServer {
 public:
  BaiduRpcServer();
  ~BaiduRpcServer();

  void initialize();
  void finalize();
  void send_request(int server_id,
                    void *service_,
                    const size_t n,
                    BinaryArchive *oars,
                    RpcCallback callback);
  void send_response(RpcMessageHead head, const size_t n, BinaryArchive *oars);
  void send_request_ex(int server_id,
                       int consumer_id,
                       void *service_,
                       const size_t n,
                       BinaryArchive *oars,
                       RpcCallback callback);

 public:
  /**
   * @Brief 主要处理baidu-rpc异步响应
   */
  virtual void *add_service(RpcCallback callback, bool simplex = true);

 private:
  void send_message(int send_id,
                    const RpcMessageHead &head,
                    const size_t n,
                    BinaryArchive *oars);

 private:
  std::shared_ptr<BRpcServiceImpl> _service_impl;
  std::shared_ptr<brpc::Server> _server;
  std::vector<std::unique_ptr<SimpleRpcService_Stub>> _senders;
  std::atomic<int> _ref;
};
}  // namespace simple
}  // namespace distributed
}  // namespace paddle
#endif
