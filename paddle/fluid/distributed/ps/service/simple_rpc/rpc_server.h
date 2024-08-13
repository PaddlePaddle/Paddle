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
#include <glog/logging.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "paddle/fluid/framework/archive.h"

namespace paddle {
namespace framework {
class GlooWrapper;
}
namespace distributed {
namespace simple {
using BinaryArchive = paddle::framework::BinaryArchive;

class RpcService;
class RpcRequest;

struct RpcMessageHead {
  RpcService *service;
  RpcRequest *request;
  int client_id;
  int server_id;
  enum { REQUEST, RESPONSE } message_type;
  int consumer_id;
};

typedef std::function<void(const RpcMessageHead &, BinaryArchive &)>
    RpcCallback;  // NOLINT

class RpcService {
 public:
  RpcService() {}
  explicit RpcService(RpcCallback callback);
  ~RpcService();
  RpcService *remote_pointer(int rank) { return _remote_ptrs[rank]; }
  RpcCallback &callback() { return _callback; }
  void increase_request() { ++_request_counter; }
  void decrease_request() { --_request_counter; }

 protected:
  std::vector<RpcService *> _remote_ptrs;
  RpcCallback _callback;
  std::atomic<int> _request_counter{0};
};

class RpcRequest {
 public:
  explicit RpcRequest(RpcCallback callback) : _callback(std::move(callback)) {}
  RpcCallback &callback() { return _callback; }

 protected:
  RpcCallback _callback;
};

class RpcServer {
 public:
  RpcServer();
  virtual ~RpcServer();

 public:
  void set_connection_num(int n);
  void set_thread_num(int n);
  void set_connection_idle_timeout_sec(int timeout_sec) {
    _connection_idle_timeout_sec = timeout_sec;
  }
  void set_max_retry(int retry_cnt) { _max_retry = retry_cnt; }
  void set_connect_timeout_ms(int timeout_ms) {
    _connect_timeout_ms = timeout_ms;
  }
  void set_connection_type(const std::string &conn_type) {
    _connection_type = conn_type;
  }
  void set_client_timeout_ms(int timeout_ms) {
    _client_timeout_ms = timeout_ms;
  }

 public:
  virtual void initialize() = 0;
  virtual void finalize() = 0;
  virtual void send_request(int server_id,
                            void *service_,
                            const size_t n,
                            BinaryArchive *oars,
                            RpcCallback callback) = 0;
  virtual void send_response(RpcMessageHead head,
                             const size_t n,
                             BinaryArchive *oars) = 0;
  virtual void send_request_ex(int server_id,
                               int consumer_id,
                               void *service_,
                               const size_t n,
                               BinaryArchive *oars,
                               RpcCallback callback) = 0;

 public:
  virtual void *add_service(RpcCallback callback, bool simplex = true);
  virtual void remove_service(void *service);

 public:
  void send_request_wrapper(int server_id,
                            void *service,
                            BinaryArchive &oar,  // NOLINT
                            RpcCallback callback) {
    send_request(server_id, service, 1, &oar, std::move(callback));
  }
  void send_request_consumer(int server_id,
                             int consumer_id,
                             void *service,
                             BinaryArchive &oar,  // NOLINT
                             RpcCallback callback) {
    send_request_ex(
        server_id, consumer_id, service, 1, &oar, std::move(callback));
  }
  void send_response(RpcMessageHead head, BinaryArchive &oar) {  // NOLINT
    send_response(head, 1, &oar);
  }

 protected:
  int _conn_num = 1;
  int _thread_num = 10;
  std::vector<uint32_t> _ips;
  paddle::framework::GlooWrapper *_gloo = NULL;
  // configure for rpc
  int _connection_idle_timeout_sec = 3600;
  int _max_retry = 1000;
  int _connect_timeout_ms = -1;
  std::string _connection_type = "pooled";
  int _client_timeout_ms = -1;
};

extern RpcServer &global_rpc_server();
}  // namespace simple
}  // namespace distributed
}  // namespace paddle
#endif
