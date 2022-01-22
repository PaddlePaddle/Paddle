// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "butil/endpoint.h"
#include "google/protobuf/service.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace google {
namespace protobuf {
class RpcController;
}  // namespace protobuf
}  // namespace google
namespace paddle {
namespace distributed {
class PSEnvironment;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace framework {
class Executor;
class ProgramDesc;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

class Table;

using paddle::distributed::PsRequestMessage;
using paddle::distributed::PsResponseMessage;

class PSServer {
 public:
  PSServer() {}
  virtual ~PSServer() {}
  PSServer(PSServer &&) = delete;
  PSServer(const PSServer &) = delete;

  virtual int32_t configure(
      const PSParameter &config, PSEnvironment &env, size_t server_rank,
      const std::vector<framework::ProgramDesc> &server_sub_program = {});

  // return server_ip
  virtual std::string ip() { return butil::my_ip_cstr(); }
  // return server_port
  virtual int32_t port() = 0;

  virtual uint64_t start(const std::string &ip, uint32_t port) = 0;
  virtual int32_t stop() = 0;

  inline size_t rank() const { return _rank; }

  inline PSEnvironment *environment() { return _environment; }

  inline const ServerParameter *config() const { return &_config; }
  inline Table *table(size_t table_id) {
    auto itr = _table_map.find(table_id);
    if (itr != _table_map.end()) {
      return itr->second.get();
    }
    return NULL;
  }

  inline std::unordered_map<uint32_t, std::shared_ptr<Table>> *table() {
    return &_table_map;
  }

  typedef std::function<int32_t(int, int, const std::string &)> MsgHandlerFunc;
  virtual int registe_pserver2pserver_msg_handler(int msg_type,
                                                  MsgHandlerFunc handler) {
    _msg_handler_map[msg_type] = handler;
    return 0;
  }

  paddle::framework::Channel<std::pair<uint64_t, std::string>> _shuffled_ins;

 protected:
  virtual int32_t initialize() = 0;

 protected:
  size_t _rank;
  ServerParameter _config;
  PSEnvironment *_environment;
  std::unordered_map<uint32_t, std::shared_ptr<Table>> _table_map;
  std::unordered_map<int32_t, MsgHandlerFunc> _msg_handler_map;

 protected:
  std::shared_ptr<framework::Scope> scope_;
  platform::Place place_ = platform::CPUPlace();
};

REGISTER_PSCORE_REGISTERER(PSServer);

typedef std::function<void(void *)> PServerCallBack;

class PServerClosure : public google::protobuf::Closure {
 public:
  PServerClosure(PServerCallBack callback) : _callback(callback) {}
  virtual ~PServerClosure() {}
  virtual void set_promise_value(int value) {
    for (auto &promise : _promises) {
      promise->set_value(value);
    }
  }
  void add_promise(std::shared_ptr<std::promise<int32_t>> &promise) {
    _promises.push_back(promise);
  }

 protected:
  PServerCallBack _callback;
  std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
};

class PsBaseService : public PsService {
 public:
  PsBaseService() : _rank(0), _server(NULL), _config(NULL) {}
  virtual ~PsBaseService() {}
  virtual size_t get_rank() { return _rank; }
  virtual int32_t configure(PSServer *server) {
    _server = server;
    _rank = _server->rank();
    _config = _server->config();
    return 0;
  }
  virtual void service(::google::protobuf::RpcController *controller,
                       const PsRequestMessage *request,
                       PsResponseMessage *response,
                       ::google::protobuf::Closure *done) override = 0;

  virtual void set_response_code(PsResponseMessage &response, int err_code,
                                 const char *err_msg) {
    response.set_err_msg(err_msg);
    response.set_err_code(err_code);
    LOG(WARNING) << "Resonse err_code:" << err_code << " msg:" << err_msg;
  }

  virtual int32_t initialize() = 0;
  PSServer *get_server() { return _server; }

 protected:
  size_t _rank;
  PSServer *_server;
  const ServerParameter *_config;
};
REGISTER_PSCORE_REGISTERER(PsBaseService);

class PSServerFactory {
 public:
  static PSServer *create(const PSParameter &config);
};
}  // namespace distributed
}  // namespace paddle
