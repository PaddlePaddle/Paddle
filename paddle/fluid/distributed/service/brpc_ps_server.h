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

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"

#include <memory>
#include <vector>
#include "paddle/fluid/distributed/service/brpc_utils.h"
#include "paddle/fluid/distributed/service/server.h"

namespace paddle {
namespace distributed {

class BrpcPsServer : public PSServer {
 public:
  BrpcPsServer() {}
  virtual ~BrpcPsServer() {}
  virtual uint64_t start(const std::string &ip, uint32_t port);
  virtual int32_t stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    stoped_ = true;
    cv_.notify_all();

    _server.Stop(1000);
    _server.Join();
    return 0;
  }
  virtual int32_t port();

 private:
  virtual int32_t initialize();
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stoped_ = false;
  brpc::Server _server;
  std::shared_ptr<PsBaseService> _service;
  std::vector<std::shared_ptr<brpc::Channel>> _pserver_channels;
};

class BrpcPsService;

typedef int32_t (BrpcPsService::*serviceHandlerFunc)(
    Table *table, const PsRequestMessage &request, PsResponseMessage &response,
    brpc::Controller *cntl);

class BrpcPsService : public PsBaseService {
 public:
  virtual int32_t initialize() override;

  virtual void service(::google::protobuf::RpcController *controller,
                       const PsRequestMessage *request,
                       PsResponseMessage *response,
                       ::google::protobuf::Closure *done) override;

 private:
  int32_t initialize_shard_info();
  int32_t pull_dense(Table *table, const PsRequestMessage &request,
                     PsResponseMessage &response, brpc::Controller *cntl);
  int32_t push_dense(Table *table, const PsRequestMessage &request,
                     PsResponseMessage &response, brpc::Controller *cntl);
  int32_t push_dense_param(Table *table, const PsRequestMessage &request,
                           PsResponseMessage &response, brpc::Controller *cntl);
  int32_t push_sparse_param(Table *table, const PsRequestMessage &request,
                            PsResponseMessage &response,
                            brpc::Controller *cntl);
  int32_t pull_sparse(Table *table, const PsRequestMessage &request,
                      PsResponseMessage &response, brpc::Controller *cntl);
  int32_t pull_geo_param(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t barrier(Table *table, const PsRequestMessage &request,
                  PsResponseMessage &response, brpc::Controller *cntl);
  int32_t push_sparse(Table *table, const PsRequestMessage &request,
                      PsResponseMessage &response, brpc::Controller *cntl);
  int32_t load_one_table(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t load_all_table(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t save_one_table(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t save_all_table(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t shrink_table(Table *table, const PsRequestMessage &request,
                       PsResponseMessage &response, brpc::Controller *cntl);
  int32_t clear_one_table(Table *table, const PsRequestMessage &request,
                          PsResponseMessage &response, brpc::Controller *cntl);
  int32_t clear_all_table(Table *table, const PsRequestMessage &request,
                          PsResponseMessage &response, brpc::Controller *cntl);
  int32_t stop_server(Table *table, const PsRequestMessage &request,
                      PsResponseMessage &response, brpc::Controller *cntl);
  int32_t start_profiler(Table *table, const PsRequestMessage &request,
                         PsResponseMessage &response, brpc::Controller *cntl);
  int32_t stop_profiler(Table *table, const PsRequestMessage &request,
                        PsResponseMessage &response, brpc::Controller *cntl);

  int32_t print_table_stat(Table *table, const PsRequestMessage &request,
                           PsResponseMessage &response, brpc::Controller *cntl);

  int32_t push_global_step(Table *table, const PsRequestMessage &request,
                           PsResponseMessage &response, brpc::Controller *cntl);

  bool _is_initialize_shard_info;
  std::mutex _initialize_shard_mutex;
  std::unordered_map<int32_t, serviceHandlerFunc> _service_handler_map;
  std::unordered_map<int32_t, serviceHandlerFunc> _msg_handler_map;
  std::vector<float> _ori_values;
};

class DownpourPServerBrpcClosure : public PServerClosure {
 public:
  DownpourPServerBrpcClosure(size_t num, PServerCallBack callback)
      : PServerClosure(callback) {
    _waiting_num = num;
    _cntls.resize(num);
    _requests.resize(num);
    _responses.resize(num);
    for (size_t i = 0; i < num; ++i) {
      _cntls[i].reset(new brpc::Controller());
    }
  }
  virtual ~DownpourPServerBrpcClosure() {}

  virtual void Run() override {
    if (_waiting_num.fetch_sub(1) == 1) {
      _callback(this);
      delete this;
    }
  }
  PsRequestMessage *request(size_t i) { return &_requests[i]; }
  PsResponseMessage *response(size_t i) { return &_responses[i]; }
  brpc::Controller *cntl(size_t i) { return _cntls[i].get(); }
  int check_response(size_t request_idx, int cmd_id) { return 1; }
  int check_save_response(size_t request_idx, int cmd_id) { return 1; }

 private:
  std::atomic<int32_t> _waiting_num;
  std::vector<PsRequestMessage> _requests;
  std::vector<PsResponseMessage> _responses;
  std::vector<std::shared_ptr<brpc::Controller>> _cntls;
};
}  // namespace distributed
}  // namespace paddle
