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
#include "paddle/fluid/distributed/service/ps_client.h"

namespace paddle {
namespace distributed {

class DownpourPsClientService : public PsService {
 public:
  DownpourPsClientService() {}
  virtual ~DownpourPsClientService() {}

  virtual int32_t configure(PSClient *client, size_t rank_id) {
    _client = client;
    _rank = rank_id;
    return 0;
  }
  virtual void service(::google::protobuf::RpcController *controller,
                       const ::paddle::PsRequestMessage *request,
                       ::paddle::PsResponseMessage *response,
                       ::google::protobuf::Closure *done) override;

 protected:
  size_t _rank;
  PSClient *_client;
};

class DownpourBrpcClosure : public PSClientClosure {
 public:
  DownpourBrpcClosure(size_t num, PSClientCallBack callback)
      : PSClientClosure(callback) {
    _waiting_num = num;

    _cntls.resize(num);
    _requests.resize(num);
    _responses.resize(num);
    for (size_t i = 0; i < num; ++i) {
      _cntls[i].reset(new brpc::Controller());
    }
  }
  virtual ~DownpourBrpcClosure() {}
  virtual void Run() override {
    if (_waiting_num.fetch_sub(1) == 1) {
      _callback(this);
      delete this;
    }
  }
  PsRequestMessage *request(size_t i) { return &_requests[i]; }
  PsResponseMessage *response(size_t i) { return &_responses[i]; }
  brpc::Controller *cntl(size_t i) { return _cntls[i].get(); }
  int check_response(size_t request_idx, int cmd_id);
  int check_save_response(size_t request_idx, int cmd_id);
  std::string get_response(size_t request_idx, int cmd_id);

 private:
  std::atomic<int32_t> _waiting_num;
  std::vector<PsRequestMessage> _requests;
  std::vector<PsResponseMessage> _responses;
  std::vector<std::shared_ptr<brpc::Controller>> _cntls;
};

template <class T>
struct array_deleter {
  void operator()(T *&x) const { delete[] x; }
};

class BrpcPsClient : public PSClient {
 public:
  BrpcPsClient() {}
  virtual ~BrpcPsClient() {
    // _running = false;
    // try {
    // _async_push_dense_thread.join();
    // _async_push_sparse_thread.join();
    //} catch (...) {
    //}
  }
  virtual int32_t create_client2client_connection(
      int pserver_timeout_ms, int pserver_connect_timeout_ms, int max_retry);
  // 触发table数据退场
  virtual std::future<int32_t> shrink(uint32_t table_id) override;
  // 全量table进行数据load
  virtual std::future<int32_t> load(const std::string &epoch,
                                    const std::string &mode) override;
  // 指定table数据load
  virtual std::future<int32_t> load(uint32_t table_id, const std::string &epoch,
                                    const std::string &mode) override;

  // 全量table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> save(const std::string &epoch,
                                    const std::string &mode) override;

  // 指定table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> save(uint32_t table_id, const std::string &epoch,
                                    const std::string &mode) override;

  //清空table数据
  virtual std::future<int32_t> clear() override;

  virtual std::future<int32_t> clear(uint32_t table_id) override;

  // server优雅退出
  virtual std::future<int32_t> stop_server() override;

  virtual void finalize_worker() override;
  // pull dense的参数部分，并分块填充到本地网络参数中
  // start和num用于拉取部分参数
  // future结束前keys和values缓冲区不能再次使用
  // client将values按照区块拆包后送交多个sender
  // sender聚集同一区块的请求，累计多个填充buffer
  // server将参数区块中配置的某一维提取返回
  // 返回数据解包后填充到累计的多个buffer中
  virtual std::future<int32_t> pull_dense(Region *regions, size_t region_num,
                                          size_t table_id);

  virtual std::future<int32_t> push_dense_param(const Region *regions,
                                                size_t region_num,
                                                size_t table_id);

  // 使用keys进行pull请求，结果填充values
  // keys和values的个数均为num个，每个value占用select_size空间
  // future结束前keys和values缓冲区不能再次使用
  // 整合多个线程请求的keys，聚集并分散发送到server
  // 返回结果后，遍历buffer并对values赋值
  virtual std::future<int32_t> pull_sparse(float **select_values,
                                           size_t table_id,
                                           const uint64_t *keys, size_t num);

  virtual std::future<int32_t> print_table_stat(uint32_t table_id);

  virtual std::future<int32_t> barrier(size_t table_id, uint32_t barrier_type);

  virtual std::future<int32_t> pull_geo_param(size_t table_id,
                                              std::vector<float> *values,
                                              std::vector<uint64_t> *keys);

  // 确保所有积攒中的请求都发送完成
  virtual std::future<int32_t> flush();

  // client to client, 消息发送
  virtual std::future<int32_t> send_client2client_msg(
      int msg_type, int to_client_id, const std::string &msg) override;

 private:
  virtual int32_t initialize() override;

  //计算每个shard 对 dense的存储量
  inline uint32_t dense_dim_per_shard(uint32_t dense_dim_total,
                                      uint32_t shard_num) {
    return dense_dim_total / shard_num + 1;
  }

  std::future<int32_t> send_cmd(uint32_t table_id, int cmd_id,
                                const std::vector<std::string> &param);

  std::future<int32_t> send_save_cmd(uint32_t table_id, int cmd_id,
                                     const std::vector<std::string> &param);

  inline brpc::Channel *get_sparse_channel(size_t server_id) {
    return _server_channels[server_id][0].get();
  }
  inline brpc::Channel *get_dense_channel(size_t server_id) {
    return _server_channels[server_id][1].get();
  }
  inline brpc::Channel *get_cmd_channel(size_t server_id) {
    return _server_channels[server_id][2].get();
  }

  bool _running = false;
  bool _flushing = false;
  std::atomic<uint32_t> _async_call_num;  //异步请求计数

  std::vector<std::shared_ptr<brpc::Channel>>
      _client_channels;  // client2client
  std::vector<std::array<std::shared_ptr<brpc::Channel>, 3>>
      _server_channels;  // client2server
  virtual void push_dense_raw_gradient(int table_id, float *total_send_data,
                                       size_t total_send_data_size,
                                       void *done) override;

  virtual void push_sparse_raw_gradient(size_t table_id, const uint64_t *keys,
                                        const float **update_values, size_t num,
                                        void *done) override;

  virtual size_t get_server_nums() { return _server_channels.size(); }

 private:
  int32_t start_client_service();

  float _mae = 0;
  float _mse = 0;
  uint16_t _push_times = 0;
  brpc::Server _server;
  DownpourPsClientService _service;
  std::atomic_uint grad_num_{0};
};
}  // namespace distributed
}  // namespace paddle
