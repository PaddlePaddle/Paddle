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

#include <ThreadPool.h>

#include <memory>
#include <string>
#include <vector>

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
namespace brpc {
class Channel;
class Controller;
}  // namespace brpc
namespace google {
namespace protobuf {
class Closure;
class RpcController;
}  // namespace protobuf
}  // namespace google

namespace paddle {
namespace distributed {

struct Region;

class DownpourPsClientService : public PsService {
 public:
  DownpourPsClientService() {}
  virtual ~DownpourPsClientService() {}

  virtual int32_t Configure(PSClient *client, size_t rank_id) {
    _client = client;
    _rank = rank_id;
    return 0;
  }

  virtual void service(::google::protobuf::RpcController *controller,
                       const PsRequestMessage *request,
                       PsResponseMessage *response,
                       ::google::protobuf::Closure *done);

  virtual void FLService(::google::protobuf::RpcController *controller UNUSED,
                         const CoordinatorReqMessage *request,
                         CoordinatorResMessage *response,
                         ::google::protobuf::Closure *done) {
    brpc::ClosureGuard done_guard(done);
    size_t client_id = request->client_id();
    PADDLE_ENFORCE_EQ(
        client_id,
        (_client->_client_id),
        common::errors::PreconditionNotMet(
            "Wrong request client's id. Expect to match self. But received "
            "request client's id = %lu and self = %lu.",
            client_id,
            (_client->_client_id)));
    _fl_strategy = request->str_params();
    _is_fl_strategy_ready = true;
    response->set_err_code(0);
    response->set_err_msg("");
    VLOG(0) << "fl-ps > DownpourPsClientService::FLService finished!";
    return;
  }

 public:
  std::string _fl_strategy;
  bool _is_fl_strategy_ready = false;

 protected:
  size_t _rank;
  PSClient *_client;
};

class FlClientBrpcClosure : public PSClientClosure {
 public:
  FlClientBrpcClosure(size_t num, PSClientCallBack callback)
      : PSClientClosure(callback) {
    _waiting_num = num;

    _cntls.resize(num);
    _requests.resize(num);
    _responses.resize(num);
    for (size_t i = 0; i < num; ++i) {
      _cntls[i].reset(new brpc::Controller());
    }
  }
  virtual ~FlClientBrpcClosure() {}
  void Run() override {
    if (_waiting_num.fetch_sub(1) == 1) {
      _callback(this);
      delete this;
    }
  }
  CoordinatorReqMessage *request(size_t i) { return &_requests[i]; }
  CoordinatorResMessage *response(size_t i) { return &_responses[i]; }
  brpc::Controller *cntl(size_t i) { return _cntls[i].get(); }
  int check_response(size_t request_idx, int cmd_id);
  int check_save_response(size_t request_idx, int cmd_id);
  std::string get_response(size_t request_idx, int cmd_id);

 private:
  std::atomic<int32_t> _waiting_num;
  std::vector<CoordinatorReqMessage> _requests;
  std::vector<CoordinatorResMessage> _responses;
  std::vector<std::shared_ptr<brpc::Controller>> _cntls;
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
  void Run() override {
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

struct SharedSparsePushData {
  SharedSparsePushData() {}
  ~SharedSparsePushData() noexcept {}
  size_t kv_num;
  std::vector<uint64_t> key_list;
  std::vector<std::string> value_list;
};
struct SparsePushTaskData {
  std::vector<SharedSparsePushData> shared_data;  // sparse数据按key hash分片
};

// push sparse 对象池
struct SparseTaskPool {
  std::shared_ptr<SparsePushTaskData> get() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_pool.empty()) {
      return std::make_shared<SparsePushTaskData>();
    } else {
      auto ret = _pool.back();
      _pool.pop_back();
      return ret;
    }
  }
  void push(std::shared_ptr<SparsePushTaskData> data) {
    std::lock_guard<std::mutex> lock(_mutex);
    _pool.push_back(std::move(data));
  }
  std::vector<std::shared_ptr<SparsePushTaskData>> _pool;
  std::mutex _mutex;
};

template <class T>
struct array_deleter {
  void operator()(T *&x) const { delete[] x; }  // NOLINT
};

class BrpcPsClient : public PSClient {
 public:
  BrpcPsClient() {}
  virtual ~BrpcPsClient() {
    if (_running) {
      Flush();
      _running = false;
    }
    if (_async_push_dense_thread.joinable()) {
      _async_push_dense_thread.join();
    }
    if (_async_push_sparse_thread.joinable()) {
      _async_push_sparse_thread.join();
    }
    if (_server_started) {
      _server.Stop(1000);
      _server.Join();
      _server_started = false;
    }
  }
  virtual int32_t CreateClient2ClientConnection(int pserver_timeout_ms,
                                                int pserver_connect_timeout_ms,
                                                int max_retry);
  std::future<int32_t> Shrink(uint32_t table_id,
                              const std::string threshold) override;
  std::future<int32_t> Load(const std::string &epoch,
                            const std::string &mode) override;
  std::future<int32_t> Load(uint32_t table_id,
                            const std::string &epoch,
                            const std::string &mode) override;

  std::future<int32_t> Save(const std::string &epoch,
                            const std::string &mode) override;

  std::future<int32_t> Save(uint32_t table_id,
                            const std::string &epoch,
                            const std::string &mode) override;

  std::future<int32_t> Clear() override;

  std::future<int32_t> Clear(uint32_t table_id) override;

  std::future<int32_t> Revert() override;
  std::future<int32_t> CheckSavePrePatchDone() override;

  std::future<int32_t> StopServer() override;

  std::future<int32_t> StartProfiler() override;
  std::future<int32_t> StopProfiler() override;

  void FinalizeWorker() override;

  virtual std::future<int32_t> PullDense(Region *regions,
                                         size_t region_num,
                                         size_t table_id);

  virtual std::future<int32_t> PushDenseParam(const Region *regions,
                                              size_t region_num,
                                              size_t table_id);

  virtual std::future<int32_t> PushDense(const Region *regions,
                                         size_t region_num,
                                         size_t table_id);
  void PushDenseTaskConsume();
  virtual std::future<int32_t> PullSparse(float **select_values,
                                          size_t table_id,
                                          const uint64_t *keys,
                                          size_t num,
                                          bool is_training);
  virtual std::future<int32_t> PullSparseParam(float **select_values,
                                               size_t table_id,
                                               const uint64_t *keys,
                                               size_t num,
                                               bool is_training);

  virtual std::future<int32_t> PrintTableStat(uint32_t table_id,
                                              uint16_t pass_id,
                                              size_t threshold);

  virtual std::future<int32_t> Barrier(size_t table_id, uint32_t barrier_type);

  virtual std::future<int32_t> PullGeoParam(size_t table_id,
                                            std::vector<float> *values,
                                            std::vector<uint64_t> *keys,
                                            int pserver_idx);
  virtual std::future<int32_t> PushGlobalStep(int table_id,
                                              int64_t *total_send_data,
                                              void *done);
  virtual std::future<int32_t> Flush();

  std::future<int32_t> SendClient2ClientMsg(int msg_type,
                                            int to_client_id,
                                            const std::string &msg) override;

  // for local save sparse
  virtual int32_t RecvAndSaveTable(const uint64_t table_id,
                                   const std::string &path);

  std::future<int32_t> CacheShuffle(
      uint32_t table_id,
      const std::string &path,
      const std::string &mode,
      const std::string &cache_threshold) override;

  std::future<int32_t> CacheShuffleMultiTable(
      std::vector<int> tables,
      const std::string &path,
      const std::string &mode,
      const std::string &cache_threshold);

  std::future<int32_t> SaveCache(uint32_t table_id,
                                 const std::string &path,
                                 const std::string &mode) override;

  std::future<int32_t> GetCacheThreshold(uint32_t table_id,
                                         double &cache_threshold) override;

  void PrintQueueSize();
  void PrintQueueSizeThread();

 protected:
  virtual size_t GetServerNums() { return _server_channels.size(); }
  inline brpc::Channel *GetSparseChannel(size_t server_id) {
    return _server_channels[server_id][0].get();
  }
  inline brpc::Channel *GetDenseChannel(size_t server_id) {
    return _server_channels[server_id][1].get();
  }
  inline brpc::Channel *GetCmdChannel(size_t server_id) {
    return _server_channels[server_id][2].get();
  }
  int32_t Initialize() override;

  // for fl
 public:
  virtual int32_t InitializeFlWorker(const std::string &self_endpoint);
  int32_t StartFlClientService(const std::string &self_endpoint);
  virtual void PushFLClientInfoSync(const std::string &fl_client_info);
  std::string PullFlStrategy();
  // for fl

 private:
  inline uint32_t DenseDimPerShard(uint32_t dense_dim_total,
                                   uint32_t shard_num) {
    return dense_dim_total / shard_num + 1;
  }

  std::future<int32_t> SendCmd(uint32_t table_id,
                               int cmd_id,
                               const std::vector<std::string> &param);

  std::future<int32_t> SendSaveCmd(uint32_t table_id,
                                   int cmd_id,
                                   const std::vector<std::string> &param);

  bool _running = false;
  bool _flushing = false;
  std::atomic<uint32_t> _async_call_num;  // 异步请求计数

  // 异步push dense task
  std::thread _async_push_dense_thread;
  typedef AsyncRequestTask<std::shared_ptr<std::vector<float>>> DenseAsyncTask;
  std::unordered_map<uint32_t, paddle::framework::Channel<DenseAsyncTask *>>
      _push_dense_task_queue_map;
  // 异步push sparse task
  std::thread _async_push_sparse_thread;
  typedef AsyncRequestTask<std::shared_ptr<SparsePushTaskData>> SparseAsyncTask;
  std::unordered_map<uint32_t, paddle::framework::Channel<SparseAsyncTask *>>
      _push_sparse_task_queue_map;
  std::unordered_map<uint32_t, uint32_t> _push_sparse_merge_count_map;

  std::thread _print_thread;

  int PushSparseAsyncShardMerge(
      std::vector<std::shared_ptr<SparseAsyncTask>> &task_list,  // NOLINT
      std::vector<int> &request_kv_num,                          // NOLINT
      int table_id,
      int shard_idx,
      ValueAccessor *accessor);

  int PushSparseAsyncShardPush(
      std::vector<std::shared_ptr<SparseAsyncTask>> &task_list,  // NOLINT
      std::vector<int> &request_kv_num,                          // NOLINT
      int table_id,
      int shard_idx,
      DownpourBrpcClosure *closure,
      ValueAccessor *accessor);

  SparseTaskPool _sparse_task_pool;

  std::vector<std::shared_ptr<brpc::Channel>>
      _client_channels;  // client2client
  std::vector<std::array<std::shared_ptr<brpc::Channel>, 3>>
      _server_channels;  // client2server
  std::vector<std::array<std::shared_ptr<brpc::Channel>, 1>>
      _coordinator_channels;  // client2coordinator
  std::future<int32_t> PushDenseRawGradient(int table_id,
                                            float *total_send_data,
                                            size_t total_send_data_size,
                                            void *done) override;

  std::future<int32_t> PushSparseRawGradient(size_t table_id,
                                             const uint64_t *keys,
                                             const float **update_values,
                                             size_t num,
                                             void *done) override;

  std::future<int32_t> PushSparseRawGradientPartial(size_t table_id,
                                                    const uint64_t *keys,
                                                    const float **update_values,
                                                    uint32_t num,
                                                    void *done,
                                                    int pserver_idx) override;

  std::future<int32_t> PushSparseParam(size_t table_id,
                                       const uint64_t *keys,
                                       const float **update_values,
                                       size_t num,
                                       void *done) override;
  std::future<int32_t> PushSparse(size_t table_id,
                                  const uint64_t *keys,
                                  const float **update_values,
                                  size_t num) override;
  void PushSparseTaskConsume();

 private:
  int32_t StartClientService();

  void PushDenseRawGradient(std::shared_ptr<DenseAsyncTask> &task,  // NOLINT
                            float *total_send_data,
                            size_t total_send_data_size,
                            DownpourBrpcClosure *closure);
  float _mae = 0;
  float _mse = 0;
  uint16_t _push_times = 0;
  brpc::Server _server;
  brpc::Server _fl_server;
  DownpourPsClientService _service;
  bool _server_started = false;
  std::atomic_uint grad_num_{0};
};
}  // namespace distributed
}  // namespace paddle
