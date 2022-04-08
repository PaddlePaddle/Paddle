// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License 0//
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
#include "paddle/fluid/distributed/ps/service/ps_client.h"

namespace paddle {
namespace distributed {

class Table;

class PsLocalClient : public PSClient {
 public:
  PsLocalClient() {}
  virtual ~PsLocalClient() { _running = false; }
  virtual int32_t CreateClient2ClientConnection(int pslib_timeout_ms,
                                                int pslib_connect_timeout_ms,
                                                int max_retry) {
    return 0;
  }

  virtual ::std::future<int32_t> Shrink(uint32_t table_id,
                                        const std::string threshold) override;
  virtual ::std::future<int32_t> Load(const std::string& epoch,
                                      const std::string& mode) override;
  virtual ::std::future<int32_t> Load(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode) override;

  virtual ::std::future<int32_t> Save(const std::string& epoch,
                                      const std::string& mode) override;
  virtual ::std::future<int32_t> Save(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode) override;

  virtual ::std::future<int32_t> Clear() override;
  virtual ::std::future<int32_t> Clear(uint32_t table_id) override;

  virtual ::std::future<int32_t> StopServer() override;

  virtual void FinalizeWorker() override {}
  virtual ::std::future<int32_t> PullDense(Region* regions, size_t region_num,
                                           size_t table_id);

  virtual ::std::future<int32_t> PushDense(const Region* regions,
                                           size_t region_num, size_t table_id);

  virtual ::std::future<int32_t> PushDenseParam(const Region* regions,
                                                size_t region_num,
                                                size_t table_id);

  virtual ::std::future<int32_t> PullSparse(float** select_values,
                                            size_t table_id,
                                            const uint64_t* keys, size_t num,
                                            bool is_training) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual ::std::future<int32_t> PullSparsePtr(char** select_values,
                                               size_t table_id,
                                               const uint64_t* keys,
                                               size_t num);

  virtual ::std::future<int32_t> PrintTableStat(uint32_t table_id) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }
  virtual ::std::future<int32_t> PushSparse(size_t table_id,
                                            const uint64_t* keys,
                                            const float** update_values,
                                            size_t num);

  virtual ::std::future<int32_t> Flush();
  // server profilera
  virtual std::future<int32_t> StartProfiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  };

  virtual std::future<int32_t> StopProfiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> Barrier(size_t table_id, uint32_t barrier_type) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PullGeoParam(size_t table_id,
                                            std::vector<float>* values,
                                            std::vector<uint64_t>* keys,
                                            int pserver_idx) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PushGlobalStep(int table_id,
                                              int64_t* total_send_data,
                                              void* done) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  // recv table from server and save it in LodTensor
  virtual int32_t RecvAndSaveTable(const uint64_t table_id,
                                   const std::string& path) {
    return 0;
  }

  virtual ::std::future<int32_t> SendClient2ClientMsg(
      int msg_type, int to_client_id, const std::string& msg) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }
  virtual size_t GetServerNums() { return 1; }

  virtual std::future<int32_t> PushDenseRawGradient(int table_id,
                                                    float* total_send_data,
                                                    size_t total_send_data_size,
                                                    void* callback) override;

  virtual std::future<int32_t> PushSparseRawGradient(
      size_t table_id, const uint64_t* keys, const float** update_values,
      size_t num, void* callback) override;

  virtual std::future<int32_t> PushSparseRawGradientPartial(
      size_t table_id, const uint64_t* keys, const float** update_values,
      uint32_t num, void* done, int pserver_idx) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PushSparseParam(size_t table_id,
                                               const uint64_t* keys,
                                               const float** update_values,
                                               size_t num,
                                               void* done) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

 private:
  virtual int32_t Initialize() override;

  std::future<int32_t> done() {
    std::shared_ptr<std::promise<int32_t>> prom =
        std::make_shared<std::promise<int32_t>>();
    std::future<int32_t> fut = prom->get_future();
    prom->set_value(0);
    return fut;
  }

  inline uint32_t DenseDimPerShard(uint32_t dense_dim_total,
                                   uint32_t shard_num) {
    return dense_dim_total / shard_num + 1;
  }

  inline std::unordered_map<uint32_t, std::shared_ptr<Table>>* GetTable() {
    return &_table_map;
  }

  inline Table* GetTable(size_t table_id) {
    auto itr = _table_map.find(table_id);
    if (itr != _table_map.end()) {
      return itr->second.get();
    }
    LOG(ERROR) << "table not found " << table_id;
    return NULL;
  }

  std::unordered_map<uint32_t, std::shared_ptr<Table>> _table_map;

  bool _running = false;
  bool _flushing = false;

 private:
  float _mae = 0;
  float _mse = 0;
  uint16_t _push_times = 0;
};
}
}
