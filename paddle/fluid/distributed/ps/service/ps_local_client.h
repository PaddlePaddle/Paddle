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
  virtual int32_t CreateClient2ClientConnection(int pslib_timeout_ms UNUSED,
                                                int pslib_connect_timeout_ms
                                                    UNUSED,
                                                int max_retry UNUSED) {
    return 0;
  }

  virtual ::std::future<int32_t> Shrink(uint32_t table_id,
                                        const std::string threshold);
  virtual ::std::future<int32_t> Load(const std::string& epoch,
                                      const std::string& mode);
  virtual ::std::future<int32_t> Load(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode);

  virtual ::std::future<int32_t> Save(const std::string& epoch,
                                      const std::string& mode);
  virtual ::std::future<int32_t> Save(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode);

  virtual ::std::future<int32_t> Clear();
  virtual ::std::future<int32_t> Clear(uint32_t table_id);

  virtual ::std::future<int32_t> StopServer();

  virtual void FinalizeWorker() {}
  virtual ::std::future<int32_t> PullDense(Region* regions,
                                           size_t region_num,
                                           size_t table_id);

  virtual ::std::future<int32_t> PushDense(const Region* regions,
                                           size_t region_num,
                                           size_t table_id);

  virtual ::std::future<int32_t> PushDenseParam(const Region* regions,
                                                size_t region_num,
                                                size_t table_id);

  virtual ::std::future<int32_t> PullSparse(float** select_values UNUSED,
                                            size_t table_id UNUSED,
                                            const uint64_t* keys UNUSED,
                                            size_t num UNUSED,
                                            bool is_training UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual ::std::future<int32_t> PullSparsePtr(
      const int shard_id,
      char** select_values,
      size_t table_id,
      const uint64_t* keys,
      size_t num,
      uint16_t pass_id,
      const std::vector<std::unordered_map<uint64_t, uint32_t>>& keys2rank_vec,
      const uint16_t& dim_id = 0);

  virtual ::std::future<int32_t> PrintTableStat(uint32_t table_id,
                                                uint16_t pass_id,
                                                size_t threshold);

  virtual ::std::future<int32_t> SaveCacheTable(uint32_t table_id,
                                                uint16_t pass_id,
                                                size_t threshold);

  virtual ::std::future<int32_t> PushSparse(size_t table_id,
                                            const uint64_t* keys,
                                            const float** update_values,
                                            size_t num);

  virtual ::std::future<int32_t> Flush();
  // server profiler
  virtual std::future<int32_t> StartProfiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> StopProfiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> Barrier(size_t table_id UNUSED,
                                       uint32_t barrier_type UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PullGeoParam(size_t table_id UNUSED,
                                            std::vector<float>* values UNUSED,
                                            std::vector<uint64_t>* keys UNUSED,
                                            int pserver_idx UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PushGlobalStep(int table_id UNUSED,
                                              int64_t* total_send_data UNUSED,
                                              void* done UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  // recv table from server and save it in DenseTensor
  virtual int32_t RecvAndSaveTable(const uint64_t table_id UNUSED,
                                   const std::string& path UNUSED) {
    return 0;
  }

  virtual ::std::future<int32_t> SendClient2ClientMsg(int msg_type UNUSED,
                                                      int to_client_id UNUSED,
                                                      const std::string& msg
                                                          UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }
  virtual size_t GetServerNums() { return 1; }

  virtual std::future<int32_t> PushDenseRawGradient(int table_id,
                                                    float* total_send_data,
                                                    size_t total_send_data_size,
                                                    void* callback);

  virtual std::future<int32_t> PushSparseRawGradient(
      size_t table_id,
      const uint64_t* keys,
      const float** update_values,
      size_t num,
      void* callback);

  virtual std::future<int32_t> PushSparseRawGradientPartial(
      size_t table_id UNUSED,
      const uint64_t* keys UNUSED,
      const float** update_values UNUSED,
      uint32_t num UNUSED,
      void* done UNUSED,
      int pserver_idx UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> PushSparseParam(size_t table_id UNUSED,
                                               const uint64_t* keys UNUSED,
                                               const float** update_values
                                                   UNUSED,
                                               size_t num UNUSED,
                                               void* done UNUSED) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual ::std::future<int32_t> SetDayId(size_t table_id, int day_id);

 protected:
  virtual int32_t Initialize();

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
}  // namespace distributed
}  // namespace paddle
