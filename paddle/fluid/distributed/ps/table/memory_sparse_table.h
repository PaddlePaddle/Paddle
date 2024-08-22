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
#include <assert.h>
#include <pthread.h>

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#include "paddle/utils/string/string_helper.h"

#define PSERVER_SAVE_SUFFIX ".shard"

namespace paddle {
namespace distributed {

class MemorySparseTable : public Table {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  MemorySparseTable() {}
  virtual ~MemorySparseTable() {}

  // unused method end
  static int32_t sparse_local_shard_num(uint32_t shard_num,
                                        uint32_t server_num) {
    if (shard_num % server_num == 0) {
      return shard_num / server_num;
    }
    size_t local_shard_num = shard_num / server_num + 1;
    return local_shard_num;
  }

  static size_t get_sparse_shard(uint32_t shard_num,
                                 uint32_t server_num,
                                 uint64_t key) {
    return (key % shard_num) / sparse_local_shard_num(shard_num, server_num);
  }

  int32_t Pull(TableContext& context) override;
  int32_t Push(TableContext& context) override;

  int32_t Initialize() override;
  int32_t InitializeShard() override { return 0; }
  int32_t InitializeValue();

  int32_t Load(const std::string& path, const std::string& param) override;

  int32_t Save(const std::string& path, const std::string& param) override;
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  int32_t Save_v2(const std::string& path, const std::string& param) override;
#endif

  int32_t SaveCache(
      const std::string& path,
      const std::string& param,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel) override;
  virtual double GetCacheThreshold() { return _local_show_threshold; }
  int64_t CacheShuffle(
      const std::string& path,
      const std::string& param,
      double cache_threshold,
      std::function<std::future<int32_t>(
          int msg_type, int to_pserver_id, std::string& msg)> send_msg_func,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel,
      const std::vector<Table*>& table_ptrs) override;
  int64_t LocalSize();
  int64_t LocalMFSize();

  std::pair<int64_t, int64_t> PrintTableStat() override;
  int32_t PullSparse(float* values, const PullSparseValue& pull_value);

  int32_t PullSparsePtr(int shard_id,
                        char** pull_values,
                        const uint64_t* keys,
                        size_t num,
                        uint16_t pass_id);

  int32_t PushSparse(const uint64_t* keys, const float* values, size_t num);

  int32_t PushSparse(const uint64_t* keys, const float** values, size_t num);

  int32_t Flush() override;
  int32_t Shrink(const std::string& param) override;
  void Clear() override;

  void* GetShard(size_t shard_idx) override {
    return &_local_shards[shard_idx];
  }

  virtual void Revert();
  virtual void CheckSavePrePatchDone();

 protected:
  virtual int32_t SavePatch(const std::string& path, int save_param);
  virtual int32_t LoadPatch(const std::vector<std::string>& file_list,
                            int save_param);

  int _task_pool_size = 24;
  int _avg_local_shard_num;
  int _real_local_shard_num;
  int _sparse_table_shard_num;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::unique_ptr<shard_type[]> _local_shards;

  // for patch model
  int _m_avg_local_shard_num;
  int _m_real_local_shard_num;
  int _m_sparse_table_shard_num;
  float _shard_merge_rate{1.0f};
  double _local_show_threshold{0.0};

  std::unique_ptr<shard_type[]> _local_shards_new;
  std::unique_ptr<shard_type[]> _local_shards_patch_model;
  std::thread _save_patch_model_thread;
  bool _use_gpu_graph = false;
};

}  // namespace distributed
}  // namespace paddle
