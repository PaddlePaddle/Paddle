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

#pragma once

#include "gflags/gflags.h"
#include "paddle/fluid/distributed/ps/table/depends/rocksdb_warpper.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"

namespace paddle {
namespace distributed {

class SSDSparseTable : public MemorySparseTable {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  SSDSparseTable() {}
  virtual ~SSDSparseTable() {}

  int32_t Initialize() override;
  int32_t InitializeShard() override;

  // exchange data
  int32_t UpdateTable();

  int32_t Pull(TableContext& context) override {
    CHECK(context.value_type == Sparse);
    float* pull_values = context.pull_context.values;
    const PullSparseValue& pull_value = context.pull_context.pull_value;
    return PullSparse(pull_values, pull_value.feasigns_, pull_value.numel_);
  }

  int32_t Push(TableContext& context) override {
    const uint64_t* keys = context.push_context.keys;
    const float* values = context.push_context.values;
    size_t num = context.num;
    return PushSparse(keys, values, num);
  }

  virtual int32_t PullSparse(float* pull_values, const uint64_t* keys,
                             size_t num);
  virtual int32_t PushSparse(const uint64_t* keys, const float* values,
                             size_t num);

  int32_t Flush() override { return 0; }
  virtual int32_t Shrink(const std::string& param) override;
  virtual void Clear() override {
    for (size_t i = 0; i < _real_local_shard_num; ++i) {
      _local_shards[i].clear();
    }
  }

  virtual int32_t Save(const std::string& path,
                       const std::string& param) override;
  virtual int32_t SaveCache(
      const std::string& path, const std::string& param,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel) override;
  virtual double GetCacheThreshold() override { return _local_show_threshold; }
  virtual int64_t CacheShuffle(
      const std::string& path, const std::string& param, double cache_threshold,
      std::function<std::future<int32_t>(int msg_type, int to_pserver_id,
                                         std::string& msg)>
          send_msg_func,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel,
      const std::vector<Table*>& table_ptrs) override;
  //加载path目录下数据
  virtual int32_t Load(const std::string& path,
                       const std::string& param) override;
  //加载path目录下数据[start_idx, end_idx)
  virtual int32_t Load(size_t start_idx, size_t end_idx,
                       const std::vector<std::string>& file_list,
                       const std::string& param);
  int64_t LocalSize();

 private:
  RocksDBHandler* _db;
  int64_t _cache_tk_size;
  double _local_show_threshold{0.0};
};

}  // namespace distributed
}  // namespace paddle
