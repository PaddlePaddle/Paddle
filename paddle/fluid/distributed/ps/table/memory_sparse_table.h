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
#include "paddle/fluid/string/string_helper.h"

#define PSERVER_SAVE_SUFFIX ".shard"

namespace paddle {
namespace distributed {

class MemorySparseTable : public SparseTable {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  MemorySparseTable() {}
  virtual ~MemorySparseTable() {}

  // unused method begin
  virtual int32_t pull_dense(float* pull_values, size_t num) { return 0; }
  virtual int32_t push_dense_param(const float* values, size_t num) {
    return 0;
  }
  virtual int32_t push_dense(const float* values, size_t num) { return 0; }
  // unused method end

  virtual int32_t initialize();
  virtual int32_t initialize_shard() { return 0; }
  virtual int32_t initialize_value();

  virtual int32_t load(const std::string& path, const std::string& param);

  virtual int32_t save(const std::string& path, const std::string& param);

  int32_t load_local_fs(const std::string& path, const std::string& param);
  int32_t save_local_fs(const std::string& path, const std::string& param,
                        const std::string& prefix);

  int64_t local_size();
  int64_t local_mf_size();

  virtual std::pair<int64_t, int64_t> print_table_stat();
  virtual int32_t pull_sparse(float* values, const PullSparseValue& pull_value);

  virtual int32_t pull_sparse_ptr(char** pull_values, const uint64_t* keys,
                                  size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float* values,
                              size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float** values,
                              size_t num);

  virtual int32_t flush();
  virtual int32_t shrink(const std::string& param);
  virtual void clear();

 protected:
  virtual int32_t _push_sparse(const uint64_t* keys, const float** values,
                               size_t num);

 protected:
  const int _task_pool_size = 24;
  size_t _avg_local_shard_num;
  size_t _real_local_shard_num;
  size_t _sparse_table_shard_num;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::unique_ptr<shard_type[]> _local_shards;
};

}  // namespace distributed
}  // namespace paddle
