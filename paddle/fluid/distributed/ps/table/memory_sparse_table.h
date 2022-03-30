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
  virtual int32_t PullDense(float* pull_values, size_t num) { return 0; }
  virtual int32_t PushDenseParam(const float* values, size_t num) { return 0; }
  virtual int32_t PushDense(const float* values, size_t num) { return 0; }
  // unused method end

  virtual int32_t Pull(TableContext& context);
  virtual int32_t Push(TableContext& context);

  virtual int32_t Initialize();
  virtual int32_t InitializeShard() { return 0; }
  virtual int32_t InitializeValue();

  virtual int32_t Load(const std::string& path, const std::string& param);

  virtual int32_t Save(const std::string& path, const std::string& param);

  int32_t LoadLocalFS(const std::string& path, const std::string& param);
  int32_t SaveLocalFS(const std::string& path, const std::string& param,
                      const std::string& prefix);

  int64_t LocalSize();
  int64_t LocalMFSize();

  virtual std::pair<int64_t, int64_t> PrintTableStat();
  virtual int32_t PullSparse(float* values, const PullSparseValue& pull_value);

  virtual int32_t PullSparsePtr(char** pull_values, const uint64_t* keys,
                                size_t num);

  virtual int32_t PushSparse(const uint64_t* keys, const float* values,
                             size_t num);

  virtual int32_t PushSparse(const uint64_t* keys, const float** values,
                             size_t num);

  virtual int32_t Flush();
  virtual int32_t Shrink(const std::string& param);
  virtual void Clear();

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
