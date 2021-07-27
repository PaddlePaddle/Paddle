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
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/distributed/table/depends/sparse.h"
#include "paddle/fluid/string/string_helper.h"

#define PSERVER_SAVE_SUFFIX ".shard"
using boost::lexical_cast;

namespace paddle {
namespace distributed {


class DownpourSparseTable : public SparseTable {
 public:
  DownpourSparseTable() {}
  virtual ~DownpourSparseTable() {}

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
  //virtual int32_t initialize_optimizer();
  //virtual int32_t initialize_recorder();

  virtual int32_t load(const std::string& path, const std::string& param);

  virtual int32_t save(const std::string& path, const std::string& param);

  //TODO
  //int64_t SaveValueToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
  //                        std::shared_ptr<::ThreadPool> pool, const int mode,
  //                        int shard_id);
  //TODO
  //virtual void ProcessALine(const std::vector<std::string>& columns,
  //                          const Meta& meta, const int64_t id,
  //                          std::vector<std::vector<float>>* values);
  //TODO
  //virtual int64_t LoadFromText(
  //    const std::string& valuepath, const std::string& metapath,
  //    const int pserver_id, const int pserver_num, const int local_shard_num,
  //    std::vector<std::shared_ptr<ValueBlock>>* blocks);

  //TODO: need this?
  virtual std::pair<int64_t, int64_t> print_table_stat();
  virtual int32_t pull_sparse(float* values, const PullSparseValue& pull_value);

  //virtual int32_t pull_sparse_ptr(char** pull_values, const uint64_t* keys,
                                  size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float* values,
                              size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float** values,
                              size_t num);
  //TODO: need this?
  //virtual int32_t set_global_lr(float* lr) override;

  virtual int32_t flush();
  virtual int32_t shrink(const std::string& param);
  virtual void clear();

 protected:
  virtual int32_t _push_sparse(const uint64_t* keys, const float* values,
                               size_t num);
  virtual int32_t _push_sparse(const uint64_t* keys, const float** values,
                               size_t num);

 protected:
  const int task_pool_size_ = 11;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::vector<std::shared_ptr<DownpourValueBlock>> shard_values_;
};

}  // namespace distributed
}  // namespace paddle
