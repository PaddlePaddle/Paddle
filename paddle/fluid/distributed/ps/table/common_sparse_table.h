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
#include "paddle/fluid/distributed/ps/table/depends/initializers.h"
#include "paddle/fluid/distributed/ps/table/depends/large_scale_kv.h"
#include "paddle/fluid/distributed/ps/table/depends/sparse.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/pten/core/utils/rw_lock.h"

#define PSERVER_SAVE_SUFFIX ".shard"

namespace paddle {
namespace distributed {

class SparseOptimizer;

enum SaveMode { all, base, delta };

struct Meta {
  std::string param;
  int shard_id;
  std::vector<std::string> names;
  std::vector<int> dims;
  uint64_t count;
  std::unordered_map<std::string, int> dims_map;

  explicit Meta(const std::string& metapath) {
    std::ifstream file(metapath);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      if (StartWith(line, "#")) {
        continue;
      }
      auto pairs = paddle::string::split_string<std::string>(line, "=");
      PADDLE_ENFORCE_EQ(
          pairs.size(), 2,
          paddle::platform::errors::InvalidArgument(
              "info in %s except k=v, but got %s", metapath, line));

      if (pairs[0] == "param") {
        param = pairs[1];
      }
      if (pairs[0] == "shard_id") {
        shard_id = std::stoi(pairs[1]);
      }
      if (pairs[0] == "row_names") {
        names = paddle::string::split_string<std::string>(pairs[1], ",");
      }
      if (pairs[0] == "row_dims") {
        auto dims_strs =
            paddle::string::split_string<std::string>(pairs[1], ",");
        for (auto& str : dims_strs) {
          dims.push_back(std::stoi(str));
        }
      }
      if (pairs[0] == "count") {
        count = std::stoull(pairs[1]);
      }
    }
    for (int x = 0; x < names.size(); ++x) {
      dims_map[names[x]] = dims[x];
    }
  }

  Meta(std::string param, int shard_id, std::vector<std::string> row_names,
       std::vector<int> dims, uint64_t count) {
    this->param = param;
    this->shard_id = shard_id;
    this->names = row_names;
    this->dims = dims;
    this->count = count;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "param=" << param << "\n";
    ss << "shard_id=" << shard_id << "\n";
    ss << "row_names=" << paddle::string::join_strings(names, ',') << "\n";
    ss << "row_dims=" << paddle::string::join_strings(dims, ',') << "\n";
    ss << "count=" << count << "\n";
    return ss.str();
  }
};

class CommonSparseTable : public SparseTable {
 public:
  CommonSparseTable() { rwlock_.reset(new pten::RWLock); }
  virtual ~CommonSparseTable() {}

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
  virtual int32_t initialize_optimizer();
  virtual int32_t initialize_recorder();

  virtual int32_t load(const std::string& path, const std::string& param);

  virtual int32_t save(const std::string& path, const std::string& param);

  void SaveMetaToText(std::ostream* os, const CommonAccessorParameter& common,
                      const size_t shard_idx, const int64_t total);

  int64_t SaveValueToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
                          std::shared_ptr<::ThreadPool> pool, const int mode,
                          int shard_id);

  virtual void ProcessALine(const std::vector<std::string>& columns,
                            const Meta& meta, const int64_t id,
                            std::vector<std::vector<float>>* values);

  virtual int64_t LoadFromText(
      const std::string& valuepath, const std::string& metapath,
      const int pserver_id, const int pserver_num, const int local_shard_num,
      std::vector<std::shared_ptr<ValueBlock>>* blocks);

  virtual std::pair<int64_t, int64_t> print_table_stat();
  virtual int32_t pull_sparse(float* values, const PullSparseValue& pull_value);

  virtual int32_t pull_sparse_ptr(char** pull_values, const uint64_t* keys,
                                  size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float* values,
                              size_t num);

  virtual int32_t push_sparse(const uint64_t* keys, const float** values,
                              size_t num);

  // only for sparse geo table
  virtual int32_t push_sparse_param(const uint64_t* keys, const float* values,
                                    size_t num);

  virtual int32_t set_global_lr(float* lr) override;

  virtual int32_t pour();
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

  bool sync = false;
  int param_dim_ = 0;
  int param_offset_ = 0;

  std::unordered_map<std::string, int> value_idx_;
  std::vector<std::string> value_names_;
  std::vector<int> value_dims_;
  std::vector<int> value_offsets_;
  std::vector<std::string> initializer_attrs_;

  std::shared_ptr<SparseOptimizer> optimizer_;
  std::vector<std::shared_ptr<ValueBlock>> shard_values_;
  std::unordered_map<uint64_t, ReservoirValue<float>> pull_reservoir_;
  std::unique_ptr<pten::RWLock> rwlock_{nullptr};
};

}  // namespace distributed
}  // namespace paddle
