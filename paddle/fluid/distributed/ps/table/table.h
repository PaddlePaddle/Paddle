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

#include <assert.h>
#include <atomic>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <utility>
#include "paddle/fluid/distributed/common/afs_warpper.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/depends/sparse_utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

enum ValueType { Sparse = 0, Dense = 1 };

struct PullContext {
  const uint64_t *keys;
  const PullSparseValue pull_value;
  float *values;
  char **ptr_values;
};

struct TablePushContext {
  const uint64_t *keys;
  const float *values;
  const float **ptr_values;
};

struct TableContext {
  ValueType value_type;
  PullContext pull_context;
  TablePushContext push_context;
  size_t num;
  bool use_ptr;
};

class Table {
 public:
  Table() {}
  virtual ~Table() {}
  virtual int32_t initialize(const TableParameter &config,
                             const FsClientParameter &fs_config);

  virtual int32_t Pull(TableContext &context) = 0;
  virtual int32_t Push(TableContext &context) = 0;
  virtual int32_t pull_dense(float *values, size_t num) = 0;
  virtual int32_t push_dense(const float *values, size_t num) = 0;
  // for push global_step
  virtual int32_t push_dense(const int64_t *values, const int32_t trainer_id) {
    return 0;
  }
  virtual int32_t push_dense_param(const float *values, size_t num) {
    return 0;
  }

  virtual int32_t pull_sparse_ptr(char **pull_values, const uint64_t *keys,
                                  size_t num) {
    VLOG(0) << "NOT IMPLEMENT";
    return 0;
  }
  virtual int32_t pull_sparse(float *values,
                              const PullSparseValue &pull_value) = 0;
  virtual int32_t push_sparse(const uint64_t *keys, const float *values,
                              size_t num) = 0;
  virtual int32_t push_sparse(const uint64_t *keys, const float **values,
                              size_t num) {
    return 0;
  }
  virtual int32_t push_sparse_param(const uint64_t *keys, const float *values,
                                    size_t num) {
    return 0;
  }

  // only for sparse geo table
  virtual int32_t pull_geo_param(const uint32_t trainer_id,
                                 std::vector<float> *values,
                                 std::vector<uint64_t> *keys) {
    return 0;
  }

  // only for barrier
  virtual int32_t barrier(const uint32_t trainer_id,
                          const std::string barrier_type) {
    return 0;
  }

  // only for barrier table
  virtual int32_t set_table_map(
      std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map) {
    return 0;
  }

  // only for tensor table
  virtual int32_t set_program_env(
      framework::Scope *scope, platform::Place place,
      const std::vector<framework::ProgramDesc> *sub_program) {
    return 0;
  }

  virtual int32_t set_global_lr(float *lr) {
    _global_lr = lr;
    return 0;
  }

  virtual int32_t pour() { return 0; }

  virtual void clear() = 0;
  virtual int32_t flush() = 0;
  virtual int32_t shrink(const std::string &param) = 0;

  // 指定加载路径
  virtual int32_t load(const std::string &path,
                       const std::string &converter) = 0;
  // 指定保存路径
  virtual int32_t save(const std::string &path,
                       const std::string &converter) = 0;

  virtual int32_t set_shard(size_t shard_idx, size_t shard_num) {
    _shard_idx = shard_idx;
    _shard_num = shard_num;
    return initialize_shard();
  }

  inline std::shared_ptr<ValueAccessor> value_accesor() {
    return _value_accesor;
  }

  virtual void *get_shard(size_t shard_idx) = 0;
  virtual std::pair<int64_t, int64_t> print_table_stat() { return {0, 0}; }

 protected:
  virtual int32_t initialize() = 0;
  virtual int32_t initialize_accessor();
  virtual int32_t initialize_shard() = 0;
  virtual std::string table_dir(const std::string &model_dir) {
    return paddle::string::format_string("%s/%03d/", model_dir.c_str(),
                                         _config.table_id());
  }

  size_t _shard_idx;  // table 分片编号
  size_t _shard_num;  // table 分片总数
  TableParameter _config;
  float *_global_lr = nullptr;
  std::shared_ptr<ValueAccessor> _value_accesor;
  AfsClient _afs_client;
};
REGISTER_PSCORE_REGISTERER(Table);

class TableManager {
 public:
  static TableManager &instance() {
    static TableManager manager;
    return manager;
  }
  int32_t initialize();

 private:
  TableManager() {}
  ~TableManager() {}
};

}  // namespace distributed
}  // namespace paddle
