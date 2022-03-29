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

#include <assert.h>
// #include <pthread.h>
#include <stdint.h>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#include "paddle/fluid/distributed/ps/table/depends/geo_recorder.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class GeoRecorder;

class MemorySparseGeoTable : public SparseTable {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  MemorySparseGeoTable() { _geo_recorder = nullptr; }
  virtual ~MemorySparseGeoTable() {}

  virtual int32_t initialize();
  virtual int32_t initialize_shard() { return 0; }
  virtual int32_t load(const std::string& path, const std::string& param) {
    return 0;
  }
  virtual int32_t save(const std::string& path, const std::string& param) {
    return 0;
  }
  virtual int32_t Pull(TableContext& context) { return 0; }
  virtual int32_t Push(TableContext& context) { return 0; }
  virtual int32_t flush() { return 0; }
  virtual int32_t shrink(const std::string& param) { return 0; }
  virtual void clear() { return; }
  virtual int32_t pull_sparse(float* values, const PullSparseValue& pull_value);

  int32_t push_sparse_param(const uint64_t* keys, const float* values,
                            size_t num);
  // TODO(zhaocaibei123): change to pull_sparse, and rename pull_sparse
  int32_t pull_geo_param(const uint32_t trainer_id, std::vector<float>* values,
                         std::vector<uint64_t>* keys);

  int32_t push_sparse(const uint64_t* keys, const float* values,
                      size_t num) override;

  int32_t _push_sparse(const uint64_t* keys, const float* values, size_t num);
  // int32_t _pull_sparse(float* pull_values, const PullSparseValue&
  // pull_value);

 private:
  std::shared_ptr<GeoRecorder> _geo_recorder;
  const int _task_pool_size = 10;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::unique_ptr<shard_type[]> _local_shards;
  int _dim;
};

}  // namespace distributed
}  // namespace paddle
