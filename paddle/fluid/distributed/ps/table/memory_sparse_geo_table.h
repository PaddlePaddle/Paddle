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

#include "paddle/common/enforce.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#include "paddle/fluid/distributed/ps/table/depends/geo_recorder.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace distributed {

class GeoRecorder;

class MemorySparseGeoTable : public Table {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  MemorySparseGeoTable() { _geo_recorder = nullptr; }
  virtual ~MemorySparseGeoTable() {}

  int32_t Initialize() override;
  int32_t InitializeShard() override { return 0; }
  int32_t Load(const std::string& path, const std::string& param) override {
    return 0;
  }
  int32_t Save(const std::string& path, const std::string& param) override {
    return 0;
  }
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  int32_t Save_v2(const std::string& path, const std::string& param) override {
    return 0;
  }
#endif
  int32_t Pull(TableContext& context) override;
  int32_t Push(TableContext& context) override;
  int32_t Flush() override { return 0; }
  int32_t Shrink(const std::string& param) override { return 0; }
  void Clear() override { return; }

  int32_t PullSparse(float* values, const PullSparseValue& pull_value);

  int32_t PushSparseParam(const uint64_t* keys,
                          const float* values,
                          size_t num);

  int32_t PullGeoParam(const uint32_t trainer_id,
                       std::vector<float>* values,
                       std::vector<uint64_t>* keys);

  int32_t PushSparse(const uint64_t* keys, const float* values, size_t num);

  int32_t _PushSparse(const uint64_t* keys, const float* values, size_t num);
  // int32_t _pull_sparse(float* pull_values, const PullSparseValue&
  // pull_value);

  void* GetShard(size_t shard_idx) override {
    return &_local_shards[shard_idx];
  }

 private:
  std::shared_ptr<GeoRecorder> _geo_recorder;
  const int _task_pool_size = 10;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::unique_ptr<shard_type[]> _local_shards;
  int _dim;
};

}  // namespace distributed
}  // namespace paddle
