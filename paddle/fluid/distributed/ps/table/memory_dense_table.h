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
#include <string>
#include "Eigen/Dense"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/depends/dense.h"
#include "paddle/fluid/distributed/ps/table/depends/initializers.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class DenseOptimizer;

class MemoryDenseTable : public Table {
 public:
  MemoryDenseTable() {}
  virtual ~MemoryDenseTable() {}
  int32_t Initialize() override;
  int32_t InitializeShard() override { return 0; }
  void CreateInitializer(const std::string& attr, const std::string& name);
  int32_t InitializeValue();
  int32_t InitializeOptimizer();

  int32_t Pull(TableContext& context) override;
  int32_t Push(TableContext& context) override;

  int32_t PullDense(float* pull_values, size_t num);
  int32_t PushDenseParam(const float* values, size_t num);
  int32_t PushDense(const float* values, size_t num);
  int32_t Pour() override;
  int32_t SetGlobalLR(float* lr) override;

  int32_t Load(const std::string& path, const std::string& param) override;
  int32_t Save(const std::string& path, const std::string& param) override;

  int32_t Flush() override { return 0; }
  int32_t Shrink(const std::string& param) override { return 0; }
  void Clear() override { return; }
  void* GetShard(size_t shard_idx) override { return 0; }

 protected:
  int32_t _PushDense(const float* values, size_t num);

 private:
  const int task_pool_size_ = 10;
  bool sync = true;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  int param_dim_ = 0;
  int param_idx_ = 0;
  std::shared_ptr<DenseOptimizer> optimizer_;
  std::vector<std::vector<float>> values_;
  ReservoirValue<float> pull_reservoir_;
  std::unordered_map<std::string, Initializer*> initializers_;
  std::unordered_map<std::string, int> names_index_;
  int total_dim_ = 0;
  int fixed_len_params_dim_ = 0;    // used for save/load
  std::vector<int> param_col_ids_;  // used for save/load
};

}  // namespace distributed
}  // namespace paddle
