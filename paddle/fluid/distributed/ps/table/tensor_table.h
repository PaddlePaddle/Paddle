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

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class Executor;
class Scope;
struct ExecutorPrepareContext;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

#define LEARNING_RATE_DECAY_COUNTER "@LR_DECAY_COUNTER@"
#define STEP_COUNTER "@PS_STEP_COUNTER@"

class TensorTable : public Table {
 public:
  TensorTable() {}
  virtual ~TensorTable() {}

  int32_t pull_dense(float *values, size_t num) override { return 0; }

  int32_t push_dense(const float *values, size_t num) override { return 0; }

  int32_t pull_sparse(float *values,
                      const PullSparseValue &pull_value) override {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values,
                      size_t num) override {
    return 0;
  }
  int32_t shrink(const std::string &param) override { return 0; }

  virtual void *get_shard(size_t shard_idx) { return 0; }

  virtual int32_t initialize_shard() { return 0; };

  virtual int32_t flush() { return 0; };

  virtual int32_t load(const std::string &path, const std::string &param) {
    return 0;
  }
  virtual int32_t save(const std::string &path, const std::string &param) {
    return 0;
  }

  virtual void clear(){};

  virtual int32_t initialize() override { return 0; };

  virtual int32_t push_dense(const int64_t *values,
                             const int32_t trainer_id) override {
    return 0;
  };

  virtual int32_t set_program_env(
      framework::Scope *scope, platform::Place place,
      const std::vector<framework::ProgramDesc> *sub_program) override;

 protected:
  framework::Executor *executor_;
  framework::Scope *scope_;
  platform::Place place_ = platform::CPUPlace();
  const std::vector<framework::ProgramDesc> *sub_program_;
  paddle::distributed::TensorAccessorParameter program_config_;
  std::shared_ptr<framework::ExecutorPrepareContext> exec_context_ = nullptr;
};

class DenseTensorTable : public TensorTable {
 public:
  DenseTensorTable() {}
  virtual ~DenseTensorTable() {}

  int32_t pull_sparse(float *values,
                      const PullSparseValue &pull_value) override {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values,
                      size_t num) override {
    return 0;
  }
  int32_t shrink(const std::string &param) override { return 0; }

  virtual void *get_shard(size_t shard_idx) { return 0; }

  virtual int32_t initialize_shard() { return 0; }

  virtual int32_t flush() { return 0; }

  virtual void clear() {}

  // Todo: Support program Load & Save
  virtual int32_t load(const std::string &path, const std::string &param) {
    return 0;
  }
  virtual int32_t save(const std::string &path, const std::string &param) {
    return 0;
  }

  // Todo: Support pull dense
  int32_t pull_dense(float *values, size_t num) override { return 0; }

  /*----------------------------------------------------------------------*/

  virtual int32_t initialize() override { return 0; }

  int32_t push_dense(const float *values, size_t num) override { return 0; }

  int32_t push_dense(const int64_t *values, const int32_t trainer_id) {
    return 0;
  }

 protected:
  virtual int32_t _run_program(const float *values, size_t num,
                               const uint32_t trainer_id) {
    return 0;
  }

  int startup_program_id_ = -1;
  int main_program_id_ = -1;
  std::string feed_var_name_ = "";
  std::string fetch_var_name_ = "";
};

class GlobalStepTable : public DenseTensorTable {
 public:
  GlobalStepTable() {}
  virtual ~GlobalStepTable() {}

  int32_t pull_sparse(float *values,
                      const PullSparseValue &pull_value) override {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values,
                      size_t num) override {
    return 0;
  }
  int32_t shrink(const std::string &param) override { return 0; }

  virtual void *get_shard(size_t shard_idx) { return 0; }

  virtual int32_t initialize_shard() { return 0; }

  virtual int32_t flush() { return 0; }

  virtual void clear() {}

  virtual int32_t load(const std::string &path, const std::string &param) {
    return 0;
  }
  virtual int32_t save(const std::string &path, const std::string &param) {
    return 0;
  }

  int32_t pull_dense(float *values, size_t num) override { return 0; }

  /*----------------------------------------------------------------------*/

  int32_t initialize() override;

  int32_t push_dense(const float *values, size_t num) override { return 0; }

  int32_t push_dense(const int64_t *values, const int32_t trainer_id);

  int32_t set_table_map(
      std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map) override;

 private:
  virtual int32_t _run_program(const int64_t *values,
                               const uint32_t trainer_id);

 private:
  std::unordered_map<int, int64_t> decay_counters_;
  int32_t trainers_;
};

}  // namespace distributed
}  // namespace paddle
