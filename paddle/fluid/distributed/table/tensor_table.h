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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ThreadPool.h>
#include "paddle/fluid/distributed/table/table.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

class TensorTable : public Table {
 public:
  TensorTable() : Table() {}

  virtual ~TensorTable() {}

  virtual int32_t initialize() { return 0; }

  virtual int32_t pull_dense(float *values, size_t num) override { return 0; };

  virtual int32_t push_dense(const float *values, size_t num) override {
    return 0;
  };

  virtual void *get_shard(size_t shard_idx) override { return 0; }

  virtual int32_t pull_sparse(float *values, const uint64_t *keys,
                              size_t num) override {
    return 0;
  };

  virtual int32_t push_sparse(const uint64_t *keys, const float *values,
                              size_t num) override {
    return 0;
  };

  virtual int32_t push_dense_param(const float *values, size_t num) {
    return 0;
  }

  virtual int32_t shrink() { return 0; }

  virtual void clear() {}

  virtual int32_t flush() { return 0; }

  //指定加载路径
  virtual int32_t load(const std::string &path, const std::string &converter) {
    return 0;
  }
  //指定保存路径
  virtual int32_t save(const std::string &path, const std::string &converter) {
    return 0;
  }

 protected:
  virtual int32_t initialize_shard() { return 0; }

  virtual int32_t initialize_tensor(paddle::framework::Scope *scope,
                                    paddle::framework::ProgramDesc *program,
                                    paddle::framework::Executor *executor) {
    return 0;
  }

  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;

  framework::Executor *executor_;
  framework::Scope *scope_;
  framework::ProgramDesc *program_;
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      *prepared_ctx_;
};

class DenseTensorTable : public TensorTable {
 public:
  DenseTensorTable() : TensorTable() {}
  ~DenseTensorTable() {}
  virtual int32_t initialize();

  void *get_shard(size_t shard_idx) { return 0; }

  int32_t pull_sparse(float *values, const uint64_t *keys, size_t num) {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values, size_t num) {
    return 0;
  }
  int32_t shrink() { return 0; }

  int32_t pull_dense(float *values, size_t num) override;
  int32_t push_dense_param(const float *values, size_t num) override;
  int32_t push_dense(const float *values, size_t num) override;

  virtual void clear() {}
  virtual int32_t flush() { return 0; }

  //指定加载路径
  virtual int32_t load(const std::string &path, const std::string &converter) {
    return 0;
  }
  //指定保存路径
  virtual int32_t save(const std::string &path, const std::string &converter) {
    return 0;
  }

 protected:
  virtual int32_t initialize_shard() { return 0; }

  virtual int32_t initialize_tensor(paddle::framework::Scope *scope,
                                    paddle::framework::ProgramDesc *program,
                                    paddle::framework::Executor *executor);

 protected:
  framework::Tensor _data;
};
//
//// common sparse table [0, N) with out large scale
// class SparseTensorTable : public TensorTable {
//  void *get_shard(size_t shard_idx) { return 0; }
//
//  int32_t pull_sparse(float *values, const uint64_t *keys, size_t num)
//  override;
//  int32_t push_sparse(const uint64_t *keys, const float *values, size_t num)
//  override ;
//  int32_t shrink() { return 0; }
//  void *get_shard(size_t shard_idx) { return 0; };
//
//  int32_t pull_dense(float *values, size_t num) { return 0; };
//  int32_t push_dense_param(const float *values, size_t num) { return 0; };
//  int32_t push_dense(const float *values, size_t num) { return 0; };
//
// protected:
//  framework::Tensor _data;
//};

//// for Large scale kv tensor  [0, int64] do not use specific optimizer
// class KvTensorTable : public TensorTable {
//  int32_t pull_dense(float *values, size_t num) { return 0; };
//  int32_t push_dense_param(const float *values, size_t num) { return 0; };
//  int32_t push_dense(const float *values, size_t num) { return 0; };
//
//  void *get_shard(size_t shard_idx) override;
//  int32_t pull_sparse(float *values, const uint64_t *keys, size_t num)
//  override;
//  int32_t push_sparse(const uint64_t *keys, const float *values,
//                      size_t num) override;
//  int32_t shrink() override;
//  void *get_shard(size_t shard_idx) override;
//};
//
//// for Geo sparse handle
// class GeoSparseTensorTable : public TensorTable {};
}  // namespace distributed
}  // namespace paddle
