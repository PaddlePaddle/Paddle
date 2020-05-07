// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum Mode { training, infer };

struct SparseMeta {
  std::string name;
  std::vector<std::string> value_names;
  std::vector<int> value_dims;
  Mode mode;
};

struct VALUE {
  VALUE(const std::vector<std::string>& names, const std::vector<int>& dims)
      : names_(names), dims_(dims) {
    values_.resize(names.size());
  }

  void init() {
    for (int i = 0; i < static_cast<int>(names_.size()); i++) {
      values_[i].resize(dims_[i]);
      std::fill(values_[i].data(), values_[i].data() + dims_[i],
                static_cast<float>(0.0));
    }
  }

  void set() {}

  std::vector<std::vector<float>> get() { return values_; }

  std::vector<std::vector<float>> get(const std::vector<std::string> names) {
    return values_;
  }

  std::vector<std::string> names_;
  std::vector<std::vector<float>> values_;
  std::vector<int> dims_;
  std::vector<int> initializers_;
};

class SparseVariable {
 public:
  explicit SparseVariable(const SparseMeta& meta) {
    meta_.name = meta.name;
    meta_.mode = meta.mode;
    meta_.value_names = meta.value_names;
    meta_.value_dims = meta.value_dims;

    for (int i = 0; i < meta_.value_names.size(); i++) {
      values_dims[meta_.value_names[i]] = meta_.value_dims[i];
    }
  }

  void Print() {
    std::stringstream ss;
    ss << "name: " << meta_.name << " ";
    ss << "mode: " << meta_.mode << " ";

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      ss << "value_name: " << meta_.value_names[i]
         << " dim: " << meta_.value_dims[i] << " ";
    }
    VLOG(1) << ss.str();
  }

  void Get(const std::vector<int64_t>& ids,
           const std::vector<std::string>& value_names,
           std::vector<std::vector<std::vector<float>>>* values) {
    for (auto id : ids) {
      auto got = values_.find(id);
      if (got == values_.end()) {
        auto value = new VALUE(meta_.value_names, meta_.value_dims);
        value->init();
        values_[id] = value;
      }
      auto value = values_.at(id);
      values->push_back(value->get(value_names));
    }
  }

  void Get(const framework::Tensor& ids,
           const std::vector<std::string> value_names,
           std::vector<framework::Tensor>* values) {}

  void Dims(std::vector<std::string> value_names, std::vector<int64_t>* dims) {
    for (auto& name : value_names) {
      dims->push_back(values_dims.at(name));
    }
  }

  int64_t Size() { return static_cast<int64_t>(values_.size()); }

 public:
  SparseMeta meta_;
  std::unordered_map<std::string, int64_t> values_dims;
  std::unordered_map<int64_t, VALUE*> values_;
};

class LargeScaleKV {
 public:
  LargeScaleKV() {}

  explicit LargeScaleKV(const std::vector<SparseMeta>& table_metas) {
    for (auto& sparse_meta : table_metas) {
      auto table_name = sparse_meta.name;
      auto meta = std::shared_ptr<SparseVariable>(
          new SparseVariable(std::move(sparse_meta)));
      sparse_variables[table_name] = meta;
    }
  }

  ~LargeScaleKV() {}

  static LargeScaleKV* GetInstance() { return scale_kv_.get(); }

  static LargeScaleKV* InitInstance(
      const std::vector<SparseMeta>& table_metas) {
    std::call_once(init_flag_, &LargeScaleKV::Init, table_metas);
    return scale_kv_.get();
  }

  static void Init(const std::vector<SparseMeta>& table_metas) {
    if (scale_kv_.get() == nullptr) {
      scale_kv_.reset(new LargeScaleKV(table_metas));
    }
  }

  SparseVariable* Get(std::string name) {
    auto variable = sparse_variables.at(name);
    return variable.get();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
  static std::shared_ptr<LargeScaleKV> scale_kv_;
  static std::once_flag init_flag_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
