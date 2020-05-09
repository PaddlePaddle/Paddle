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
  std::vector<std::string> grad_names;
  Mode mode;
};

struct VALUE {
  explicit VALUE(const std::vector<std::string>& names) : names_(names) {
    values_.resize(names.size());
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      places[names[i]] = i;
    }
  }

  void set(std::vector<std::vector<float>>&& values) {
    values_ = std::move(values);
  }

  void set(const std::vector<std::string>& names,
           const std::vector<std::vector<float>>& values) {
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      auto idx = places[names[i]];
      auto value = values[i];
      values_[idx].assign(value.begin(), value.end());
    }
  }

  std::vector<std::vector<float>> get() { return values_; }

  std::vector<std::vector<float>> get(const std::vector<std::string> names) {
    auto rets = std::vector<std::vector<float>>();

    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      rets.push_back(values_[places[names[i]]]);
    }
    return rets;
  }

  std::vector<std::string> names_;
  std::vector<std::vector<float>> values_;
  std::unordered_map<std::string, int> places;
};

class SparseVariable {
 public:
  explicit SparseVariable(const SparseMeta& meta) {
    meta_.name = meta.name;
    meta_.mode = meta.mode;
    meta_.value_names = meta.value_names;
    meta_.value_dims = meta.value_dims;
    meta_.grad_names = meta.grad_names;

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      values_dims[meta_.value_names[i]] = meta_.value_dims[i];
    }
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "name: " << meta_.name << " ";
    ss << "mode: " << meta_.mode << " ";

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      ss << "value_name: " << meta_.value_names[i]
         << " dim: " << meta_.value_dims[i] << " ";
    }

    ss << " grads: ";
    for (int i = 0; i < static_cast<int>(meta_.grad_names.size()); i++) {
      ss << meta_.grad_names[i] << " ";
    }
    return ss.str();
  }

  void Get(const std::vector<int64_t>& ids,
           const std::vector<std::string>& value_names,
           std::vector<std::vector<std::vector<float>>>* values) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto id : ids) {
      auto got = values_.find(id);
      if (got == values_.end()) {
        auto value = new VALUE(meta_.value_names);
        value->set(Init());
        values_[id] = value;
      }
      auto value = values_.at(id);
      values->push_back(value->get(value_names));
    }
  }
  void Set(const std::vector<int64_t>& ids,
           const std::vector<std::string>& value_names,
           const std::vector<std::vector<std::vector<float>>>& values) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < static_cast<int>(ids.size()); i++) {
      auto value = values_.at(ids[i]);
      auto values_for_id = values[i];
      value->set(value_names, values_for_id);
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

  std::vector<std::string>& ValueNames() const { return meta_.value_names; }

  int64_t Size() { return static_cast<int64_t>(values_.size()); }

 private:
  std::vector<std::vector<float>> Init() {
    auto rets = std::vector<std::vector<float>>();
    rets.resize(meta_.value_names.size());

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      auto name = meta_.value_names[i];
      auto dim = meta_.value_dims[i];
      rets[i].resize(dim);
      std::fill(rets[i].data(), rets[i].data() + dim, static_cast<float>(0.0));
    }
    return rets;
  }

  mutable std::mutex mutex_;
  SparseMeta meta_;
  std::unordered_map<std::string, int64_t> values_dims;
  std::vector<int> initializers_;
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

      VLOG(3) << "Init LargeScaleKV with: " << meta->ToString();

      for (auto& grad : sparse_meta.grad_names) {
        grad_to_param[grad] = table_name;
      }
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

  SparseVariable* GetByGrad(std::string name) {
    return sparse_variables[grad_to_param[name]];
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
  std::unordered_map<std::string, std::string> grad_to_param;
  static std::shared_ptr<LargeScaleKV> scale_kv_;
  static std::once_flag init_flag_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
