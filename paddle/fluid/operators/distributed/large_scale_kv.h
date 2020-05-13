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
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {
namespace distributed {

enum Mode { training, infer };

struct SparseMeta {
  std::string name;
  std::string grad_name;
  std::vector<std::string> value_names;
  std::vector<int> value_dims;
  std::vector<std::string> cached_varnames;
  Mode mode;

  std::string ToString() {
    std::stringstream ss;
    ss << "name: " << name << " ";
    ss << "mode: " << mode << " ";

    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      ss << "value_name: " << value_names[i] << " dim: " << value_dims[i]
         << " ";
    }

    ss << " grad var: " << grad_name;
    ss << " cached varnames: ";
    for (int i = 0; i < static_cast<int>(cached_varnames.size()); i++) {
      ss << cached_varnames[i] << " ";
    }
    return ss.str();
  }
};

struct VALUE {
  explicit VALUE(const std::vector<std::string> &names) : names_(names) {
    values_.resize(names.size());
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      places[names[i]] = i;
    }
  }

  void set(std::vector<std::vector<float>> &&values) {
    values_ = std::move(values);
  }

  void set(const std::vector<std::string> &names,
           const std::vector<std::vector<float>> &values) {
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
  explicit SparseVariable(const SparseMeta &meta) {
    meta_.name = meta.name;
    meta_.mode = meta.mode;
    meta_.value_names = meta.value_names;
    meta_.value_dims = meta.value_dims;
    meta_.grad_name = meta.grad_name;
    meta_.cached_varnames = meta.cached_varnames;

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      values_dims[meta_.value_names[i]] = meta_.value_dims[i];
    }
  }

  void Get(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           std::vector<std::vector<std::vector<float>>> *values) {
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

  void Set(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           const std::vector<std::vector<std::vector<float>>> &values) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < static_cast<int>(ids.size()); i++) {
      auto value = values_.at(ids[i]);
      auto values_for_id = values[i];
      value->set(value_names, values_for_id);
    }
  }

  void Get(const framework::Tensor &ids,
           const std::vector<std::string> value_names,
           std::vector<framework::Tensor> *values) {}

  void Dims(std::vector<std::string> value_names, std::vector<int64_t> *dims) {
    for (auto &name : value_names) {
      dims->push_back(values_dims.at(name));
    }
  }

  std::vector<std::string> CachedVarnames() const {
    return meta_.cached_varnames;
  }

  void Save(const std::string &dirname) {
    VLOG(1) << "save " << meta_.name << " in dir: " << dirname << " begin";

    MkDirRecursively(dirname.c_str());

    std::vector<std::string> filenames;
    for (auto &value_name : meta_.value_names) {
      auto filename = string::Sprintf("%s/%s", dirname, value_name);
      filenames.push_back(filename);
    }

    Save(filenames, meta_.value_names);

    VLOG(1) << "save " << meta_.name << " in dir: " << dirname << " done";
  }

  void Save(const std::vector<std::string> &filenames,
            const std::vector<std::string> &valuenames) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &value_name : valuenames) {
      auto it = std::find(meta_.value_names.begin(), meta_.value_names.end(),
                          value_name);
      if (it == meta_.value_names.end()) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "[%s] is invalid param for [%s]", value_name, meta_.name));
      }
    }

    std::vector<std::unique_ptr<std::ofstream>> fouts;

    for (auto filename : filenames) {
      std::unique_ptr<std::ofstream> fout(new std::ofstream(filename));
      fouts.push_back(std::move(fout));
    }

    for (auto value : values_) {
      std::vector<std::vector<float>> vss = value.second->get(valuenames);

      auto id = value.first;

      for (int i = 0; i < static_cast<int>(vss.size()); i++) {
        auto &vs = vss[i];

        std::stringstream ss;
        ss << id << "\t";

        ss << vs.size() << "\t";

        for (auto v : vs) {
          ss << v << " ";
        }

        ss << "\n";

        fouts[i]->write(ss.str().c_str(), sizeof(char) * ss.str().size());
      }
    }

    for (int i = 0; i < static_cast<int>(fouts.size()); i++) {
      fouts[i]->close();
    }
  }

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
  std::unordered_map<int64_t, VALUE *> values_;
};

class LargeScaleKV {
 public:
  LargeScaleKV() {}

  explicit LargeScaleKV(const std::vector<SparseMeta> &table_metas) {
    for (auto &sparse_meta : table_metas) {
      auto table_name = sparse_meta.name;
      auto meta = std::shared_ptr<SparseVariable>(
          new SparseVariable(std::move(sparse_meta)));
      sparse_variables[table_name] = meta;
      grad_to_variables[sparse_meta.grad_name] = table_name;
      grad_names_.push_back(sparse_meta.grad_name);
    }
  }

  ~LargeScaleKV() {}

  static LargeScaleKV *GetInstance() { return scale_kv_.get(); }

  static LargeScaleKV *InitInstance(
      const std::vector<SparseMeta> &table_metas) {
    std::call_once(init_flag_, &LargeScaleKV::Init, table_metas);
    return scale_kv_.get();
  }

  static void Init(const std::vector<SparseMeta> &table_metas) {
    if (scale_kv_.get() == nullptr) {
      scale_kv_.reset(new LargeScaleKV(table_metas));
    }
  }

  SparseVariable *Get(std::string name) {
    auto variable = sparse_variables.at(name);
    return variable.get();
  }

  SparseVariable *GetByGrad(std::string name) {
    return Get(grad_to_variables[name]);
  }

  const std::vector<std::string> &GetAllGrads() { return grad_names_; }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
  std::unordered_map<std::string, std::string> grad_to_variables;
  std::vector<std::string> grad_names_;
  static std::shared_ptr<LargeScaleKV> scale_kv_;
  static std::once_flag init_flag_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
