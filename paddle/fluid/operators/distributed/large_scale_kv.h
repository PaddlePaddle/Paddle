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
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {
namespace distributed {

enum Mode { training, infer };
enum InitType { uniform_random, fill_constant, gaussian_random };

inline std::vector<int> bucket(const int v_size, const int b_size) {
  int remainder = v_size % b_size;
  int bucket = v_size / b_size;
  std::vector<int> ret_vec(b_size, bucket);
  for (int i = 0; i < remainder; ++i) {
    ret_vec[i] = ret_vec[i] + 1;
  }
  int cur_bucket = 0;
  for (int &j : ret_vec) {
    int tmp = j;
    j = cur_bucket;
    cur_bucket += tmp;
  }
  ret_vec.push_back(cur_bucket);
  return ret_vec;
}

class Initializer {
 public:
  Initializer() {}

  explicit Initializer(const std::vector<std::string> &attrs) {}

  virtual float GetValue() = 0;

  virtual ~Initializer() {}

 protected:
  std::string name_;
  unsigned int seed_;
};

class UniformInitializer : public Initializer {
 public:
  explicit UniformInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    min_ = std::stof(attrs[2]);
    max_ = std::stof(attrs[3]);

    if (seed_ == 0) {
      seed_ = std::random_device()();
    }

    random_engine_.seed(seed_);
    dist_ = std::uniform_real_distribution<float>(min_, max_);
  }

  float GetValue() override { return dist_(random_engine_); }

 private:
  float min_;
  float max_;

  std::minstd_rand random_engine_;
  std::uniform_real_distribution<float> dist_;
};

class GaussianInitializer : public Initializer {
 public:
  explicit GaussianInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    mean_ = std::stof(attrs[2]);
    std_ = std::stof(attrs[3]);

    if (seed_ == 0) {
      seed_ = std::random_device()();
    }

    random_engine_->seed(seed_);
    dist_ = std::make_shared<std::normal_distribution<float>>(mean_, std_);
  }

  float GetValue() override { return (*dist_)(*random_engine_); }

 private:
  float std_;
  float mean_;
  std::shared_ptr<std::minstd_rand> random_engine_;
  std::shared_ptr<std::normal_distribution<float>> dist_;
};

class FillConstantInitializer : public Initializer {
 public:
  explicit FillConstantInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    value_ = std::stof(attrs[1]);
  }

  float GetValue() override { return value_; }

 private:
  float value_;
};

struct SparseMeta {
  std::string name;
  std::string grad_name;
  std::vector<std::string> value_names;
  std::vector<int> value_dims;
  std::vector<std::string> cached_varnames;
  std::vector<std::string> initializer_attrs;
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

    ss << " initializer attrs";
    for (int i = 0; i < static_cast<int>(initializer_attrs.size()); i++) {
      ss << initializer_attrs[i] << " ";
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

  void set(std::vector<std::vector<float>> *values) {
    values_ = std::move(*values);
  }

  void set(const std::vector<std::string> &names,
           const std::vector<std::vector<float>> &values) {
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      auto idx = places[names[i]];
      auto value = values[i];
      values_[idx].assign(value.begin(), value.end());
    }
  }

  std::vector<std::vector<float> *> get() {
    auto pts = std::vector<std::vector<float> *>();
    pts.reserve(values_.size());

    for (auto &value : values_) {
      pts.push_back(&value);
    }
    return pts;
  }

  std::vector<std::vector<float> *> get(const std::vector<std::string> names) {
    auto pts = std::vector<std::vector<float> *>();
    pts.reserve(values_.size());

    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      pts.push_back(&(values_[places[names[i]]]));
    }
    return pts;
  }

  std::vector<std::string> names_;
  std::vector<std::vector<float>> values_;
  std::unordered_map<std::string, int> places;
};

class ValueBlock {
 public:
  explicit ValueBlock(const std::vector<std::string> value_names,
                      const std::vector<int> value_dims, const Mode &mode,
                      const std::vector<std::string> &attrs)
      : value_names_(value_names), value_dims_(value_dims), mode_(mode) {
    for (size_t i = 0; i < value_names.size(); i++) {
      auto name = value_names[i];
      auto slices = string::split_string<std::string>(attrs[i], "&");

      if (slices[0] == "gaussian_random") {
        initializers_[name] = new GaussianInitializer(slices);
      } else if (slices[0] == "fill_constant") {
        initializers_[name] = new FillConstantInitializer(slices);
      } else if (slices[0] == "uniform_random") {
        initializers_[name] = new UniformInitializer(slices);
      } else {
        PADDLE_THROW(
            platform::errors::InvalidArgument("%s can not be supported", name));
      }
    }

    rwlock_.reset(new framework::RWLock);
  }

  ~ValueBlock() {
    //    for (auto init : initializers_) {
    //      delete init.second;
    //      initializers_.erase(init.first);
    //    }
    //
    //    for (auto value : values_) {
    //      delete value.second;
    //      values_.erase(value.first);
    //    }
  }

  void Init(const int64_t &id) {
    //    if (Has(id)) {
    //      return;
    //    }

    rwlock_->WRLock();

    if (Has(id)) {
      rwlock_->UNLock();
      return;
    }

    auto rets = std::vector<std::vector<float>>();
    rets.resize(value_names_.size());

    for (int i = 0; i < static_cast<int>(value_names_.size()); i++) {
      auto name = value_names_[i];
      auto *init = initializers_.at(name);

      auto dim = value_dims_[i];
      rets[i].resize(dim);

      for (int j = 0; j < static_cast<int>(dim); j++) {
        rets[i][j] = init->GetValue();
      }
    }

    auto value = new VALUE(value_names_);
    value->set(rets);
    values_[id] = value;

    rwlock_->UNLock();
  }

  std::vector<std::vector<float> *> Get(
      const int64_t &id, const std::vector<std::string> &value_names) {
    rwlock_->RDLock();
    auto ret_values = values_.at(id)->get(value_names);
    rwlock_->UNLock();
    return ret_values;
  }

  std::vector<std::vector<float> *> GetAndInit(
      const int64_t &id, const std::vector<std::string> &value_names) {
    Init(id);
    return Get(id, value_names);
  }

  void Set(const int64_t &id, const std::vector<std::string> &value_names,
           const std::vector<std::vector<float>> &values) {
    rwlock_->WRLock();
    auto value = values_.at(id);
    value->set(value_names, values);
    rwlock_->UNLock();
  }

 private:
  bool Has(const int64_t id) {
    auto got = values_.find(id);
    if (got == values_.end()) {
      return false;
    } else {
      return true;
    }
  }

 public:
  std::unordered_map<int64_t, VALUE *> values_;

 private:
  std::vector<std::string> value_names_;
  std::vector<int> value_dims_;
  Mode mode_;
  std::unordered_map<std::string, Initializer *> initializers_;
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};
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
    meta_.initializer_attrs = meta.initializer_attrs;

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      values_dims_[meta_.value_names[i]] = meta_.value_dims[i];
    }

    for (size_t i = 0; i < shard_num_; i++) {
      auto block = std::make_shared<ValueBlock>(
          meta.value_names, meta.value_dims, meta.mode, meta.initializer_attrs);
      shard_blocks_.emplace_back(block);
    }
  }

  void GetAndInit(const std::vector<int64_t> &ids,
                  const std::vector<std::string> &value_names,
                  std::vector<std::vector<std::vector<float> *>> *values) {
    for (auto &id : ids) {
      std::vector<std::vector<float> *> id_values;
      auto *block = GetShard(id);
      id_values = block->GetAndInit(id, value_names);
      values->push_back(id_values);
    }
  }

  void Get(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           std::vector<std::vector<std::vector<float> *>> *values) {
    values->resize(ids.size());

    auto buckets = bucket(ids.size(), 8);
    std::vector<std::future<void>> fs;

    for (int j = 0; j < 8; ++j) {
      auto begin = buckets[j];
      auto end = buckets[j + 1];

      fs.push_back(
          framework::Async([begin, end, &values, &ids, &value_names, this]() {
            for (int x = begin; x < end; x++) {
              auto id = ids[x];
              auto *block = GetShard(id);
              auto id_values = block->Get(id, value_names);
              (*values)[x] = id_values;
            }
          }));
    }

    for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
  }

  void Set(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           const std::vector<std::vector<std::vector<float>>> &values) {
    for (int i = 0; i < static_cast<int>(ids.size()); i++) {
      GetShard(ids[i])->Set(ids[i], value_names, values[i]);
    }
  }

  void Dims(std::vector<std::string> value_names, std::vector<int64_t> *dims) {
    for (auto &name : value_names) {
      dims->push_back(values_dims_.at(name));
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
    SaveToSelectedRows(filenames, meta_.value_names);

    // save sparse to text
    //    std::vector<std::string> txt_filenames;
    //    for (auto &value_name : meta_.value_names) {
    //      auto filename = string::Sprintf("%s/%s.txt", dirname, value_name);
    //      txt_filenames.push_back(filename);
    //    }
    //    SaveToText(txt_filenames, meta_.value_names);

    VLOG(1) << "save " << meta_.name << " in dir: " << dirname << " done";
  }

  void SaveToSelectedRows(const std::vector<std::string> &filenames,
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

    auto place = platform::CPUPlace();
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    int64_t ids_num = 0;
    for (auto &block : shard_blocks_) {
      ids_num += block->values_.size();
    }

    std::vector<std::shared_ptr<framework::Variable>> variables;
    std::vector<float *> tensors;
    std::vector<int64_t> ids;
    std::vector<int64_t> dims;

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto dim = values_dims_.at(valuenames[i]);
      auto var = std::make_shared<framework::Variable>();
      auto *slr = var->GetMutable<framework::SelectedRows>();
      auto *src_t = slr->mutable_value();

      src_t->Resize({ids_num, dim});
      auto *value = src_t->mutable_data<float>(place);

      dims.push_back(dim);
      variables.push_back(var);
      tensors.push_back(value);
    }

    int64_t offset = 0;
    for (auto &block : shard_blocks_) {
      for (auto value : block->values_) {
        ids.push_back(value.first);
        std::vector<std::vector<float> *> vss = value.second->get(valuenames);

        for (int i = 0; i < static_cast<int>(vss.size()); i++) {
          auto &vs = vss[i];
          std::memcpy(tensors[i] + offset * dims[i], vs->data(),
                      sizeof(float) * dims[i]);
        }

        offset += 1;
      }
    }

    for (auto &var : variables) {
      auto *slr = var->GetMutable<framework::SelectedRows>();
      slr->set_rows(ids);
      slr->set_height(ids.size());
    }

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto &filename = filenames[i];

      auto &selectedRows = variables[i]->Get<framework::SelectedRows>();

      std::ofstream fout(filename, std::ios::binary);
      PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                        platform::errors::Unavailable(
                            "Cannot open %s to save variables.", filename));

      framework::SerializeToStream(fout, selectedRows, dev_ctx);
      fout.close();
    }
  }

  void SaveToText(const std::vector<std::string> &filenames,
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

    for (auto &block : shard_blocks_) {
      for (auto value : block->values_) {
        std::vector<std::vector<float> *> vss = value.second->get(valuenames);

        auto id = value.first;

        for (int i = 0; i < static_cast<int>(vss.size()); i++) {
          auto &vs = vss[i];
          std::stringstream ss;
          ss << id << "\t";
          ss << vs->size() << "\t";
          for (auto v : (*vs)) {
            ss << v << " ";
          }
          ss << "\n";

          fouts[i]->write(ss.str().c_str(), sizeof(char) * ss.str().size());
        }
      }
    }

    for (int i = 0; i < static_cast<int>(fouts.size()); i++) {
      fouts[i]->close();
    }
  }

  int64_t Size() {
    int64_t cnt = 0;

    for (auto &block : shard_blocks_) {
      cnt += block->values_.size();
    }
    return cnt;
  }

  ValueBlock *GetShard(const int64_t id) {
    return shard_blocks_[id & shard_mask_].get();
  }

 private:
  mutable std::mutex mutex_;

  SparseMeta meta_;
  std::unordered_map<std::string, int64_t> values_dims_;
  const size_t shard_mask_ = 127;
  const size_t shard_num_ = 128;
  std::vector<std::shared_ptr<ValueBlock>> shard_blocks_;
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
