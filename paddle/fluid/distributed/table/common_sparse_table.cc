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

#include "paddle/fluid/distributed/table/common_sparse_table.h"
#include <algorithm>
#include <sstream>
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

struct Meta {
  std::string param;
  int shard_id;
  std::vector<std::string> names;
  std::vector<int> dims;
  uint64_t count;
  std::unordered_map<std::string, int> dims_map;

  explicit Meta(const std::string& metapath) {
    std::ifstream file(metapath);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      if (StartWith(line, "#")) {
        continue;
      }
      auto pairs = paddle::string::split_string<std::string>(line, "=");
      PADDLE_ENFORCE_EQ(
          pairs.size(), 2,
          paddle::platform::errors::InvalidArgument(
              "info in %s except k=v, but got %s", metapath, line));

      if (pairs[0] == "param") {
        param = pairs[1];
      }
      if (pairs[0] == "shard_id") {
        shard_id = std::stoi(pairs[1]);
      }
      if (pairs[0] == "row_names") {
        names = paddle::string::split_string<std::string>(pairs[1], ",");
      }
      if (pairs[0] == "row_dims") {
        auto dims_strs =
            paddle::string::split_string<std::string>(pairs[1], ",");
        for (auto& str : dims_strs) {
          dims.push_back(std::stoi(str));
        }
      }
      if (pairs[0] == "count") {
        count = std::stoull(pairs[1]);
      }
    }
    for (int x = 0; x < names.size(); ++x) {
      dims_map[names[x]] = dims[x];
    }
  }

  Meta(std::string param, int shard_id, std::vector<std::string> row_names,
       std::vector<int> dims, uint64_t count) {
    this->param = param;
    this->shard_id = shard_id;
    this->names = row_names;
    this->dims = dims;
    this->count = count;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "param=" << param << "\n";
    ss << "shard_id=" << shard_id << "\n";
    ss << "row_names=" << paddle::string::join_strings(names, ',') << "\n";
    ss << "row_dims=" << paddle::string::join_strings(dims, ',') << "\n";
    ss << "count=" << count << "\n";
    return ss.str();
  }
};

void ProcessALine(const std::vector<std::string>& columns, const Meta& meta,
                  std::vector<std::vector<float>>* values) {
  PADDLE_ENFORCE_EQ(columns.size(), meta.names.size() + 1,
                    paddle::platform::errors::InvalidArgument(
                        "record in txt do not match meta."));

  values->reserve(columns.size() - 1);

  for (int x = 1; x < columns.size(); ++x) {
    auto& column = columns[x];
    auto val_ = paddle::string::split_string<std::string>(column, ",");

    std::vector<float> val;
    std::transform(val_.begin(), val_.end(), std::back_inserter(val),
                   [](std::string va) { return std::stof(va); });
    PADDLE_ENFORCE_EQ(val.size(), meta.dims[x - 1],
                      paddle::platform::errors::InvalidArgument(
                          "record in txt do not match meta."));
    values->push_back(val);
  }
}

int64_t SaveToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
                   const std::vector<std::string>& saved_names,
                   const int mode) {
  for (auto value : block->values_) {
    std::vector<std::vector<float>*> vss = value.second->get(saved_names);
    std::stringstream ss;
    auto id = value.first;
    ss << id << "\t";
    for (int i = 0; i < static_cast<int>(vss.size()); i++) {
      auto& vs = vss[i];
      ss << paddle::string::join_strings((*vs), ',');
      ss << "\t";
    }
    ss << "\n";

    os->write(ss.str().c_str(), sizeof(char) * ss.str().size());
  }

  return block->values_.size();
}

int64_t LoadFromText(const std::string& valuepath, const std::string& metapath,
                     const int pserver_id, const int pserver_num,
                     const int local_shard_num,
                     std::vector<std::shared_ptr<ValueBlock>>* blocks) {
  Meta meta = Meta(metapath);

  int num_lines = 0;
  std::ifstream file(valuepath);
  std::string line;

  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    auto id = std::stoull(values[0]);

    if (id % pserver_num != pserver_id) {
      VLOG(0) << "will not load " << values[0] << " from " << valuepath
              << ", please check id distribution";
      continue;
    }

    auto shard_id = id % local_shard_num;
    auto block = blocks->at(shard_id);

    std::vector<std::vector<float>> kvalues;
    ProcessALine(values, meta, &kvalues);
    block->Init(id, &kvalues, 1);
  }

  return 0;
}

void SaveShard(std::shared_ptr<ValueBlock> block, const std::string& dirname,
               const CommonAccessorParameter& common, const int mode,
               const int pserver_id, const int shard_id) {
  auto varname = common.table_name();
  std::string var_store = string::Sprintf("%s/%s", dirname, varname);
  VLOG(3) << "save " << varname << " in dir: " << var_store << " begin";
  MkDirRecursively(var_store.c_str());

  std::string shard_var_pre =
      string::Sprintf("%s.block%d.%d", varname, pserver_id, shard_id);
  std::string meta_ = string::Sprintf("%s/%s.meta", var_store, shard_var_pre);
  std::string value_ = string::Sprintf("%s/%s.txt", var_store, shard_var_pre);

  // save values
  std::vector<std::string> params(common.params().begin(),
                                  common.params().end());
  std::unique_ptr<std::ofstream> value_out(new std::ofstream(value_));
  SaveToText(value_out.get(), block, params, mode);
  // save meta
  std::stringstream stream;
  stream << "param=" << common.table_name() << "\n";
  stream << "server_id=" << pserver_id << "\n";
  stream << "shard_id=" << shard_id << "\n";
  stream << "row_names=" << paddle::string::join_strings(common.params(), ',')
         << "\n";
  stream << "row_dims=" << paddle::string::join_strings(common.dims(), ',')
         << "\n";
  stream << "count=" << block->values_.size() << "\n";
  std::unique_ptr<std::ofstream> meta_out(new std::ofstream(meta_));
  meta_out->write(stream.str().c_str(), sizeof(char) * stream.str().size());
  meta_out->close();
  VLOG(3) << "save " << varname << " in dir: " << var_store << " done";
}

void CommonSparseTable::create_initializer(const std::string& attr,
                                           const std::string& name) {
  auto slices = string::split_string<std::string>(attr, "&");

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

int32_t CommonSparseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  sync = _config.common().sync();
  VLOG(1) << "table " << _config.common().table_name() << " is sync: " << sync;

  initialize_value();
  initialize_optimizer();
  initialize_recorder();
  return 0;
}

int32_t CommonSparseTable::initialize_recorder() { return 0; }

int32_t CommonSparseTable::initialize_value() {
  auto common = _config.common();
  int size = static_cast<int>(common.params().size());

  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];
    if (varname == "Param") {
      param_dim_ = dim;
    }
    auto& initializer = common.initializers()[x];
    create_initializer(initializer, varname);
  }

  shard_values_.reserve(task_pool_size_);
  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard = std::make_shared<ValueBlock>(common, &initializers_);
    shard_values_.emplace_back(shard);
  }
  return 0;
}

int32_t CommonSparseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();
  auto attrs = common.attributes();

  if (name == "sgd") {
    optimizer_ = std::make_shared<SSGD>(common);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<SAdam>(common);
  } else if (name == "sum") {
    optimizer_ = std::make_shared<SSUM>(common);
  } else {
    VLOG(0) << "init optimizer failed";
  }

  VLOG(0) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonSparseTable::load(const std::string& path,
                                const std::string& param) {
  rwlock_->WRLock();
  VLOG(0) << "sparse table load with " << path << " with meta " << param;
  LoadFromText(path, param, _shard_idx, _shard_num, task_pool_size_,
               &shard_values_);
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::save(const std::string& dirname,
                                const std::string& param) {
  rwlock_->WRLock();
  int mode = std::stoi(param);
  VLOG(0) << "sparse table save: " << dirname << " mode: " << mode;

  auto varname = _config.common().table_name();
  std::string var_store = string::Sprintf("%s/%s", dirname, varname);
  MkDirRecursively(var_store.c_str());

  VLOG(3) << "save " << varname << " in dir: " << var_store << " begin";
  std::vector<std::string> params(_config.common().params().begin(),
                                  _config.common().params().end());
  std::string shard_var_pre =
      string::Sprintf("%s.block%d", varname, _shard_idx);

  std::string value_ = string::Sprintf("%s/%s.txt", var_store, shard_var_pre);

  std::unique_ptr<std::ofstream> value_out(new std::ofstream(value_));

  int64_t total_ins = 0;
  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // save values
    total_ins +=
        SaveToText(value_out.get(), shard_values_[shard_id], params, mode);
  }
  value_out->close();

  // save meta
  std::stringstream stream;
  stream << "param=" << _config.common().table_name() << "\n";
  stream << "shard_id=" << _shard_idx << "\n";
  stream << "row_names="
         << paddle::string::join_strings(_config.common().params(), ',')
         << "\n";
  stream << "row_dims="
         << paddle::string::join_strings(_config.common().dims(), ',') << "\n";
  stream << "count=" << total_ins << "\n";
  std::string meta_ = string::Sprintf("%s/%s.meta", var_store, shard_var_pre);
  std::unique_ptr<std::ofstream> meta_out(new std::ofstream(meta_));
  meta_out->write(stream.str().c_str(), sizeof(char) * stream.str().size());
  meta_out->close();
  VLOG(3) << "save " << varname << " in dir: " << var_store << " done";
  rwlock_->UNLock();
  return 0;
}

std::pair<int64_t, int64_t> CommonSparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& value : shard_values_) {
    feasign_size += value->values_.size();
  }

  return {feasign_size, mf_size};
}

int32_t CommonSparseTable::pour() {
  rwlock_->RDLock();

  std::vector<float> values;
  std::vector<uint64_t> keys;

  keys.reserve(pull_reservoir_.size());
  values.reserve(pull_reservoir_.size() * param_dim_);

  for (auto& val : pull_reservoir_) {
    keys.push_back(val.first);
    auto& reservoir = val.second;
    reservoir.avg();
    std::copy(reservoir.values.begin(), reservoir.values.end(),
              std::back_inserter(values));
  }
  _push_sparse(keys.data(), values.data(), pull_reservoir_.size());

  pull_reservoir_.clear();
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::pull_sparse(float* pull_values, const uint64_t* keys,
                                       size_t num) {
  rwlock_->RDLock();
  std::vector<std::string> value_names;
  for (auto name : _config.common().params()) {
    value_names.push_back(name);
  }

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &value_names,
         &pull_values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            block->InitFromInitializer(id, value_names);
            auto values = block->Get(id, {"Param"});
            auto dim = values[0]->size();
            std::copy(values[0]->begin(), values[0]->end(),
                      pull_values + dim * offset);
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::_push_sparse(const uint64_t* keys,
                                        const float* values, size_t num) {
  rwlock_->RDLock();
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &values, num, &offset_bucket]() -> int {
          auto& offsets = offset_bucket[shard_id];
          optimizer_->update(keys, values, num, offsets,
                             shard_values_[shard_id].get());
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::push_sparse(const uint64_t* keys,
                                       const float* values, size_t num) {
  if (sync) {
    std::future<int> task =
        _shards_task_pool[0]->enqueue([this, &keys, &values, num]() -> int {
          for (int x = 0; x < num; ++x) {
            auto id = keys[x];
            auto has = pull_reservoir_.find(id);

            if (has == pull_reservoir_.end()) {
              pull_reservoir_[id] = ReservoirValue<float>(param_dim_);
            }

            auto& reservoir = pull_reservoir_[id];
            reservoir.add(values + x * param_dim_, param_dim_);
          }
          return 0;
        });
    task.wait();
  } else {
    _push_sparse(keys, values, num);
  }

  return 0;
}

int32_t CommonSparseTable::push_sparse_param(const uint64_t* keys,
                                             const float* values, size_t num) {
  rwlock_->RDLock();
  std::vector<std::string> value_names;
  for (auto name : _config.common().params()) {
    value_names.push_back(name);
  }

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &value_names,
         &values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            block->InitFromInitializer(id, value_names);
            auto values_ = block->Get(id, {"Param"});
            auto dim = values_[0]->size();
            std::copy_n(values + dim * offset, dim, values_[0]->data());
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::flush() { return 0; }

int32_t CommonSparseTable::shrink() {
  VLOG(0) << "shrink coming soon";
  return 0;
}
void CommonSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
