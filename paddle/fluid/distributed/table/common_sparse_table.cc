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

#include <sstream>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
class ValueBlock;
}  // namespace distributed
}  // namespace paddle

#define PSERVER_SAVE_SUFFIX "_txt"

namespace paddle {
namespace distributed {

void nondecay_func(const uint64_t* keys,
                  float* update_values,
                  const std::vector<uint64_t>& offsets,
                  ValueBlock* block,
                  const float coeff,
                  const int param_offset,
                  const int param_dim) {
  return;
}

void l1decay_func(const uint64_t* keys,
                  float* update_values,
                  const std::vector<uint64_t>& offsets,
                  ValueBlock* block,
                  const float coeff,
                  const int param_offset,
                  const int param_dim) {
  for (auto x : offsets) {
    auto id = keys[x];
    if (!block->GetEntry(id)) continue;
    auto* value = block->Get(id);
    float* param = value + param_offset;
    float* grad = update_values + x * param_dim;

    for(auto i = 0; i < param_dim; ++i) {
      auto tmp = *(param + i);
      *(grad + i) += coeff * ((fabs(tmp) < 1.0) ? tmp : ((tmp > 0.0) ? 1.0 : -1.0));
    }
  }
}

void l2decay_func(const uint64_t* keys, 
                  float* update_values,
                  const std::vector<uint64_t>& offsets,
                  ValueBlock* block,
                  const float coeff,
                  const int param_offset,
                  const int param_dim) {
  auto blas = GetBlas<float>();
  for (auto x : offsets) {
    auto id = keys[x];
    if (!block->GetEntry(id)) continue;
    auto* value = block->Get(id);
    float* param = value + param_offset;

    std::vector<float> tmp;
    tmp.resize(param_dim);

    blas.VCOPY(param_dim, param, tmp.data());
    blas.SCAL(param_dim, coeff, tmp.data());

    float* grad = update_values + x * param_dim;
    blas.VADD(param_dim, grad, tmp.data(), grad);
  }
}

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
  auto colunmn_size = columns.size();
  auto load_values =
      paddle::string::split_string<std::string>(columns[colunmn_size - 1], ",");
  values->reserve(meta.names.size());

  int offset = 0;
  for (int x = 0; x < meta.names.size(); ++x) {
    std::vector<float> val;
    auto start = load_values.begin() + offset;
    auto end = load_values.begin() + offset + meta.dims[x];
    PADDLE_ENFORCE_LE(offset + meta.dims[x], load_values.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The data format in txt does not meet the field "
                          "requirements defined in meta"));

    std::transform(start, end, std::back_inserter(val),
                   [](std::string va) { return std::stof(va); });
    values->push_back(val);
    offset += meta.dims[x];
  }
}

int64_t SaveToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
                   const int mode) {
  int64_t not_save_num = 0;
  for (auto value : block->values_) {
    if (mode == SaveMode::delta && !value.second->need_save_) {
      not_save_num++;
      continue;
    }

    auto* vs = value.second->data_.data();
    std::stringstream ss;
    auto id = value.first;
    ss << id << "\t" << value.second->count_ << "\t"
       << value.second->unseen_days_ << "\t" << value.second->is_entry_ << "\t";

    for (int i = 0; i < block->value_length_; i++) {
      ss << vs[i];
      ss << ",";
    }

    ss << "\n";

    os->write(ss.str().c_str(), sizeof(char) * ss.str().size());

    if (mode == SaveMode::base || mode == SaveMode::delta) {
      value.second->need_save_ = false;
    }
  }

  return block->values_.size() - not_save_num;
}

int64_t saveStatToBin(std::string path, std::vector<std::shared_ptr<ValueBlock>>& blocks) {
  size_t block_num = blocks.size();
  if (block_num <= 0) {
    return 0;
  }

  size_t total_size = 0;

  std::ofstream fout(path, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&block_num), sizeof(block_num));

  for (size_t x = 0; x < blocks.size(); ++x) {
    auto block_size = blocks[x]->values_.size();
    std::vector<uint64_t> ids(block_size);
    auto ids_iter = ids.begin();
    // collect ids
    for (auto value : blocks[x]->values_) {
      *ids_iter = value.first;
      ++ids_iter;
    }

    fout.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
    fout.write(reinterpret_cast<const char*>(&ids[0]), sizeof(uint64_t) * block_size);

    for (auto value : blocks[x]->values_) {
      fout.write(reinterpret_cast<const char*>(&value.second->count_), sizeof(value.second->count_));
      fout.write(reinterpret_cast<const char*>(&value.second->unseen_days_), sizeof(value.second->unseen_days_));
      fout.write(reinterpret_cast<const char*>(&value.second->is_entry_), sizeof(value.second->is_entry_));
    }

    total_size += block_size;
  }

  fout.close();

  return total_size;
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
      VLOG(3) << "will not load " << values[0] << " from " << valuepath
              << ", please check id distribution";
      continue;
    }

    auto shard_id = id % local_shard_num;
    auto block = blocks->at(shard_id);

    std::vector<std::vector<float>> kvalues;
    ProcessALine(values, meta, &kvalues);

    auto value_instant = block->Init(id);
    value_instant->initialize();

    if (values.size() == 5) {
      value_instant->count_ = std::stoi(values[1]);
      value_instant->unseen_days_ = std::stoi(values[2]);
      value_instant->is_entry_ = static_cast<bool>(std::stoi(values[3]));
    }

    std::vector<float*> block_values = block->Get(id, meta.names, meta.dims);
    auto blas = GetBlas<float>();
    for (int x = 0; x < meta.names.size(); ++x) {
      blas.VCOPY(meta.dims[x], kvalues[x].data(), block_values[x]);
    }
  }

  return 0;
}

int64_t LoadStatFromBin(const std::string& path, const int pserver_id, 
                         const int pserver_num, const int local_shard_num,
                         std::vector<std::shared_ptr<ValueBlock>>& blocks) {
  std::ifstream fin(path, std::ios::binary);

  size_t block_size;
  size_t block_num;
  size_t length = sizeof(int) + sizeof(int) + sizeof(bool);

  fin.read(reinterpret_cast<char*>(&block_num), sizeof(block_num));
  for(size_t x = 0; x < block_num; ++x) {
    fin.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));

    std::vector<uint64_t> ids(block_size);
    fin.read(reinterpret_cast<char*>(&ids[0]), sizeof(uint64_t) * block_size);

    std::vector<char> values(length * block_size);
    auto* ptr = reinterpret_cast<char*>(&values[0]);
    fin.read(ptr, length * block_size);

    for (size_t i = 0; i < block_size; ++i) {
      auto& id = ids[i];

      if (id % pserver_num != pserver_id) {
        VLOG(3) << "will not load " << id << " from " << path
                << ", please check id distribution";
        ptr += length;
        continue;
      }

      auto shard_id = id % local_shard_num;
      auto block = blocks.at(shard_id);
      auto value_instant = block->Init(id);
      value_instant->count_ = *(reinterpret_cast<int*>(ptr));
      ptr += sizeof(int);
      value_instant->unseen_days_ = *(reinterpret_cast<int*>(ptr));
      ptr += sizeof(int);
      value_instant->is_entry_ = *(reinterpret_cast<bool*>(ptr));
      ptr += sizeof(bool);
    }
  }

  fin.close();

  return 0;
}

int32_t CommonSparseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  sync = _config.common().sync();
  VLOG(1) << "table " << _config.common().table_name() << " is sync: " << sync;

  _global_lr = new float(1.0);

  auto common = _config.common();
  int size = static_cast<int>(common.params().size());

  size_t offset = 0;
  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];

    value_idx_[varname] = x;
    value_names_.push_back(varname);
    value_dims_.push_back(dim);
    value_offsets_.push_back(offset);
    initializer_attrs_.push_back(common.initializers()[x]);

    if (varname == "Param") {
      param_dim_ = dim;
      param_offset_ = offset;
    }

    offset += dim;
  }

  int opt_attr_size = static_cast<int>(common.optimizer_attrs().size());
  for(int x = 0; x < opt_attr_size; ++x) {
    auto& attr = common.optimizer_attrs()[x];
    auto pairs = paddle::string::split_string<std::string>(attr, "&");
    optimizer_attrs_[pairs[0]] = std::stof(pairs[1]);
  }

  auto regularizer = string::split_string<std::string>(common.regularizer(), "&");
  if (regularizer[0] == "l1decay") {
    regularizer_func_ = std::bind(l1decay_func, 
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3,
                                  std::placeholders::_4,
                                  std::stof(regularizer[1]),
                                  param_offset_,
                                  param_dim_);
  } else if (regularizer[0] == "l2decay") {
    regularizer_func_ = std::bind(l2decay_func,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3,
                                  std::placeholders::_4,
                                  std::stof(regularizer[1]),
                                  param_offset_,
                                  param_dim_);
  } else {
    regularizer_func_ = std::bind(nondecay_func,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3,
                                  std::placeholders::_4,
                                  0.0,
                                  param_offset_,
                                  param_dim_);
  }

  initialize_value();
  initialize_optimizer();
  initialize_recorder();
  return 0;
}

int32_t CommonSparseTable::initialize_recorder() { return 0; }

int32_t CommonSparseTable::initialize_value() {
  auto common = _config.common();
  shard_values_.reserve(task_pool_size_);

  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard = std::make_shared<ValueBlock>(
        value_names_, value_dims_, value_offsets_, value_idx_,
        initializer_attrs_, common.entry());

    shard_values_.emplace_back(shard);
  }

  auto accessor = _config.accessor();

  std::vector<uint64_t> feasigns;

  for (size_t x = 0; x < accessor.fea_dim(); ++x) {
    if (x % _shard_num == _shard_idx) {
      feasigns.push_back(x);
    }
  }

  VLOG(0) << "has " << feasigns.size() << " ids need to be pre inited";

  auto buckets = bucket(feasigns.size(), 10);
  for (int x = 0; x < 10; ++x) {
    auto bucket_feasigns = buckets[x + 1] - buckets[x];
    std::vector<uint64_t> ids(bucket_feasigns);
    std::copy(feasigns.begin() + buckets[x], feasigns.begin() + buckets[x + 1],
              ids.begin());
    std::vector<int> batch_cnts(bucket_feasigns, 0);
    std::vector<float> pulls;
    pulls.resize(bucket_feasigns * param_dim_);
    pull_sparse(pulls.data(), ids.data(), bucket_feasigns, batch_cnts);
  }

  return 0;
}

int32_t CommonSparseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();

  if (name == "sgd") {
    optimizer_ = std::make_shared<SSGD>(value_names_, value_dims_,
                                        value_offsets_, value_idx_,
                                        optimizer_attrs_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<SAdam>(value_names_, value_dims_,
                                         value_offsets_, value_idx_,
                                         optimizer_attrs_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "sum") {
    optimizer_ = std::make_shared<SSUM>(value_names_, value_dims_,
                                        value_offsets_, value_idx_,
                                        optimizer_attrs_);
  } else if (name == "adagrad") {
    optimizer_ = std::make_shared<SAdagrad>(value_names_, value_dims_,
                                            value_offsets_, value_idx_,
                                            optimizer_attrs_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "decayed_adagrad") {
    optimizer_ = std::make_shared<SDecayedAdagrad>(value_names_, value_dims_,
                                                   value_offsets_, value_idx_,
                                                   optimizer_attrs_);
    optimizer_->set_global_lr(_global_lr);
  } else {
    VLOG(0) << "init optimizer failed";
  }

  VLOG(0) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonSparseTable::set_global_lr(float* lr) {
  _global_lr = lr;
  optimizer_->set_global_lr(_global_lr);
  return 0;
}

int32_t CommonSparseTable::load(const std::string& path,
                                const std::string& param) {
  rwlock_->WRLock();
  VLOG(0) << "sparse table load with " << path << " with meta " << param;
  // LoadFromText(path, param, _shard_idx, _shard_num, task_pool_size_,
  //              &shard_values_);

  auto varname = _config.common().table_name();
  auto prefix = string::Sprintf("%s/%s", path, varname);

  optimizer_->load(_shard_idx, prefix, shard_values_);

  LoadStatFromBin(string::Sprintf("%s.stat.block%s", prefix, _shard_idx),
                   _shard_idx, _shard_num, task_pool_size_, shard_values_);

  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::save(const std::string& dirname,
                                const std::string& param) {
  rwlock_->WRLock();
  int mode = std::stoi(param);
  VLOG(0) << "sparse table save: " << dirname << " mode: " << mode;

  auto varname = _config.common().table_name();
  std::string var_store =
      string::Sprintf("%s/%s%s", dirname, varname, PSERVER_SAVE_SUFFIX);
  MkDirRecursively(var_store.c_str());

  std::string prefix = string::Sprintf("%s/%s", var_store, varname);

  auto total_ins = optimizer_->save(mode, _shard_idx, prefix, shard_values_);
  // VLOG(3) << "save " << varname << " in dir: " << var_store << " begin";
  // std::vector<std::string> params(_config.common().params().begin(),
  //                                 _config.common().params().end());
  // std::string shard_var_pre =
  //     string::Sprintf("%s.block%d", varname, _shard_idx);

  // std::string value_ = string::Sprintf("%s/%s.txt", var_store, shard_var_pre);

  // std::unique_ptr<std::ofstream> value_out(new std::ofstream(value_));

  // int64_t total_ins = 0;
  // for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
  //   // save values
  //   total_ins += SaveToText(value_out.get(), shard_values_[shard_id], mode);
  // }
  // value_out->close();

  if (mode != SaveMode::delta) {
    saveStatToBin(string::Sprintf("%s.stat.block%s", prefix, _shard_idx), 
                   shard_values_);
  }

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
  std::string meta_ = string::Sprintf("%s.block%s.meta", prefix, _shard_idx);
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
                                       size_t num, std::vector<int>& batch_cnts) {
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
        [this, shard_id, &keys, &batch_cnts, &offset_bucket, &pull_values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            block->Init(id);
            block->PullUpdate(id, batch_cnts[offset]);
            auto val = block->GetValue(id);
            if(val->is_entry_) {
              std::copy_n(val->data_.data() + param_offset_, param_dim_,
                          pull_values + param_dim_ * offset);
            } else {
              memset(pull_values + param_dim_ * offset, 0.0, param_dim_ * sizeof(float));
            }
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
          regularizer_func_(keys, const_cast<float*>(values), offsets, shard_values_[shard_id].get());
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

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            auto val = block->Init(id);
            val->initialize();
            std::copy_n(values + param_dim_ * offset, param_dim_,
                        val->data_.data() + param_offset_);
            block->SetEntry(id, true);
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

int32_t CommonSparseTable::shrink(const std::string& param) {
  rwlock_->WRLock();
  auto params = paddle::string::split_string<std::string>(param, ",");
  float decay_rate = std::stof(params[0]);
  float count_threshold = std::stof(params[1]);
  int unseen_threshold = std::stoi(params[2]);
  VLOG(1) << "sparse table shrink, decay_rate: " << decay_rate 
          << ", count_threshold: " << count_threshold
          << ", unseen_threshold: " << unseen_threshold;

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // shrink
    VLOG(1) << shard_id << " " << task_pool_size_ << " begin shrink";
    shard_values_[shard_id]->Shrink(decay_rate, count_threshold, unseen_threshold);
  }
  rwlock_->UNLock();
  return 0;
}

void CommonSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
