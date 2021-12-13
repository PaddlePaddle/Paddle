// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/distributed/table/ssd_sparse_table.h"

DEFINE_string(rocksdb_path, "database", "path of sparse table rocksdb file");

namespace paddle {
namespace distributed {

int32_t SSDSparseTable::initialize() {
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

  initialize_value();
  initialize_optimizer();
  initialize_recorder();
  _db = paddle::distributed::RocksDBHandler::GetInstance();
  _db->initialize(FLAGS_rocksdb_path, task_pool_size_);
  return 0;
}

int32_t SSDSparseTable::pull_sparse(float* pull_values,
                                    const PullSparseValue& pull_value) {
  auto shard_num = task_pool_size_;
  std::vector<std::future<int>> tasks(shard_num);

  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, shard_num, &pull_value, &pull_values]() -> int {
          auto& block = shard_values_[shard_id];

          std::vector<int> offsets;
          pull_value.Fission(shard_id, shard_num, &offsets);

          for (auto& offset : offsets) {
            auto feasign = pull_value.feasigns_[offset];
            auto frequencie = pull_value.frequencies_[offset];
            float* embedding = nullptr;
            auto iter = block->Find(feasign);
            // in mem
            if (iter == block->end()) {
              embedding = iter->second->data_.data();
              if (pull_value.is_training_) {
                block->AttrUpdate(iter->second, frequencie);
              }
            } else {
              // need create
              std::string tmp_str("");
              if (_db->get(shard_id, (char*)&feasign, sizeof(uint64_t),
                           tmp_str) > 0) {
                embedding = block->Init(feasign, true, frequencie);
              } else {
                // in db
                int data_size = tmp_str.size() / sizeof(float);
                int value_size = block->value_length_;
                float* db_value = (float*)const_cast<char*>(tmp_str.c_str());
                VALUE* value = block->InitGet(feasign);

                // copy to mem
                memcpy(value->data_.data(), db_value,
                       value_size * sizeof(float));
                embedding = db_value;

                // param, count, unseen_day
                value->count_ = db_value[value_size];
                value->unseen_days_ = db_value[value_size + 1];
                value->is_entry_ = db_value[value_size + 2];
                if (pull_value.is_training_) {
                  block->AttrUpdate(value, frequencie);
                }
              }
            }
            std::copy_n(embedding + param_offset_, param_dim_,
                        pull_values + param_dim_ * offset);
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t SSDSparseTable::pull_sparse_ptr(char** pull_values,
                                        const uint64_t* keys, size_t num) {
  auto shard_num = task_pool_size_;
  std::vector<std::future<int>> tasks(shard_num);

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &pull_values, &offset_bucket]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (auto& offset : offsets) {
            auto feasign = keys[offset];
            auto iter = block->Find(feasign);
            VALUE* value = nullptr;
            // in mem
            if (iter != block->end()) {
              value = iter->second;
            } else {
              // need create
              std::string tmp_str("");
              if (_db->get(shard_id, (char*)&feasign, sizeof(uint64_t),
                           tmp_str) > 0) {
                value = block->InitGet(feasign);
              } else {
                // in db
                int data_size = tmp_str.size() / sizeof(float);
                int value_size = block->value_length_;
                float* db_value = (float*)const_cast<char*>(tmp_str.c_str());
                value = block->InitGet(feasign);

                // copy to mem
                memcpy(value->data_.data(), db_value,
                       value_size * sizeof(float));

                // param, count, unseen_day
                value->count_ = db_value[value_size];
                value->unseen_days_ = db_value[value_size + 1];
                value->is_entry_ = db_value[value_size + 2];
              }
            }
            pull_values[offset] = (char*)value;
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t SSDSparseTable::shrink(const std::string& param) { return 0; }

int32_t SSDSparseTable::update_table() {
  int count = 0;
  int value_size = shard_values_[0]->value_length_;
  int db_size = 3 + value_size;
  float tmp_value[db_size];

  for (size_t i = 0; i < task_pool_size_; ++i) {
    auto& block = shard_values_[i];

    for (auto& table : block->values_) {
      for (auto iter = table.begin(); iter != table.end();) {
        VALUE* value = iter->second;
        if (value->unseen_days_ >= 1) {
          tmp_value[value_size] = value->count_;
          tmp_value[value_size + 1] = value->unseen_days_;
          tmp_value[value_size + 2] = value->is_entry_;
          memcpy(tmp_value, value->data_.data(), sizeof(float) * value_size);
          _db->put(i, (char*)&(iter->first), sizeof(uint64_t), (char*)tmp_value,
                   db_size * sizeof(float));
          count++;

          butil::return_object(iter->second);
          iter = table.erase(iter);
        } else {
          ++iter;
        }
      }
    }
    _db->flush(i);
  }
  VLOG(1) << "Table>> update count: " << count;
  return 0;
}

int64_t SSDSparseTable::SaveValueToText(std::ostream* os,
                                        std::shared_ptr<ValueBlock> block,
                                        std::shared_ptr<::ThreadPool> pool,
                                        const int mode, int shard_id) {
  int64_t save_num = 0;

  for (auto& table : block->values_) {
    for (auto& value : table) {
      if (mode == SaveMode::delta && !value.second->need_save_) {
        continue;
      }

      ++save_num;

      std::stringstream ss;
      auto* vs = value.second->data_.data();

      auto id = value.first;

      ss << id << "\t" << value.second->count_ << "\t"
         << value.second->unseen_days_ << "\t" << value.second->is_entry_
         << "\t";

      for (int i = 0; i < block->value_length_ - 1; i++) {
        ss << std::to_string(vs[i]) << ",";
      }

      ss << std::to_string(vs[block->value_length_ - 1]);
      ss << "\n";

      os->write(ss.str().c_str(), sizeof(char) * ss.str().size());

      if (mode == SaveMode::base || mode == SaveMode::delta) {
        value.second->need_save_ = false;
      }
    }
  }

  if (mode != 1) {
    int value_size = block->value_length_;
    auto* it = _db->get_iterator(shard_id);

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      float* value = (float*)const_cast<char*>(it->value().data());
      std::stringstream ss;
      ss << *((uint64_t*)const_cast<char*>(it->key().data())) << "\t"
         << value[value_size] << "\t" << value[value_size + 1] << "\t"
         << value[value_size + 2] << "\t";
      for (int i = 0; i < block->value_length_ - 1; i++) {
        ss << std::to_string(value[i]) << ",";
      }

      ss << std::to_string(value[block->value_length_ - 1]);
      ss << "\n";

      os->write(ss.str().c_str(), sizeof(char) * ss.str().size());
    }
  }

  return save_num;
}

int32_t SSDSparseTable::load(const std::string& path,
                             const std::string& param) {
  rwlock_->WRLock();
  VLOG(3) << "ssd sparse table load with " << path << " with meta " << param;
  LoadFromText(path, param, _shard_idx, _shard_num, task_pool_size_,
               &shard_values_);
  rwlock_->UNLock();
  return 0;
}

int64_t SSDSparseTable::LoadFromText(
    const std::string& valuepath, const std::string& metapath,
    const int pserver_id, const int pserver_num, const int local_shard_num,
    std::vector<std::shared_ptr<ValueBlock>>* blocks) {
  Meta meta = Meta(metapath);

  int num_lines = 0;
  std::ifstream file(valuepath);
  std::string line;

  int value_size = shard_values_[0]->value_length_;
  int db_size = 3 + value_size;
  float tmp_value[db_size];

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
    ProcessALine(values, meta, id, &kvalues);

    block->Init(id, false);

    VALUE* value_instant = block->GetValue(id);

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
    VLOG(3) << "loading: " << id
            << "unseen day: " << value_instant->unseen_days_;
    if (value_instant->unseen_days_ >= 1) {
      tmp_value[value_size] = value_instant->count_;
      tmp_value[value_size + 1] = value_instant->unseen_days_;
      tmp_value[value_size + 2] = value_instant->is_entry_;
      memcpy(tmp_value, value_instant->data_.data(),
             sizeof(float) * value_size);
      _db->put(shard_id, (char*)&(id), sizeof(uint64_t), (char*)tmp_value,
               db_size * sizeof(float));
      block->erase(id);
    }
  }

  return 0;
}

}  // namespace ps
}  // namespace paddle
#endif
