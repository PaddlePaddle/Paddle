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

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
class ValueBlock;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace distributed {

//TODO: need this?
void DownpourSparseTable::ProcessALine(const std::vector<std::string>& columns,
                                     const Meta& meta, const int64_t id,
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

    std::transform(start, end, std::back_inserter(val), [id](std::string va) {
      float v = 0.0;

      try {
        v = lexical_cast<float>(va);
      } catch (boost::bad_lexical_cast& e) {
        VLOG(0) << "id: " << id << " get unexpected value: " << va
                << " and be reset to: 0.0";
      }
      return v;
    });

    values->push_back(val);
    offset += meta.dims[x];
  }
}

void DownpourSparseTable::SaveMetaToText(std::ostream* os,
                                       const CommonAccessorParameter& common,
                                       const size_t shard_idx,
                                       const int64_t total) {
  // save meta
  std::stringstream stream;
  stream << "param=" << common.table_name() << "\n";
  stream << "shard_id=" << shard_idx << "\n";
  stream << "row_names=" << paddle::string::join_strings(common.params(), ',')
         << "\n";
  stream << "row_dims=" << paddle::string::join_strings(common.dims(), ',')
         << "\n";
  stream << "count=" << total << "\n";
  os->write(stream.str().c_str(), sizeof(char) * stream.str().size());
}

int64_t DownpourSparseTable::SaveValueToText(std::ostream* os,
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

  return save_num;
}

int64_t DownpourSparseTable::LoadFromText(
    const std::string& valuepath, const std::string& metapath,
    const int pserver_id, const int pserver_num, const int local_shard_num,
    std::vector<std::shared_ptr<ValueBlock>>* blocks) {
  Meta meta = Meta(metapath);

  int num_lines = 0;
  std::ifstream file(valuepath);
  std::string line;

  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    auto id = lexical_cast<uint64_t>(values[0]);

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
      value_instant->count_ = lexical_cast<int>(values[1]);
      value_instant->unseen_days_ = lexical_cast<int>(values[2]);
      value_instant->is_entry_ =
          static_cast<bool>(lexical_cast<int>(values[3]));
    }

    std::vector<float*> block_values = block->Get(id, meta.names, meta.dims);
    auto blas = GetBlas<float>();
    for (int x = 0; x < meta.names.size(); ++x) {
      blas.VCOPY(meta.dims[x], kvalues[x].data(), block_values[x]);
    }
  }

  return 0;
}

int32_t DownpourSparseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  initialize_value();
  //initialize_optimizer();
  //initialize_recorder();
  return 0;
}

//int32_t DownpourSparseTable::initialize_recorder() { return 0; }

int32_t DownpourSparseTable::initialize_value() {
  auto common = _config.common();
  shard_values_.reserve(task_pool_size_);

  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard = std::make_shared<DownpourValueBlock>();
    shard_values_.emplace_back(shard);
  }
  return 0;
}
//TODO: do not need this
int32_t DownpourSparseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();

  if (name == "sgd") {
    optimizer_ = std::make_shared<SSGD>(value_names_, value_dims_,
                                        value_offsets_, value_idx_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<SAdam>(value_names_, value_dims_,
                                         value_offsets_, value_idx_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "sum") {
    optimizer_ = std::make_shared<SSUM>(value_names_, value_dims_,
                                        value_offsets_, value_idx_);
  } else {
    VLOG(3) << "init optimizer failed";
  }

  VLOG(3) << "init optimizer " << name << " done";
  return 0;
}

int32_t DownpourSparseTable::load(const std::string& path,
                                const std::string& param) {
  std::string table_path = table_dir(path);
  auto file_list = _afs_client.list(table_path);

  int load_param = atoi(param.c_str());
  auto expect_shard_num = _shard_num * task_pool_size_;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "DownpourSparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
        //TODO load
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "DownpourSparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * shard_values_.size();

  int load_param = atoi(param.c_str());
  size_t feature_value_size = _value_accesor->size() / sizeof(float);
  int thread_num = shard_values_.size() < 15 ? shard_values_.size() : 15;
  omp_set_num_threads(thread_num);
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < task_pool_size_; ++ i) {
    FsChannelConfig channel_config;
    channel_config.path = file_list[file_start_idx + i];
    channel_config.converter = _value_accesor->converter(load_param).converter;
    channel_config.deconverter = _value_accesor->converter(load_param).deconverter;

    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char *end = NULL;
      auto& shard = shard_values_[i];
      try {
        while (read_channel->read_line(line_data) == 0 && line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto* value = shard.Init(key);
          value->resize(feature_value_size);
          int parse_size = _value_accesor->parse_from_string(++end, value->data());
          value->resize(parse_size);
          //value->shrink_to_fit();
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR) << "DownpourSparseTable load failed after read, retry it! path:"
                     << channel_config.path << " , retry_num=" << retry_num;
        }
      } catch(...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "DownpourSparseTable load failed, retry it! path:" << channel_config.path
                   << " , retry_num=" << retry_num;
      }
      if (retry_num > FLAGS_pslib_table_save_max_retry) {
         LOG(ERROR) << "DownpourSparseTable load failed reach max limit!";
         exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "DownpourSparseTable load success, path from "
            << file_list[start_idx] << " to " << file_list[end_idx - 1];
  return 0;
}

int32_t DownpourSparseTable::save(const std::string& dirname,
                                const std::string& param,
                                const std::string& prefix) {
  int save_param = atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = table_dir(dirname);
  _afs_client.remove(format_string("%s/part-%s-%03d-*", table_path.c_str(), prefix.c_str(), _shard_idx));
  int thread_num = shard_values_.size() < 20 ? shard_values_.size() : 20;
  std::atomic<uint32_t> feasign_size_all{0};

  //TODO: openmp
  omp_set_num_threads(thread_num);
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < shard_values_.size(); ++i) {
    FsChannelConfig channel_config;
    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path = format_string("%s/part-%s-%03d-%05d.gz",
        table_path.c_str(), prefix.c_str(), _shard_idx, i);
    } else {
      channel_config.path = format_string("%s/part-%s-%03d-%05d",
        table_path.c_str(), prefix.c_str(), _shard_idx, i);
    }
    channel_config.converter = _value_accesor->converter(save_param).converter;
    channel_config.deconverter = _value_accesor->converter(save_param).deconverter;
    bool is_write_failed = false;
    int feasign_size = 0;
    int retry_num = 0;
    int err_no = 0;
    auto& shard = shard_values_[i];
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto& table : shard->values_) {
        for (auto& value : table) {
          if (_value_accesor->save(value->second->data(), save_param)) {
            std::string format_value = _value_accesor->
              parse_to_string(value->second->data(), value->second->size());
            if (0 != write_channel->write_line(
              format_string("%lu %s", value->first, format_value.c_str()))) {
                ++retry_num;
                is_write_failed = true;
                LOG(ERROR) << "DownpourSparseTable save prefix failed, retry it! path:"
                           << channel_config.path << " , retry_num=" << retry_num;
                break;
            }
            ++feasign_size;
          }
        }
        write_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_write_failed = true;
          LOG(ERROR) << "DownpourSparseTable save prefix failed after write, retry it! "
                     << "path:" << channel_config.path << " , retry_num=" << retry_num;
        }
        if (is_write_failed) {
          _afs_client.remove(channel_config.path);
        }
        if (retry_num > FLAGS_pslib_table_save_max_retry) {
          LOG(ERROR) << "DownpourSparseTable save prefix failed reach max limit!";
          exit(-1);
        }
      }
    } while (is_write_failed);
      feasign_size_all += feasign_size;
      for (auto& table : shard->values_) {
        for (auto& value : table) {
          _value_accesor->update_stat_after_save(it->second->.data(), save_param);
        }
      }
    }
    LOG(INFO) << "DownpourSparseTable save prefix success, path:"
              << format_string("%s/%03d/part-%s-%03d-", path.c_str(), _config.table_id(), prefix.c_str(), _shard_idx);
    //int32 may overflow need to change return value
    return 0;
}

std::pair<int64_t, int64_t> DownpourSparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& shard : shard_values_) {
    for (auto& table : shard->values_) {
      feasign_size += table.size();
    }
  }

  return {feasign_size, mf_size};
}


int32_t DownpourSparseTable::pull_sparse(float* pull_values,
                                       const PullSparseValue& pull_value) {
  auto shard_num = task_pool_size_;
  std::vector<std::future<int>> tasks(shard_num);

  size_t value_size = _value_accesor->size() / sizeof(float);
  size_t mf_value_size = _value_accesor->mf_size() / sizeof(float);
  size_t select_value_size = _value_accesor->select_size() / sizeof(float);
  std::atomic<uint32_t> missed_keys{0};
  
  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, shard_num, &pull_value, &pull_values, value_size, mf_value_size, select_value_size]() -> int {
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_size];
          float* data_buffer_ptr = data_buffer;
          for (auto& offset : offsets) {
            uint64_t key = pull_value.feasigns_[offset];
            auto itr = local_shard.Find(key);
            size_t data_size = value_size - mf_value_size;
            if (itr == local_shard.end()) {
              ++missed_keys;
              //TODO: FLAGS
              if (FLAGS_pslib_create_value_when_push) {
                memset(data_buffer, 0, sizeof(float) * data_size);
              } else {
                 auto* feature_value = local_shard.Init(key);
                 feature_value->resize(data_size);
                 float* data_ptr = const_cast<float*>(feature_value->data());
                 _value_accesor->create(&data_buffer_ptr, 1);
                 memcpy(data_ptr, data_buffer_ptr, data_size * sizeof(float));
              }
            } else {
              data_size = itr->second->size();
              memcpy(data_buffer_ptr, itr->second->data(), data_size * sizeof(float));
            }
            for (int mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
              data_buffer[mf_idx] = 0.0;
            }
            //int pull_data_idx = keys[i].second;
            //float* select_data = pull_values + pull_data_idx * select_value_size;
            float* select_data = pull_values + select_value_size * offset;
            _value_accesor->select(&select_data, (const float**)&data_buffer_ptr, 1);
          }
          return 0;

        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

//TODO: need this?? used in ps_local_client
int32_t DownpourSparseTable::pull_sparse_ptr(char** pull_values,
                                           const uint64_t* keys, size_t num) {
  /*
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &pull_values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            auto* value = block->InitGet(id);
            // std::copy_n(value + param_offset_, param_dim_,
            //            pull_values + param_dim_ * offset);
            pull_values[offset] = reinterpret_cast<char*>(value);
          }

          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }*/
  return 0;
}

int32_t DownpourSparseTable::push_sparse(const uint64_t* keys,
                                        const float* values, size_t num) {
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  size_t value_col = _value_accesor->size() / sizeof(float);
  size_t mf_value_col = _value_accesor->mf_size() / sizeof(float);
  size_t update_value_col = _value_accesor->update_size() / sizeof(float);

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &values, num, &offset_bucket, value_col, mf_value_col, update_value_col]() -> int {
          auto& offsets = offset_bucket[shard_id];
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_col];
          float* data_buffer_ptr = data_buffer;

          for (auto& offset : offsets) {
            uint64_t key = keys[offset];
            //uint64_t push_data_idx = keys[i].second;
            const float* update_data = values + offset * update_value_col;
            auto itr = local_shard.Find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pslib_enable_create_feasign_randomly
                && !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard.Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(const_cast<float*>(feature_value->data()), data_buffer_ptr, value_size * sizeof(float));
              itr = local_shard.Find(key);
            }
            auto* feature_value = itr->second;
            float* value_data = const_cast<float*>(feature_value->data());
            size_t value_size = feature_value->size();
            if (value_size == value_col) { //已拓展到最大size, 则就地update
              _value_accesor->update(&value_data, &update_data, 1);
            } else {//拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->update(&data_buffer_ptr, &update_data, 1);
              if (_value_accesor->need_extend_mf(data_buffer)) {
                feature_value->resize(value_col);
                value_data = const_cast<float*>(feature_value->data());
                _value_accesor->create(&value_data, 1);
              }
              memcpy(value_data, data_buffer_ptr, value_size * sizeof(float));
            }
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

//TODO: ?
int32_t DownpourSparseTable::push_sparse(const uint64_t* keys,
                                       const float** values, size_t num) {
  _push_sparse(keys, values, num);
  return 0;
}

int32_t DownpourSparseTable::_push_sparse(const uint64_t* keys,
                                        const float** values, size_t num) {
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  size_t value_col = _value_accesor->size() / sizeof(float);
  size_t mf_value_col = _value_accesor->mf_size() / sizeof(float);
  size_t update_value_col = _value_accesor->update_size() / sizeof(float);

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &values, num, &offset_bucket, value_col, mf_value_col, update_value_col]() -> int {
          auto& offsets = offset_bucket[shard_id];
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_col];
          float* data_buffer_ptr = data_buffer;

          for (auto& offset : offsets) {
            uint64_t key = keys[offset];
            //uint64_t push_data_idx = keys[i].second;
            const float* update_data = values[offset];
            auto itr = local_shard.Find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pslib_enable_create_feasign_randomly
                && !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard.Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(const_cast<float*>(feature_value->data()), data_buffer_ptr, value_size * sizeof(float));
              itr = local_shard.Find(key);
            }
            auto* feature_value = itr->second;
            float* value_data = const_cast<float*>(feature_value->data());
            size_t value_size = feature_value->size();
            if (value_size == value_col) { //已拓展到最大size, 则就地update
              _value_accesor->update(&value_data, &update_data, 1);
            } else {//拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->update(&data_buffer_ptr, &update_data, 1);
              if (_value_accesor->need_extend_mf(data_buffer)) {
                feature_value->resize(value_col);
                value_data = const_cast<float*>(feature_value->data());
                _value_accesor->create(&value_data, 1);
              }
              memcpy(value_data, data_buffer_ptr, value_size * sizeof(float));
            }
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t DownpourSparseTable::push_sparse_param(const uint64_t* keys,
                                             const float* values, size_t num) {
  /*
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
            auto* value = block->Init(id, false);
            std::copy_n(values + param_dim_ * offset, param_dim_,
                        value + param_offset_);
            block->SetEntry(id, true);
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  */
  return 0;
}

int32_t DownpourSparseTable::flush() { return 0; }

//TODO: no need param
int32_t DownpourSparseTable::shrink(const std::string& param) {
  //TODO implement with multi-thread
  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // shrink
    auto& shard = shard_values_[shard_id];
    for (auto& table : shard->values_) {
      for (auto iter = table.begin(); iter != table.end();) {
        if (_value_accesor->shrink(iter->second->data())) {
          butil::return_object(iter->second); //TODO: need this?
          iter = table.erase(iter);
        } else {
          ++ iter;
        }
      }
    }
  }
  return 0;
}

void DownpourSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
