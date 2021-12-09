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

#include <sstream>

#include "paddle/fluid/distributed/table/memory_sparse_table.h"
#include "paddle/fluid/framework/io/fs.h"

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

// TODO(zhaocaibei123): configure
bool FLAGS_pslib_create_value_when_push = false;
int FLAGS_pslib_table_save_max_retry = 3;
bool FLAGS_pslib_enable_create_feasign_randomly = false;

int32_t MemorySparseTable::initialize() {
  shards_task_pool_.resize(task_pool_size_);
  for (int i = 0; i < shards_task_pool_.size(); ++i) {
    shards_task_pool_[i].reset(new ::ThreadPool(1));
  }
  initialize_value();
  VLOG(0) << "initalize MemorySparseTable succ";
  return 0;
}

int32_t MemorySparseTable::initialize_value() {
  sparse_table_shard_num_ = static_cast<int>(_config.shard_num());
  avg_local_shard_num_ =
      SparseTable::sparse_local_shard_num(sparse_table_shard_num_, _shard_num);
  real_local_shard_num_ = avg_local_shard_num_;
  if (real_local_shard_num_ * (_shard_idx + 1) > sparse_table_shard_num_) {
    real_local_shard_num_ =
        sparse_table_shard_num_ - real_local_shard_num_ * _shard_idx;
    real_local_shard_num_ =
        real_local_shard_num_ < 0 ? 0 : real_local_shard_num_;
  }
  VLOG(1) << "memory sparse table avg_local_shard_num_: "
          << avg_local_shard_num_
          << " real_local_shard_num_: " << real_local_shard_num_;

  shard_values_.reserve(real_local_shard_num_);

  for (int x = 0; x < real_local_shard_num_; ++x) {
    auto shard = std::make_shared<SparseTableShard>();
    shard_values_.emplace_back(shard);
  }
  return 0;
}

int32_t MemorySparseTable::load(const std::string& path,
                                const std::string& param) {
  std::string table_path = table_dir(path);
  auto file_list = _afs_client.list(table_path);

  std::sort(file_list.begin(), file_list.end());
  for (auto file : file_list) {
    VLOG(1) << "MemorySparseTable::load() file list: " << file;
  }

  int load_param = atoi(param.c_str());
  auto expect_shard_num = sparse_table_shard_num_;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "MemorySparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "MemorySparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * avg_local_shard_num_;

  size_t feature_value_size = _value_accesor->size() / sizeof(float);
  // TODO(zhaocaibei123): multi-thread
  // int thread_num = shard_values_.size() < 15 ? shard_values_.size() : 15;
  // omp_set_num_threads(thread_num);
  // #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < real_local_shard_num_; ++i) {
    FsChannelConfig channel_config;
    channel_config.path = file_list[file_start_idx + i];
    VLOG(1) << "MemorySparseTable::load begin load " << channel_config.path
            << " into local shard " << i;
    channel_config.converter = _value_accesor->converter(load_param).converter;
    channel_config.deconverter =
        _value_accesor->converter(load_param).deconverter;

    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char* end = NULL;
      auto& shard = shard_values_[i];
      try {
        while (read_channel->read_line(line_data) == 0 &&
               line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto* value = shard->Init(key);
          value->resize(feature_value_size);
          int parse_size =
              _value_accesor->parse_from_string(++end, value->data());
          value->resize(parse_size);

          // for debug
          for (int ii = 0; ii < parse_size; ++ii) {
            VLOG(2) << "MemorySparseTable::load key: " << key << " value " << ii
                    << ": " << value->data()[ii] << " local_shard: " << i;
          }
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR)
              << "MemorySparseTable load failed after read, retry it! path:"
              << channel_config.path << " , retry_num=" << retry_num;
        }
      } catch (...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "MemorySparseTable load failed, retry it! path:"
                   << channel_config.path << " , retry_num=" << retry_num;
      }
      if (retry_num > paddle::distributed::FLAGS_pslib_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + real_local_shard_num_ - 1];
  return 0;
}

int32_t MemorySparseTable::load_local_fs(const std::string& path,
                                         const std::string& param) {
  std::string table_path = table_dir(path);
  auto file_list = paddle::framework::localfs_list(table_path);

  int load_param = atoi(param.c_str());
  auto expect_shard_num = sparse_table_shard_num_;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "MemorySparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "MemorySparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * avg_local_shard_num_;

  size_t feature_value_size = _value_accesor->size() / sizeof(float);

  // int thread_num = shard_values_.size() < 15 ? shard_values_.size() : 15;
  // omp_set_num_threads(thread_num);
  // #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < real_local_shard_num_; ++i) {
    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      std::ifstream file(file_list[file_start_idx + i]);
      char* end = NULL;
      auto& shard = shard_values_[i];
      try {
        while (std::getline(file, line_data) && line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto* value = shard->Init(key);
          value->resize(feature_value_size);
          int parse_size =
              _value_accesor->parse_from_string(++end, value->data());
          value->resize(parse_size);
          // value->shrink_to_fit();
        }
        file.close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR)
              << "MemorySparseTable load failed after read, retry it! path:"
              << file_list[file_start_idx + i] << " , retry_num=" << retry_num;
        }
      } catch (...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "MemorySparseTable load failed, retry it! path:"
                   << file_list[file_start_idx + i]
                   << " , retry_num=" << retry_num;
      }
      if (retry_num > paddle::distributed::FLAGS_pslib_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + real_local_shard_num_ - 1];
  return 0;
}

int32_t MemorySparseTable::save(const std::string& dirname,
                                const std::string& param) {
  VLOG(0) << "MemorySparseTable::save dirname: " << dirname;
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = table_dir(dirname);
  _afs_client.remove(paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  // int thread_num = shard_values_.size() < 20 ? shard_values_.size() : 20;
  std::atomic<uint32_t> feasign_size_all{0};

  size_t file_start_idx = avg_local_shard_num_ * _shard_idx;

  // TODO(zhaocaibei123): openmp
  // omp_set_num_threads(thread_num);
  // #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < real_local_shard_num_; ++i) {
    FsChannelConfig channel_config;
    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path = paddle::string::format_string(
          "%s/part-%03d-%05d.gz", table_path.c_str(), _shard_idx,
          file_start_idx + i);
    } else {
      channel_config.path =
          paddle::string::format_string("%s/part-%03d-%05d", table_path.c_str(),
                                        _shard_idx, file_start_idx + i);
    }
    channel_config.converter = _value_accesor->converter(save_param).converter;
    channel_config.deconverter =
        _value_accesor->converter(save_param).deconverter;
    bool is_write_failed = false;
    int feasign_size = 0;
    int retry_num = 0;
    int err_no = 0;
    auto& shard = shard_values_[i];
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto& table : shard->values_) {
        for (auto& value : table) {
          if (_value_accesor->save(value.second->data(), save_param)) {
            std::string format_value = _value_accesor->parse_to_string(
                value.second->data(), value.second->size());
            if (0 !=
                write_channel->write_line(paddle::string::format_string(
                    "%lu %s", value.first, format_value.c_str()))) {
              ++retry_num;
              is_write_failed = true;
              LOG(ERROR)
                  << "MemorySparseTable save prefix failed, retry it! path:"
                  << channel_config.path << " , retry_num=" << retry_num;
              break;
            }
            ++feasign_size;
          }
        }
      }
      write_channel->close();
      if (err_no == -1) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR)
            << "MemorySparseTable save prefix failed after write, retry it! "
            << "path:" << channel_config.path << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
      if (retry_num > paddle::distributed::FLAGS_pslib_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save prefix failed reach max limit!";
        exit(-1);
      }
    } while (is_write_failed);
    feasign_size_all += feasign_size;
    for (auto& table : shard->values_) {
      for (auto& value : table) {
        _value_accesor->update_stat_after_save(value.second->data(),
                                               save_param);
      }
    }
    LOG(INFO) << "MemorySparseTable save prefix success, path: "
              << channel_config.path;
  }
  // int32 may overflow need to change return value
  return 0;
}

int32_t MemorySparseTable::save_local_fs(const std::string& dirname,
                                         const std::string& param,
                                         const std::string& prefix) {
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = table_dir(dirname);
  int feasign_cnt = 0;
  size_t file_start_idx = avg_local_shard_num_ * _shard_idx;
  for (size_t i = 0; i < real_local_shard_num_; ++i) {
    feasign_cnt = 0;
    auto& shard = shard_values_[i];
    std::string file_name = paddle::string::format_string(
        "%s/part-%s-%03d-%05d", table_path.c_str(), prefix.c_str(), _shard_idx,
        file_start_idx + i);
    std::ofstream os;
    os.open(file_name);
    for (auto& table : shard->values_) {
      for (auto& value : table) {
        if (_value_accesor->save(value.second->data(), save_param)) {
          std::string format_value = _value_accesor->parse_to_string(
              value.second->data(), value.second->size());
          std::string out_line = paddle::string::format_string(
              "%lu %s\n", value.first, format_value.c_str());
          // VLOG(2) << out_line.c_str();
          os.write(out_line.c_str(), sizeof(char) * out_line.size());
          ++feasign_cnt;
        }
      }
    }
    os.close();
    LOG(INFO) << "MemorySparseTable save prefix success, path:" << file_name
              << "feasign_cnt: " << feasign_cnt;
  }
  return 0;
}

std::pair<int64_t, int64_t> MemorySparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& shard : shard_values_) {
    for (auto& table : shard->values_) {
      feasign_size += table.size();
    }
  }

  return {feasign_size, mf_size};
}

int32_t MemorySparseTable::pull_sparse(float* pull_values,
                                       const PullSparseValue& pull_value) {
  std::vector<std::future<int>> tasks(real_local_shard_num_);

  const size_t value_size = _value_accesor->size() / sizeof(float);
  size_t mf_value_size = _value_accesor->mf_size() / sizeof(float);
  size_t select_value_size = _value_accesor->select_size() / sizeof(float);
  // std::atomic<uint32_t> missed_keys{0};

  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      real_local_shard_num_);
  size_t num = pull_value.numel_;
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (pull_value.feasigns_[i] % sparse_table_shard_num_) %
                   avg_local_shard_num_;
    task_keys[shard_id].push_back({pull_value.feasigns_[i], i});
  }
  for (int shard_id = 0; shard_id < real_local_shard_num_; ++shard_id) {
    tasks[shard_id] =
        shards_task_pool_[shard_id % shards_task_pool_.size()]->enqueue(
            [this, shard_id, &task_keys, value_size, pull_values, mf_value_size,
             select_value_size]() -> int {
              auto& local_shard = shard_values_[shard_id];
              float data_buffer[value_size];  // NOLINT
              float* data_buffer_ptr = data_buffer;

              auto& keys = task_keys[shard_id];
              for (size_t i = 0; i < keys.size(); i++) {
                uint64_t key = keys[i].first;
                auto itr = local_shard->Find(key);
                size_t data_size = value_size - mf_value_size;
                if (itr == local_shard->end()) {
                  // ++missed_keys;
                  if (FLAGS_pslib_create_value_when_push) {
                    memset(data_buffer, 0, sizeof(float) * data_size);
                  } else {
                    auto* feature_value = local_shard->Init(key);
                    feature_value->resize(data_size);
                    float* data_ptr = feature_value->data();
                    _value_accesor->create(&data_buffer_ptr, 1);
                    memcpy(data_ptr, data_buffer_ptr,
                           data_size * sizeof(float));
                  }
                } else {
                  data_size = itr->second->size();
                  memcpy(data_buffer_ptr, itr->second->data(),
                         data_size * sizeof(float));
                }
                for (int mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
                  data_buffer[mf_idx] = 0.0;
                }
                auto offset = keys[i].second;
                float* select_data = pull_values + select_value_size * offset;
                _value_accesor->select(&select_data,
                                       (const float**)&data_buffer_ptr, 1);
              }

              return 0;
            });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }

  return 0;
}

int32_t MemorySparseTable::pull_sparse_ptr(char** pull_values,
                                           const uint64_t* keys, size_t num) {
  return 0;
}

int32_t MemorySparseTable::push_sparse(const uint64_t* keys,
                                       const float* values, size_t num) {
  std::vector<std::future<int>> tasks(real_local_shard_num_);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      real_local_shard_num_);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % sparse_table_shard_num_) % avg_local_shard_num_;
    task_keys[shard_id].push_back({keys[i], i});
  }

  const size_t value_col = _value_accesor->size() / sizeof(float);
  size_t mf_value_col = _value_accesor->mf_size() / sizeof(float);
  size_t update_value_col = _value_accesor->update_size() / sizeof(float);

  for (size_t shard_id = 0; shard_id < real_local_shard_num_; ++shard_id) {
    tasks[shard_id] = shards_task_pool_[shard_id % task_pool_size_]->enqueue(
        [this, shard_id, value_col, mf_value_col, update_value_col, values,
         &task_keys]() -> int {
          auto& keys = task_keys[shard_id];
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_col];  // NOLINT
          float* data_buffer_ptr = data_buffer;

          for (int i = 0; i < keys.size(); ++i) {
            uint64_t key = keys[i].first;
            uint64_t push_data_idx = keys[i].second;
            const float* update_data =
                values + push_data_idx * update_value_col;
            auto itr = local_shard->Find(key);
            if (itr == local_shard->end()) {
              VLOG(0) << "sparse table push_sparse: " << key << "not found!";
              if (FLAGS_pslib_enable_create_feasign_randomly &&
                  !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard->Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(feature_value->data(), data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard->Find(key);
            } else {
              VLOG(2) << "sparse table debug push_sparse: " << key << " found!";
            }

            auto* feature_value = itr->second;
            float* value_data = feature_value->data();
            size_t value_size = feature_value->size();

            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accesor->update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->update(&data_buffer_ptr, &update_data, 1);

              if (_value_accesor->need_extend_mf(data_buffer)) {
                feature_value->resize(value_col);
                value_data = feature_value->data();
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

int32_t MemorySparseTable::push_sparse(const uint64_t* keys,
                                       const float** values, size_t num) {
  _push_sparse(keys, values, num);
  return 0;
}

int32_t MemorySparseTable::_push_sparse(const uint64_t* keys,
                                        const float** values, size_t num) {
  std::vector<std::future<int>> tasks(real_local_shard_num_);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      real_local_shard_num_);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % sparse_table_shard_num_) % avg_local_shard_num_;
    task_keys[shard_id].push_back({keys[i], i});
  }

  size_t value_col = _value_accesor->size() / sizeof(float);
  size_t mf_value_col = _value_accesor->mf_size() / sizeof(float);
  size_t update_value_col = _value_accesor->update_size() / sizeof(float);

  for (int shard_id = 0; shard_id < real_local_shard_num_; ++shard_id) {
    tasks[shard_id] = shards_task_pool_[shard_id % task_pool_size_]->enqueue(
        [this, shard_id, value_col, mf_value_col, update_value_col, values,
         &task_keys]() -> int {
          auto& keys = task_keys[shard_id];
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_col];  // NOLINT
          float* data_buffer_ptr = data_buffer;

          for (int i = 0; i < keys.size(); ++i) {
            uint64_t key = keys[i].first;
            uint64_t push_data_idx = keys[i].second;
            const float* update_data = values[push_data_idx];
            auto itr = local_shard->Find(key);
            if (itr == local_shard->end()) {
              if (FLAGS_pslib_enable_create_feasign_randomly &&
                  !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard->Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(feature_value->data(), data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard->Find(key);
            }
            auto* feature_value = itr->second;
            float* value_data = feature_value->data();
            size_t value_size = feature_value->size();
            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accesor->update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->update(&data_buffer_ptr, &update_data, 1);
              if (_value_accesor->need_extend_mf(data_buffer)) {
                feature_value->resize(value_col);
                value_data = feature_value->data();
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

int32_t MemorySparseTable::flush() { return 0; }

int32_t MemorySparseTable::shrink(const std::string& param) {
  VLOG(0) << "MemorySparseTable::shrink";
  // TODO(zhaocaibei123): implement with multi-thread
  for (int shard_id = 0; shard_id < real_local_shard_num_; ++shard_id) {
    // shrink
    auto& shard = shard_values_[shard_id];
    for (auto& table : shard->values_) {
      for (auto iter = table.begin(); iter != table.end();) {
        if (_value_accesor->shrink(iter->second->data())) {
          butil::return_object(iter->second);
          iter = table.erase(iter);
          VLOG(1) << "shrink erase key: " << iter->first;
        } else {
          ++iter;
        }
      }
    }
  }
  return 0;
}

void MemorySparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
