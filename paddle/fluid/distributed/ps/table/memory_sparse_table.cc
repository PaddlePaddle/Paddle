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

#include <omp.h>
#include <sstream>

#include "paddle/fluid/distributed/common/cost_timer.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"
#include "paddle/fluid/framework/io/fs.h"

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_bool(pserver_print_missed_key_num_every_push, false,
            "pserver_print_missed_key_num_every_push");
DEFINE_bool(pserver_create_value_when_push, true,
            "pserver create value when push");
DEFINE_bool(pserver_enable_create_feasign_randomly, false,
            "pserver_enable_create_feasign_randomly");
DEFINE_int32(pserver_table_save_max_retry, 3, "pserver_table_save_max_retry");

namespace paddle {
namespace distributed {

int32_t MemorySparseTable::Initialize() {
  _shards_task_pool.resize(_task_pool_size);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }
  auto& profiler = CostProfiler::instance();
  profiler.register_profiler("pserver_sparse_update_all");
  profiler.register_profiler("pserver_sparse_select_all");
  InitializeValue();
  VLOG(0) << "initalize MemorySparseTable succ";
  return 0;
}

int32_t MemorySparseTable::InitializeValue() {
  _sparse_table_shard_num = static_cast<int>(_config.shard_num());
  _avg_local_shard_num =
      sparse_local_shard_num(_sparse_table_shard_num, _shard_num);
  _real_local_shard_num = _avg_local_shard_num;
  if (_real_local_shard_num * (_shard_idx + 1) > _sparse_table_shard_num) {
    _real_local_shard_num =
        _sparse_table_shard_num - _real_local_shard_num * _shard_idx;
    _real_local_shard_num =
        _real_local_shard_num < 0 ? 0 : _real_local_shard_num;
  }
  VLOG(1) << "memory sparse table _avg_local_shard_num: "
          << _avg_local_shard_num
          << " _real_local_shard_num: " << _real_local_shard_num;

  _local_shards.reset(new shard_type[_real_local_shard_num]);

  return 0;
}

int32_t MemorySparseTable::Load(const std::string& path,
                                const std::string& param) {
  std::string table_path = TableDir(path);
  auto file_list = _afs_client.list(table_path);

  std::sort(file_list.begin(), file_list.end());
  for (auto file : file_list) {
    VLOG(1) << "MemorySparseTable::Load() file list: " << file;
  }

  int load_param = atoi(param.c_str());
  auto expect_shard_num = _sparse_table_shard_num;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "MemorySparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "MemorySparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * _avg_local_shard_num;

  size_t feature_value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);

  int thread_num = _real_local_shard_num < 15 ? _real_local_shard_num : 15;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    FsChannelConfig channel_config;
    channel_config.path = file_list[file_start_idx + i];
    VLOG(1) << "MemorySparseTable::load begin load " << channel_config.path
            << " into local shard " << i;
    channel_config.converter = _value_accesor->Converter(load_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(load_param).deconverter;

    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char* end = NULL;
      auto& shard = _local_shards[i];
      try {
        while (read_channel->read_line(line_data) == 0 &&
               line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto& value = shard[key];
          value.resize(feature_value_size);
          int parse_size = _value_accesor->ParseFromString(++end, value.data());
          value.resize(parse_size);

          // for debug
          for (int ii = 0; ii < parse_size; ++ii) {
            VLOG(2) << "MemorySparseTable::load key: " << key << " value " << ii
                    << ": " << value.data()[ii] << " local_shard: " << i;
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
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + _real_local_shard_num - 1];
  return 0;
}

int32_t MemorySparseTable::LoadLocalFS(const std::string& path,
                                       const std::string& param) {
  std::string table_path = TableDir(path);
  auto file_list = paddle::framework::localfs_list(table_path);

  int load_param = atoi(param.c_str());
  auto expect_shard_num = _sparse_table_shard_num;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "MemorySparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "MemorySparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * _avg_local_shard_num;

  size_t feature_value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);

  int thread_num = _real_local_shard_num < 15 ? _real_local_shard_num : 15;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      std::ifstream file(file_list[file_start_idx + i]);
      char* end = NULL;
      auto& shard = _local_shards[i];
      try {
        while (std::getline(file, line_data) && line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto& value = shard[key];
          value.resize(feature_value_size);
          int parse_size = _value_accesor->ParseFromString(++end, value.data());
          value.resize(parse_size);
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
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + _real_local_shard_num - 1];
  return 0;
}

int32_t MemorySparseTable::Save(const std::string& dirname,
                                const std::string& param) {
  VLOG(0) << "MemorySparseTable::save dirname: " << dirname;
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = TableDir(dirname);
  _afs_client.remove(paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  std::atomic<uint32_t> feasign_size_all{0};

  size_t file_start_idx = _avg_local_shard_num * _shard_idx;

  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
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
    channel_config.converter = _value_accesor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(save_param).deconverter;
    bool is_write_failed = false;
    int feasign_size = 0;
    int retry_num = 0;
    int err_no = 0;
    auto& shard = _local_shards[i];
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_value_accesor->Save(it.value().data(), save_param)) {
          std::string format_value = _value_accesor->ParseToString(
              it.value().data(), it.value().size());
          if (0 !=
              write_channel->write_line(paddle::string::format_string(
                  "%lu %s", it.key(), format_value.c_str()))) {
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
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save prefix failed reach max limit!";
        exit(-1);
      }
    } while (is_write_failed);
    feasign_size_all += feasign_size;
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      _value_accesor->UpdateStatAfterSave(it.value().data(), save_param);
    }
    LOG(INFO) << "MemorySparseTable save prefix success, path: "
              << channel_config.path;
  }
  // int32 may overflow need to change return value
  return 0;
}

int32_t MemorySparseTable::SaveLocalFS(const std::string& dirname,
                                       const std::string& param,
                                       const std::string& prefix) {
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = TableDir(dirname);
  int feasign_cnt = 0;
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;

  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
  std::atomic<uint32_t> feasign_size_all{0};

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    feasign_cnt = 0;
    auto& shard = _local_shards[i];
    std::string file_name = paddle::string::format_string(
        "%s/part-%s-%03d-%05d", table_path.c_str(), prefix.c_str(), _shard_idx,
        file_start_idx + i);
    std::ofstream os;
    os.open(file_name);
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      if (_value_accesor->Save(it.value().data(), save_param)) {
        std::string format_value =
            _value_accesor->ParseToString(it.value().data(), it.value().size());
        std::string out_line = paddle::string::format_string(
            "%lu %s\n", it.key(), format_value.c_str());
        // VLOG(2) << out_line.c_str();
        os.write(out_line.c_str(), sizeof(char) * out_line.size());
        ++feasign_cnt;
      }
    }
    os.close();
    LOG(INFO) << "MemorySparseTable save prefix success, path:" << file_name
              << "feasign_cnt: " << feasign_cnt;
  }
  return 0;
}

int64_t MemorySparseTable::LocalSize() {
  int64_t local_size = 0;
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    local_size += _local_shards[i].size();
  }
  return local_size;
}

int64_t MemorySparseTable::LocalMFSize() {
  std::vector<int64_t> size_arr(_real_local_shard_num, 0);
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  int64_t ret_size = 0;
  for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this, shard_id, &size_arr]() -> int {
              auto& local_shard = _local_shards[shard_id];
              for (auto it = local_shard.begin(); it != local_shard.end();
                   ++it) {
                if (_value_accesor->HasMF(it.value().size())) {
                  size_arr[shard_id] += 1;
                }
              }
              return 0;
            });
  }
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    tasks[i].wait();
  }
  for (auto x : size_arr) {
    ret_size += x;
  }
  return ret_size;
}

std::pair<int64_t, int64_t> MemorySparseTable::PrintTableStat() {
  int64_t feasign_size = LocalSize();
  int64_t mf_size = LocalMFSize();
  return {feasign_size, mf_size};
}

int32_t MemorySparseTable::Pull(TableContext& context) {
  CHECK(context.value_type == Sparse);
  if (context.use_ptr) {
    char** pull_values = context.pull_context.ptr_values;
    const uint64_t* keys = context.pull_context.keys;
    return PullSparsePtr(pull_values, keys, context.num);
  } else {
    float* pull_values = context.pull_context.values;
    const PullSparseValue& pull_value = context.pull_context.pull_value;
    return PullSparse(pull_values, pull_value);
  }
}

int32_t MemorySparseTable::Push(TableContext& context) {
  CHECK(context.value_type == Sparse);
  if (!context.use_ptr) {
    return PushSparse(context.push_context.keys, context.push_context.values,
                      context.num);
  } else {
    return PushSparse(context.push_context.keys,
                      context.push_context.ptr_values, context.num);
  }
}

int32_t MemorySparseTable::PullSparse(float* pull_values,
                                      const PullSparseValue& pull_value) {
  CostTimer timer("pserver_sparse_select_all");
  std::vector<std::future<int>> tasks(_real_local_shard_num);

  const size_t value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  size_t select_value_size =
      _value_accesor->GetAccessorInfo().select_size / sizeof(float);
  // std::atomic<uint32_t> missed_keys{0};

  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  size_t num = pull_value.numel_;
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (pull_value.feasigns_[i] % _sparse_table_shard_num) %
                   _avg_local_shard_num;
    task_keys[shard_id].push_back({pull_value.feasigns_[i], i});
  }
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this, shard_id, &task_keys, value_size, pull_values, mf_value_size,
             select_value_size]() -> int {
              auto& local_shard = _local_shards[shard_id];
              float data_buffer[value_size];  // NOLINT
              float* data_buffer_ptr = data_buffer;

              auto& keys = task_keys[shard_id];
              for (size_t i = 0; i < keys.size(); i++) {
                uint64_t key = keys[i].first;
                auto itr = local_shard.find(key);
                size_t data_size = value_size - mf_value_size;
                if (itr == local_shard.end()) {
                  // ++missed_keys;
                  if (FLAGS_pserver_create_value_when_push) {
                    memset(data_buffer, 0, sizeof(float) * data_size);
                  } else {
                    auto& feature_value = local_shard[key];
                    feature_value.resize(data_size);
                    float* data_ptr = feature_value.data();
                    _value_accesor->Create(&data_buffer_ptr, 1);
                    memcpy(data_ptr, data_buffer_ptr,
                           data_size * sizeof(float));
                  }
                } else {
                  data_size = itr.value().size();
                  memcpy(data_buffer_ptr, itr.value().data(),
                         data_size * sizeof(float));
                }
                for (int mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
                  data_buffer[mf_idx] = 0.0;
                }
                auto offset = keys[i].second;
                float* select_data = pull_values + select_value_size * offset;
                _value_accesor->Select(&select_data,
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

int32_t MemorySparseTable::PullSparsePtr(char** pull_values,
                                         const uint64_t* keys, size_t num) {
  CostTimer timer("pscore_sparse_select_all");
  size_t value_size = _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);

  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }
  // std::atomic<uint32_t> missed_keys{0};
  for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this, shard_id, &task_keys, pull_values, value_size,
             mf_value_size]() -> int {
              auto& keys = task_keys[shard_id];
              auto& local_shard = _local_shards[shard_id];
              float data_buffer[value_size];
              float* data_buffer_ptr = data_buffer;
              for (int i = 0; i < keys.size(); ++i) {
                uint64_t key = keys[i].first;
                auto itr = local_shard.find(key);
                size_t data_size = value_size - mf_value_size;
                FixedFeatureValue* ret = NULL;
                if (itr == local_shard.end()) {
                  // ++missed_keys;
                  auto& feature_value = local_shard[key];
                  feature_value.resize(data_size);
                  float* data_ptr = feature_value.data();
                  _value_accesor->Create(&data_buffer_ptr, 1);
                  memcpy(data_ptr, data_buffer_ptr, data_size * sizeof(float));
                  ret = &feature_value;
                } else {
                  ret = itr.value_ptr();
                }
                int pull_data_idx = keys[i].second;
                pull_values[pull_data_idx] = (char*)ret;
              }
              return 0;
            });
  }
  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t MemorySparseTable::PushSparse(const uint64_t* keys, const float* values,
                                      size_t num) {
  CostTimer timer("pserver_sparse_update_all");
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }

  const size_t value_col =
      _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_col =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  size_t update_value_col =
      _value_accesor->GetAccessorInfo().update_size / sizeof(float);

  for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id % _task_pool_size]->enqueue(
        [this, shard_id, value_col, mf_value_col, update_value_col, values,
         &task_keys]() -> int {
          auto& keys = task_keys[shard_id];
          auto& local_shard = _local_shards[shard_id];
          float data_buffer[value_col];  // NOLINT
          float* data_buffer_ptr = data_buffer;
          for (int i = 0; i < keys.size(); ++i) {
            uint64_t key = keys[i].first;
            uint64_t push_data_idx = keys[i].second;
            const float* update_data =
                values + push_data_idx * update_value_col;
            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pserver_enable_create_feasign_randomly &&
                  !_value_accesor->CreateValue(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto& feature_value = local_shard[key];
              feature_value.resize(value_size);
              _value_accesor->Create(&data_buffer_ptr, 1);
              memcpy(feature_value.data(), data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard.find(key);
            }

            auto& feature_value = itr.value();
            float* value_data = feature_value.data();
            size_t value_size = feature_value.size();

            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accesor->Update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->Update(&data_buffer_ptr, &update_data, 1);

              if (_value_accesor->NeedExtendMF(data_buffer)) {
                feature_value.resize(value_col);
                value_data = feature_value.data();
                _value_accesor->Create(&value_data, 1);
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

int32_t MemorySparseTable::PushSparse(const uint64_t* keys,
                                      const float** values, size_t num) {
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }

  size_t value_col = _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_col =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  size_t update_value_col =
      _value_accesor->GetAccessorInfo().update_size / sizeof(float);

  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id % _task_pool_size]->enqueue(
        [this, shard_id, value_col, mf_value_col, update_value_col, values,
         &task_keys]() -> int {
          auto& keys = task_keys[shard_id];
          auto& local_shard = _local_shards[shard_id];
          float data_buffer[value_col];  // NOLINT
          float* data_buffer_ptr = data_buffer;
          for (int i = 0; i < keys.size(); ++i) {
            uint64_t key = keys[i].first;
            uint64_t push_data_idx = keys[i].second;
            const float* update_data = values[push_data_idx];
            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pserver_enable_create_feasign_randomly &&
                  !_value_accesor->CreateValue(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto& feature_value = local_shard[key];
              feature_value.resize(value_size);
              _value_accesor->Create(&data_buffer_ptr, 1);
              memcpy(feature_value.data(), data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard.find(key);
            }
            auto& feature_value = itr.value();
            float* value_data = feature_value.data();
            size_t value_size = feature_value.size();
            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accesor->Update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accesor->Update(&data_buffer_ptr, &update_data, 1);
              if (_value_accesor->NeedExtendMF(data_buffer)) {
                feature_value.resize(value_col);
                value_data = feature_value.data();
                _value_accesor->Create(&value_data, 1);
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

int32_t MemorySparseTable::Flush() { return 0; }

int32_t MemorySparseTable::Shrink(const std::string& param) {
  VLOG(0) << "MemorySparseTable::Shrink";
  // TODO(zhaocaibei123): implement with multi-thread
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    // Shrink
    auto& shard = _local_shards[shard_id];
    for (auto it = shard.begin(); it != shard.end();) {
      if (_value_accesor->Shrink(it.value().data())) {
        it = shard.erase(it);
      } else {
        ++it;
      }
    }
  }
  return 0;
}

void MemorySparseTable::Clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
