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

#include "paddle/fluid/distributed/table/ctr_sparse_table.h"
#include "paddle/fluid/distributed/common/afs_warpper.h"
#include "paddle/fluid/framework/io/fs.h"
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

//TODO
bool FLAGS_pslib_create_value_when_push = false;
int FLAGS_pslib_table_save_max_retry = 3;
bool FLAGS_pslib_enable_create_feasign_randomly = false;


int32_t CtrSparseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }
  initialize_value();
  VLOG(0) << "initalize ctrSparseTable succ";
  return 0;
}

int32_t CtrSparseTable::initialize_value() {
  shard_values_.reserve(task_pool_size_);

  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard = std::make_shared<CtrValueBlock>();
    shard_values_.emplace_back(shard);
  }
  return 0;
}

int32_t CtrSparseTable::load(const std::string& path,
                                const std::string& param) {
//                                const std::string& prefix) {
  std::string table_path = table_dir(path);
  auto file_list = _afs_client.list(table_path);

  int load_param = atoi(param.c_str());
  auto expect_shard_num = _shard_num * task_pool_size_;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "CtrSparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "CtrSparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * shard_values_.size();

  size_t feature_value_size = _value_accesor->size() / sizeof(float);
  //TODO: multi-thread
  //int thread_num = shard_values_.size() < 15 ? shard_values_.size() : 15;
  //omp_set_num_threads(thread_num);
  //#pragma omp parallel for schedule(dynamic)
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
          auto* value = shard->Init(key);
          value->resize(feature_value_size);
          int parse_size = _value_accesor->parse_from_string(++end, value->data());
          value->resize(parse_size);
          //value->shrink_to_fit();
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR) << "CtrSparseTable load failed after read, retry it! path:"
                     << channel_config.path << " , retry_num=" << retry_num;
        }
      } catch(...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "CtrSparseTable load failed, retry it! path:" << channel_config.path
                   << " , retry_num=" << retry_num;
      }
      if (retry_num > FLAGS_pslib_table_save_max_retry) {
         LOG(ERROR) << "CtrSparseTable load failed reach max limit!";
         exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "CtrSparseTable load success, path from "
            << file_list[file_start_idx] << " to " << file_list[file_start_idx + task_pool_size_];
  return 0;
}

int32_t CtrSparseTable::load_local_fs(const std::string& path,
                                const std::string& param) {
  std::string table_path = table_dir(path);
  auto file_list = paddle::framework::localfs_list(table_path);

  int load_param = atoi(param.c_str());
  auto expect_shard_num = _shard_num * task_pool_size_;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "CtrSparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.size() == 0) {
    LOG(WARNING) << "CtrSparseTable load file is empty, path:" << path;
    return -1;
  }

  size_t file_start_idx = _shard_idx * shard_values_.size();

  size_t feature_value_size = _value_accesor->size() / sizeof(float);
  //int thread_num = shard_values_.size() < 15 ? shard_values_.size() : 15;
  //omp_set_num_threads(thread_num);
  //#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < task_pool_size_; ++ i) {
    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      std::ifstream file(file_list[file_start_idx + i]);
      char *end = NULL;
      auto& shard = shard_values_[i];
      try {
        while (std::getline(file, line_data) && line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto* value = shard->Init(key);
          value->resize(feature_value_size);
          int parse_size = _value_accesor->parse_from_string(++end, value->data());
          value->resize(parse_size);
          //value->shrink_to_fit();
        }
        file.close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR) << "CtrSparseTable load failed after read, retry it! path:"
                     << file_list[file_start_idx + i] << " , retry_num=" << retry_num;
        }
      } catch(...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "CtrSparseTable load failed, retry it! path:" << file_list[file_start_idx + i]
                   << " , retry_num=" << retry_num;
      }
      if (retry_num > FLAGS_pslib_table_save_max_retry) {
         LOG(ERROR) << "CtrSparseTable load failed reach max limit!";
         exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "CtrSparseTable load success, path from "
            << file_list[file_start_idx] << " to " << file_list[file_start_idx + task_pool_size_];
  return 0;
}

int32_t CtrSparseTable::save(const std::string& dirname,
                                const std::string& param) {
//                                const std::string& prefix) {
  int save_param = atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = table_dir(dirname);
  _afs_client.remove(paddle::string::format_string("%s/part-%03d-*", table_path.c_str(), _shard_idx));
  int thread_num = shard_values_.size() < 20 ? shard_values_.size() : 20;
  std::atomic<uint32_t> feasign_size_all{0};

  //TODO: openmp
  //omp_set_num_threads(thread_num);
  //#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < shard_values_.size(); ++i) {
    FsChannelConfig channel_config;
    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path = paddle::string::format_string("%s/part-%03d-%05d.gz",
        table_path.c_str(), _shard_idx, i);
    } else {
      channel_config.path = paddle::string::format_string("%s/part-%03d-%05d",
        table_path.c_str(), _shard_idx, i);
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
          if (_value_accesor->save(value.second->data(), save_param)) {
            std::string format_value = _value_accesor->
              parse_to_string(value.second->data(), value.second->size());
            if (0 != write_channel->write_line(
              paddle::string::format_string("%lu %s", value.first, format_value.c_str()))) {
                ++retry_num;
                is_write_failed = true;
                LOG(ERROR) << "CtrSparseTable save prefix failed, retry it! path:"
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
        LOG(ERROR) << "CtrSparseTable save prefix failed after write, retry it! "
                   << "path:" << channel_config.path << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
      if (retry_num > FLAGS_pslib_table_save_max_retry) { //TODO
        LOG(ERROR) << "CtrSparseTable save prefix failed reach max limit!";
        exit(-1);
      }
    } while (is_write_failed);
      feasign_size_all += feasign_size;
      for (auto& table : shard->values_) {
        for (auto& value : table) {
          _value_accesor->update_stat_after_save(value.second->data(), save_param);
        }
      }
    LOG(INFO) << "CtrSparseTable save prefix success, path:"
              << paddle::string::format_string("%s/%03d/part-%03d-", channel_config.path.c_str(), _config.table_id(), _shard_idx);
    }
    //int32 may overflow need to change return value
    return 0;
}

int32_t CtrSparseTable::save_local_fs(const std::string& dirname,
                                const std::string& param,
                                const std::string& prefix) {
  int save_param = atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2
  std::string table_path = table_dir(dirname);
  int feasign_cnt = 0;
  for (size_t i = 0; i < shard_values_.size(); ++i) {
    feasign_cnt = 0;
    auto& shard = shard_values_[i];
    std::string file_name = paddle::string::format_string("%s/part-%s-%03d-%05d", 
        table_path.c_str(), prefix.c_str(), _shard_idx, i);
    std::ofstream os;
    os.open(file_name);
    for (auto& table : shard->values_) {
      for (auto& value : table) {
        if (_value_accesor->save(value.second->data(), save_param)) {
          std::string format_value = _value_accesor->
            parse_to_string(value.second->data(), value.second->size());
          std::string out_line = paddle::string::format_string("%lu %s\n", value.first, format_value.c_str());
          //LOG(INFO) << out_line.c_str();
          os.write(out_line.c_str(), sizeof(char) * out_line.size());
          ++ feasign_cnt;
        }
      }
    }
    os.close();
    LOG(INFO) << "CtrSparseTable save prefix success, path:" << file_name << "feasign_cnt: " << feasign_cnt;
  }
  return 0;
}

std::pair<int64_t, int64_t> CtrSparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& shard : shard_values_) {
    for (auto& table : shard->values_) {
      feasign_size += table.size();
    }
  }

  return {feasign_size, mf_size};
}


int32_t CtrSparseTable::pull_sparse(float* pull_values,
                                       const PullSparseValue& pull_value) {
  auto shard_num = task_pool_size_;
  std::vector<std::future<int>> tasks(shard_num);

  size_t value_size = _value_accesor->size() / sizeof(float);
  size_t mf_value_size = _value_accesor->mf_size() / sizeof(float);
  size_t select_value_size = _value_accesor->select_size() / sizeof(float);
  //std::atomic<uint32_t> missed_keys{0};
  
  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, shard_num, &pull_value, &pull_values, value_size, mf_value_size, select_value_size]() -> int {
          auto& local_shard = shard_values_[shard_id];
          float data_buffer[value_size];
          float* data_buffer_ptr = data_buffer;

          std::vector<int> offsets;
          pull_value.Fission(shard_id, shard_num, &offsets);
          for (auto& offset : offsets) {
            uint64_t key = pull_value.feasigns_[offset];
            auto itr = local_shard->Find(key);
            size_t data_size = value_size - mf_value_size;
            if (itr == local_shard->end()) {
              //++missed_keys;
              //TODO: FLAGS
              if (FLAGS_pslib_create_value_when_push) {
                memset(data_buffer, 0, sizeof(float) * data_size);
              } else {
                 auto* feature_value = local_shard->Init(key);
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
            float* select_data = pull_values + select_value_size * offset;
            _value_accesor->select(&select_data, (const float**)&data_buffer_ptr, 1);
          }
          return 0;

        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  /*
  std::cout << "zcb debug table::pull_sparse";
  for (int i = 0; i < pull_value.numel_; ++ i) {
      std::cout << "key: " << i << ": " << pull_value.feasigns_[i];
      for (int j = 0; j < select_value_size; ++ j)
          std::cout << " value " << j << ": " << pull_values[i*select_value_size + j];
      std::cout << "\n";
  }
  std::cout << "zcb debug table::pull_sparse end";
  */
  return 0;
}

int32_t CtrSparseTable::pull_sparse_ptr(char** pull_values,
                                           const uint64_t* keys, size_t num) {
  return 0;
}

int32_t CtrSparseTable::push_sparse(const uint64_t* keys,
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
            const float* update_data = values + offset * update_value_col;
            auto itr = local_shard->Find(key);
            if (itr == local_shard->end()) {
              VLOG(0) << "zcb debug table push_sparse: " << key << "not found!" ;
              if (FLAGS_pslib_enable_create_feasign_randomly
                && !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard->Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(const_cast<float*>(feature_value->data()), data_buffer_ptr, value_size * sizeof(float));
              itr = local_shard->Find(key);
            } else {
              VLOG(1) << "zcb debug table push_sparse: " << key << " found!" ;
            }

            auto* feature_value = itr->second;
            float* value_data = const_cast<float*>(feature_value->data());
            size_t value_size = feature_value->size();

            /*
            std::cout << "push sparse, key: " << key << " value: ";
            for (int i = 0; i < value_size; ++ i)
                std::cout << value_data[i] << " ";
            std::cout << "\n";
            std::cout << "update_data: ";
            for (int i = 0; i < update_value_col; ++ i)
                std::cout << update_data[i] << " ";
            std::cout << "\n";
            */
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
            /*
            std::cout << "after update key:" << key << "\n";
            for(int i = 0; i < feature_value->size(); ++ i)
                std::cout << value_data[i] << " ";
            std::cout << "\n";
            */
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
int32_t CtrSparseTable::push_sparse(const uint64_t* keys,
                                       const float** values, size_t num) {
  _push_sparse(keys, values, num);
  return 0;
}

int32_t CtrSparseTable::_push_sparse(const uint64_t* keys,
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
            const float* update_data = values[offset];
            auto itr = local_shard->Find(key);
            if (itr == local_shard->end()) {
              if (FLAGS_pslib_enable_create_feasign_randomly
                && !_value_accesor->create_value(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto* feature_value = local_shard->Init(key);
              feature_value->resize(value_size);
              _value_accesor->create(&data_buffer_ptr, 1);
              memcpy(const_cast<float*>(feature_value->data()), data_buffer_ptr, value_size * sizeof(float));
              itr = local_shard->Find(key);
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

int32_t CtrSparseTable::flush() { return 0; }

//TODO: no need param
int32_t CtrSparseTable::shrink(const std::string& param) {
  //TODO implement with multi-thread
  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // shrink
    auto& shard = shard_values_[shard_id];
    for (auto& table : shard->values_) {
      for (auto iter = table.begin(); iter != table.end();) {
        if (_value_accesor->shrink(iter->second->data())) {
          butil::return_object(iter->second);
          iter = table.erase(iter);
        } else {
          ++ iter;
        }
      }
    }
  }
  return 0;
}

void CtrSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
