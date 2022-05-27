// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/ssd_sparse_table.h"
#include "paddle/fluid/distributed/common/cost_timer.h"
#include "paddle/fluid/distributed/common/local_random.h"
#include "paddle/fluid/distributed/common/topk_calculator.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/utils/string/string_helper.h"

DECLARE_bool(pserver_print_missed_key_num_every_push);
DECLARE_bool(pserver_create_value_when_push);
DECLARE_bool(pserver_enable_create_feasign_randomly);
DEFINE_bool(pserver_open_strict_check, false, "pserver_open_strict_check");
DEFINE_string(rocksdb_path, "database", "path of sparse table rocksdb file");
DEFINE_int32(pserver_load_batch_size, 5000, "load batch size for ssd");

namespace paddle {
namespace distributed {

int32_t SSDSparseTable::Initialize() {
  MemorySparseTable::Initialize();
  _db = paddle::distributed::RocksDBHandler::GetInstance();
  _db->initialize(FLAGS_rocksdb_path, _real_local_shard_num);
  return 0;
}

int32_t SSDSparseTable::InitializeShard() { return 0; }

int32_t SSDSparseTable::PullSparse(float* pull_values, const uint64_t* keys,
                                   size_t num) {
  CostTimer timer("pserver_downpour_sparse_select_all");
  size_t value_size = _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  size_t select_value_size =
      _value_accesor->GetAccessorInfo().select_size / sizeof(float);

  {  // 从table取值 or create
    std::vector<std::future<int>> tasks(_real_local_shard_num);
    std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
        _real_local_shard_num);
    for (size_t i = 0; i < num; ++i) {
      int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
      task_keys[shard_id].push_back({keys[i], i});
    }

    std::atomic<uint32_t> missed_keys{0};
    for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
      tasks[shard_id] =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this, shard_id, &task_keys, value_size, mf_value_size,
               select_value_size, pull_values, keys, &missed_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_size];
                float* data_buffer_ptr = data_buffer;
                for (int i = 0; i < keys.size(); ++i) {
                  uint64_t key = keys[i].first;
                  auto itr = local_shard.find(key);
                  size_t data_size = value_size - mf_value_size;
                  if (itr == local_shard.end()) {
                    // pull rocksdb
                    std::string tmp_string("");
                    if (_db->get(shard_id, (char*)&key, sizeof(uint64_t),
                                 tmp_string) > 0) {
                      ++missed_keys;
                      if (FLAGS_pserver_create_value_when_push) {
                        memset(data_buffer, 0, sizeof(float) * data_size);
                      } else {
                        auto& feature_value = local_shard[key];
                        feature_value.resize(data_size);
                        float* data_ptr =
                            const_cast<float*>(feature_value.data());
                        _value_accesor->Create(&data_buffer_ptr, 1);
                        memcpy(data_ptr, data_buffer_ptr,
                               data_size * sizeof(float));
                      }
                    } else {
                      data_size = tmp_string.size() / sizeof(float);
                      memcpy(data_buffer_ptr,
                             paddle::string::str_to_float(tmp_string),
                             data_size * sizeof(float));
                      // from rocksdb to mem
                      auto& feature_value = local_shard[key];
                      feature_value.resize(data_size);
                      memcpy(const_cast<float*>(feature_value.data()),
                             data_buffer_ptr, data_size * sizeof(float));
                      _db->del_data(shard_id, (char*)&key, sizeof(uint64_t));
                    }
                  } else {
                    data_size = itr.value().size();
                    memcpy(data_buffer_ptr, itr.value().data(),
                           data_size * sizeof(float));
                  }
                  for (int mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
                    data_buffer[mf_idx] = 0.0;
                  }
                  int pull_data_idx = keys[i].second;
                  float* select_data =
                      pull_values + pull_data_idx * select_value_size;
                  _value_accesor->Select(&select_data,
                                         (const float**)&data_buffer_ptr, 1);
                }
                return 0;
              });
    }
    for (size_t i = 0; i < _real_local_shard_num; ++i) {
      tasks[i].wait();
    }
    if (FLAGS_pserver_print_missed_key_num_every_push) {
      LOG(WARNING) << "total pull keys:" << num
                   << " missed_keys:" << missed_keys.load();
    }
  }
  return 0;
}

int32_t SSDSparseTable::PushSparse(const uint64_t* keys, const float* values,
                                   size_t num) {
  CostTimer timer("pserver_downpour_sparse_update_all");
  // 构造value push_value的数据指针
  size_t value_col = _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_col =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  size_t update_value_col =
      _value_accesor->GetAccessorInfo().update_size / sizeof(float);
  {
    std::vector<std::future<int>> tasks(_real_local_shard_num);
    std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
        _real_local_shard_num);
    for (size_t i = 0; i < num; ++i) {
      int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
      task_keys[shard_id].push_back({keys[i], i});
    }
    for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
      tasks[shard_id] =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this, shard_id, value_col, mf_value_col, update_value_col,
               values, &task_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_col];
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
                    memcpy(const_cast<float*>(feature_value.data()),
                           data_buffer_ptr, value_size * sizeof(float));
                    itr = local_shard.find(key);
                  }
                  auto& feature_value = itr.value();
                  float* value_data = const_cast<float*>(feature_value.data());
                  size_t value_size = feature_value.size();

                  if (value_size ==
                      value_col) {  // 已拓展到最大size, 则就地update
                    _value_accesor->Update(&value_data, &update_data, 1);
                  } else {  // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
                    memcpy(data_buffer_ptr, value_data,
                           value_size * sizeof(float));
                    _value_accesor->Update(&data_buffer_ptr, &update_data, 1);
                    if (_value_accesor->NeedExtendMF(data_buffer)) {
                      feature_value.resize(value_col);
                      value_data = const_cast<float*>(feature_value.data());
                      _value_accesor->Create(&value_data, 1);
                    }
                    memcpy(value_data, data_buffer_ptr,
                           value_size * sizeof(float));
                  }
                }
                return 0;
              });
    }
    for (size_t i = 0; i < _real_local_shard_num; ++i) {
      tasks[i].wait();
    }
  }
  /*
  //update && value 的转置
  thread_local Eigen::MatrixXf update_matrix;
  float* transposed_update_data[update_value_col];
  make_matrix_with_eigen(num, update_value_col, update_matrix,
  transposed_update_data);
  copy_array_to_eigen(values, update_matrix);

  thread_local Eigen::MatrixXf value_matrix;
  float* transposed_value_data[value_col];
  make_matrix_with_eigen(num, value_col, value_matrix, transposed_value_data);
  copy_matrix_to_eigen((const float**)(value_ptrs->data()), value_matrix);

  //批量update
  {
      CostTimer accessor_timer("pslib_downpour_sparse_update_accessor");
      _value_accesor->update(transposed_value_data, (const
  float**)transposed_update_data, num);
  }
  copy_eigen_to_matrix(value_matrix, value_ptrs->data());
  */
  return 0;
}

int32_t SSDSparseTable::Shrink(const std::string& param) {
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    uint64_t mem_count = 0;
    uint64_t ssd_count = 0;

    LOG(INFO) << "SSDSparseTable begin shrink shard:" << i;
    auto& shard = _local_shards[i];
    for (auto it = shard.begin(); it != shard.end();) {
      if (_value_accesor->Shrink(it.value().data())) {
        it = shard.erase(it);
        mem_count++;
      } else {
        ++it;
      }
    }
    auto* it = _db->get_iterator(i);
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      if (_value_accesor->Shrink(
              paddle::string::str_to_float(it->value().data()))) {
        _db->del_data(i, it->key().data(), it->key().size());
        ssd_count++;
      } else {
        _db->put(i, it->key().data(), it->key().size(), it->value().data(),
                 it->value().size());
      }
    }
    delete it;
    LOG(INFO) << "SSDSparseTable shrink success. shard:" << i << " delete MEM["
              << mem_count << "] SSD[" << ssd_count << "]";
    //_db->flush(i);
  }
  return 0;
}

int32_t SSDSparseTable::UpdateTable() {
  // TODO implement with multi-thread
  int count = 0;
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    auto& shard = _local_shards[i];
    // from mem to ssd
    for (auto it = shard.begin(); it != shard.end();) {
      if (_value_accesor->SaveSSD(it.value().data())) {
        _db->put(i, (char*)&it.key(), sizeof(uint64_t),
                 (char*)it.value().data(), it.value().size() * sizeof(float));
        count++;
        it = shard.erase(it);
      } else {
        ++it;
      }
    }
    _db->flush(i);
  }
  LOG(INFO) << "Table>> update count: " << count;
  return 0;
}

int64_t SSDSparseTable::LocalSize() {
  int64_t local_size = 0;
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    local_size += _local_shards[i].size();
  }
  // TODO rocksdb size
  uint64_t ssd_size = 0;
  // _db->get_estimate_key_num(ssd_size);
  // return local_size + ssd_size;
  return local_size;
}

int32_t SSDSparseTable::Save(const std::string& path,
                             const std::string& param) {
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  //    if (save_param == 5) {
  //        return save_patch(path, save_param);
  //    }

  // LOG(INFO) << "table cache rate is: " << _config.sparse_table_cache_rate();
  LOG(INFO) << "table cache rate is: " << _config.sparse_table_cache_rate();
  LOG(INFO) << "enable_sparse_table_cache: "
            << _config.enable_sparse_table_cache();
  LOG(INFO) << "LocalSize: " << LocalSize();
  if (_config.enable_sparse_table_cache()) {
    LOG(INFO) << "Enable sparse table cache, top n:" << _cache_tk_size;
  }
  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, _cache_tk_size);
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  std::string table_path = TableDir(path);
  _afs_client.remove(paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;

  // std::atomic<uint32_t> feasign_size;
  std::atomic<uint32_t> feasign_size_all{0};
  // feasign_size = 0;

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
    int err_no = 0;
    int retry_num = 0;
    bool is_write_failed = false;
    int feasign_size = 0;
    auto& shard = _local_shards[i];
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2) &&
            _value_accesor->Save(it.value().data(), 4)) {
          // tk.push(i, it.value().data()[2]);
          tk.push(i, _value_accesor->GetField(it.value().data(), "show"));
        }
        if (_value_accesor->Save(it.value().data(), save_param)) {
          std::string format_value = _value_accesor->ParseToString(
              it.value().data(), it.value().size());
          if (0 !=
              write_channel->write_line(paddle::string::format_string(
                  "%lu %s", it.key(), format_value.c_str()))) {
            ++retry_num;
            is_write_failed = true;
            LOG(ERROR) << "SSDSparseTable save failed, retry it! path:"
                       << channel_config.path << ", retry_num=" << retry_num;
            break;
          }
          ++feasign_size;
        }
      }

      if (err_no == -1 && !is_write_failed) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR) << "SSDSparseTable save failed after write, retry it! "
                   << "path:" << channel_config.path
                   << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
        continue;
      }

      // delta and cache and revert is all in mem, base in rocksdb
      if (save_param != 1) {
        auto* it = _db->get_iterator(i);
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
          bool need_save = _value_accesor->Save(
              paddle::string::str_to_float(it->value().data()), save_param);
          _value_accesor->UpdateStatAfterSave(
              paddle::string::str_to_float(it->value().data()), save_param);
          if (need_save) {
            std::string format_value = _value_accesor->ParseToString(
                paddle::string::str_to_float(it->value().data()),
                it->value().size() / sizeof(float));
            if (0 !=
                write_channel->write_line(paddle::string::format_string(
                    "%lu %s", *((uint64_t*)const_cast<char*>(it->key().data())),
                    format_value.c_str()))) {
              ++retry_num;
              is_write_failed = true;
              LOG(ERROR) << "SSDSparseTable save failed, retry it! path:"
                         << channel_config.path << ", retry_num=" << retry_num;
              break;
            }
            if (save_param == 3) {
              _db->put(i, it->key().data(), it->key().size(),
                       it->value().data(), it->value().size());
            }
            ++feasign_size;
          }
        }
        delete it;
      }

      write_channel->close();
      if (err_no == -1) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR) << "SSDSparseTable save failed after write, retry it! "
                   << "path:" << channel_config.path
                   << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
    } while (is_write_failed);
    feasign_size_all += feasign_size;
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      _value_accesor->UpdateStatAfterSave(it.value().data(), save_param);
    }
  }
  if (save_param == 3) {
    UpdateTable();
    _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
    LOG(INFO) << "SSDSparseTable update success.";
  }
  LOG(INFO) << "SSDSparseTable save success, path:"
            << paddle::string::format_string("%s/%03d/part-%03d-", path.c_str(),
                                             _config.table_id(), _shard_idx)
            << " from " << file_start_idx << " to "
            << file_start_idx + _real_local_shard_num - 1;
  // return feasign_size_all;
  _local_show_threshold = tk.top();
  LOG(INFO) << "local cache threshold: " << _local_show_threshold;
  // int32 may overflow need to change return value
  return 0;
}

int64_t SSDSparseTable::CacheShuffle(
    const std::string& path, const std::string& param, double cache_threshold,
    std::function<std::future<int32_t>(int msg_type, int to_pserver_id,
                                       std::string& msg)>
        send_msg_func,
    paddle::framework::Channel<std::pair<uint64_t, std::string>>&
        shuffled_channel,
    const std::vector<Table*>& table_ptrs) {
  LOG(INFO) << "cache shuffle with cache threshold: " << cache_threshold
            << " param:" << param;
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  if (!_config.enable_sparse_table_cache() || cache_threshold < 0) {
    LOG(WARNING)
        << "cache shuffle failed not enable table cache or cache threshold < 0 "
        << _config.enable_sparse_table_cache() << " or " << cache_threshold;
    // return -1;
  }
  int shuffle_node_num = _config.sparse_table_cache_file_num();
  LOG(INFO) << "Table>> shuffle node num is: " << shuffle_node_num;
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;

  std::vector<
      paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>>
      writers(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, std::string>>> datas(
      _real_local_shard_num);

  int feasign_size = 0;
  std::vector<paddle::framework::Channel<std::pair<uint64_t, std::string>>>
      tmp_channels;
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    tmp_channels.push_back(
        paddle::framework::MakeChannel<std::pair<uint64_t, std::string>>());
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < _real_local_shard_num; ++i) {
    paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>& writer =
        writers[i];
    //    std::shared_ptr<paddle::framework::ChannelObject<std::pair<uint64_t,
    //    std::string>>> tmp_chan =
    //        paddle::framework::MakeChannel<std::pair<uint64_t,
    //        std::string>>();
    writer.Reset(tmp_channels[i].get());

    auto& shard = _local_shards[i];
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      if (_value_accesor->SaveCache(it.value().data(), save_param,
                                    cache_threshold)) {
        std::string format_value =
            _value_accesor->ParseToString(it.value().data(), it.value().size());
        std::pair<uint64_t, std::string> pkv(it.key(), format_value.c_str());
        writer << pkv;
        ++feasign_size;
      }
    }

    writer.Flush();
    writer.channel()->Close();
  }
  LOG(INFO) << "SSDSparseTable cache KV save success to Channel feasigh size: "
            << feasign_size
            << " and start sparse cache data shuffle real local shard num: "
            << _real_local_shard_num;
  std::vector<std::pair<uint64_t, std::string>> local_datas;
  for (size_t idx_shard = 0; idx_shard < _real_local_shard_num; ++idx_shard) {
    paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>& writer =
        writers[idx_shard];
    auto channel = writer.channel();
    std::vector<std::pair<uint64_t, std::string>>& data = datas[idx_shard];
    std::vector<paddle::framework::BinaryArchive> ars(shuffle_node_num);
    while (channel->Read(data)) {
      for (auto& t : data) {
        auto pserver_id =
            paddle::distributed::local_random_engine()() % shuffle_node_num;
        if (pserver_id != _shard_idx) {
          ars[pserver_id] << t;
        } else {
          local_datas.emplace_back(std::move(t));
        }
      }
      std::vector<std::future<int32_t>> total_status;
      std::vector<uint32_t> send_data_size(shuffle_node_num, 0);
      std::vector<int> send_index(shuffle_node_num);
      for (int i = 0; i < shuffle_node_num; ++i) {
        send_index[i] = i;
      }
      std::random_shuffle(send_index.begin(), send_index.end());
      for (auto index = 0u; index < shuffle_node_num; ++index) {
        int i = send_index[index];
        if (i == _shard_idx) {
          continue;
        }
        if (ars[i].Length() == 0) {
          continue;
        }
        std::string msg(ars[i].Buffer(), ars[i].Length());
        auto ret = send_msg_func(101, i, msg);
        total_status.push_back(std::move(ret));
        send_data_size[i] += ars[i].Length();
      }
      for (auto& t : total_status) {
        t.wait();
      }
      ars.clear();
      ars = std::vector<paddle::framework::BinaryArchive>(shuffle_node_num);
      data = std::vector<std::pair<uint64_t, std::string>>();
    }
  }
  shuffled_channel->Write(std::move(local_datas));
  LOG(INFO) << "cache shuffle finished";
  return 0;
}

int32_t SSDSparseTable::SaveCache(
    const std::string& path, const std::string& param,
    paddle::framework::Channel<std::pair<uint64_t, std::string>>&
        shuffled_channel) {
  if (_shard_idx >= _config.sparse_table_cache_file_num()) {
    return 0;
  }
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  std::string table_path = paddle::string::format_string(
      "%s/%03d_cache/", path.c_str(), _config.table_id());
  _afs_client.remove(paddle::string::format_string(
      "%s/part-%03d", table_path.c_str(), _shard_idx));
  uint32_t feasign_size = 0;
  FsChannelConfig channel_config;
  // not compress cache model
  channel_config.path = paddle::string::format_string(
      "%s/part-%03d", table_path.c_str(), _shard_idx);
  channel_config.converter = _value_accesor->Converter(save_param).converter;
  channel_config.deconverter =
      _value_accesor->Converter(save_param).deconverter;
  auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40);
  std::vector<std::pair<uint64_t, std::string>> data;
  bool is_write_failed = false;
  shuffled_channel->Close();
  while (shuffled_channel->Read(data)) {
    for (auto& t : data) {
      ++feasign_size;
      if (0 !=
          write_channel->write_line(paddle::string::format_string(
              "%lu %s", t.first, t.second.c_str()))) {
        LOG(ERROR) << "Cache Table save failed, "
                      "path:"
                   << channel_config.path << ", retry it!";
        is_write_failed = true;
        break;
      }
    }
    data = std::vector<std::pair<uint64_t, std::string>>();
  }
  if (is_write_failed) {
    _afs_client.remove(channel_config.path);
  }
  write_channel->close();
  LOG(INFO) << "SSDSparseTable cache save success, feasign: " << feasign_size
            << ", path: " << channel_config.path;
  shuffled_channel->Open();
  return feasign_size;
}

int32_t SSDSparseTable::Load(const std::string& path,
                             const std::string& param) {
  return MemorySparseTable::Load(path, param);
}

//加载path目录下数据[start_idx, end_idx)
int32_t SSDSparseTable::Load(size_t start_idx, size_t end_idx,
                             const std::vector<std::string>& file_list,
                             const std::string& param) {
  if (start_idx >= file_list.size()) {
    return 0;
  }
  int load_param = atoi(param.c_str());
  size_t feature_value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);

  end_idx =
      end_idx < _sparse_table_shard_num ? end_idx : _sparse_table_shard_num;
  int thread_num = (end_idx - start_idx) < 20 ? (end_idx - start_idx) : 20;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = start_idx; i < end_idx; ++i) {
    FsChannelConfig channel_config;
    channel_config.path = file_list[i];
    channel_config.converter = _value_accesor->Converter(load_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(load_param).deconverter;

    int retry_num = 0;
    int err_no = 0;
    bool is_read_failed = false;
    std::vector<std::pair<char*, int>> ssd_keys;
    std::vector<std::pair<char*, int>> ssd_values;
    std::vector<uint64_t> tmp_key;
    ssd_keys.reserve(FLAGS_pserver_load_batch_size);
    ssd_values.reserve(FLAGS_pserver_load_batch_size);
    tmp_key.reserve(FLAGS_pserver_load_batch_size);
    do {
      ssd_keys.clear();
      ssd_values.clear();
      tmp_key.clear();
      err_no = 0;
      is_read_failed = false;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char* end = NULL;
      int local_shard_id = i % _avg_local_shard_num;
      auto& shard = _local_shards[local_shard_id];
      float data_buffer[FLAGS_pserver_load_batch_size * feature_value_size];
      float* data_buffer_ptr = data_buffer;
      uint64_t mem_count = 0;
      uint64_t ssd_count = 0;
      uint64_t mem_mf_count = 0;
      uint64_t ssd_mf_count = 0;
      try {
        while (read_channel->read_line(line_data) == 0 &&
               line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          if (FLAGS_pserver_open_strict_check) {
            if (key % _sparse_table_shard_num != i) {
              LOG(WARNING) << "SSDSparseTable key:" << key
                           << " not match shard,"
                           << " file_idx:" << i
                           << " shard num:" << _sparse_table_shard_num
                           << " file:" << channel_config.path;
              continue;
            }
          }
          int value_size =
              _value_accesor->ParseFromString(++end, data_buffer_ptr);
          // ssd or mem
          if (_value_accesor->SaveSSD(data_buffer_ptr)) {
            tmp_key.emplace_back(key);
            ssd_keys.emplace_back(
                std::make_pair((char*)&tmp_key.back(), sizeof(uint64_t)));
            ssd_values.emplace_back(std::make_pair((char*)data_buffer_ptr,
                                                   value_size * sizeof(float)));
            data_buffer_ptr += feature_value_size;
            if (ssd_keys.size() == FLAGS_pserver_load_batch_size) {
              _db->put_batch(local_shard_id, ssd_keys, ssd_values,
                             ssd_keys.size());
              ssd_keys.clear();
              ssd_values.clear();
              tmp_key.clear();
              data_buffer_ptr = data_buffer;
            }
            ssd_count++;
            if (value_size > feature_value_size - mf_value_size) {
              ssd_mf_count++;
            }
          } else {
            auto& value = shard[key];
            value.resize(value_size);
            _value_accesor->ParseFromString(end, value.data());
            mem_count++;
            if (value_size > feature_value_size - mf_value_size) {
              mem_mf_count++;
            }
          }
        }
        // last batch
        if (ssd_keys.size() > 0) {
          _db->put_batch(local_shard_id, ssd_keys, ssd_values, ssd_keys.size());
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR) << "SSDSparseTable load failed after read, retry it! path:"
                     << channel_config.path << " , retry_num=" << retry_num;
          continue;
        }

        _db->flush(local_shard_id);
        LOG(INFO) << "Table>> load done. ALL[" << mem_count + ssd_count
                  << "] MEM[" << mem_count << "] MEM_MF[" << mem_mf_count
                  << "] SSD[" << ssd_count << "] SSD_MF[" << ssd_mf_count
                  << "].";
      } catch (...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "SSDSparseTable load failed after read, retry it! path:"
                   << channel_config.path << " , retry_num=" << retry_num;
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "load num:" << LocalSize();
  LOG(INFO) << "SSDSparseTable load success, path from " << file_list[start_idx]
            << " to " << file_list[end_idx - 1];

  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  return 0;
}

}  // namespace distributed
}  // namespace paddle
