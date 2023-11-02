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
#include "paddle/fluid/platform/flags.h"
#include "paddle/utils/string/string_helper.h"
PD_DECLARE_bool(pserver_print_missed_key_num_every_push);
PD_DECLARE_bool(pserver_create_value_when_push);
PD_DECLARE_bool(pserver_enable_create_feasign_randomly);
PD_DEFINE_bool(pserver_open_strict_check, false, "pserver_open_strict_check");
PD_DEFINE_int32(pserver_load_batch_size, 5000, "load batch size for ssd");
PADDLE_DEFINE_EXPORTED_string(rocksdb_path,
                              "database",
                              "path of sparse table rocksdb file");

namespace paddle {
namespace distributed {

int32_t SSDSparseTable::Initialize() {
  MemorySparseTable::Initialize();
  _db = ::paddle::distributed::RocksDBHandler::GetInstance();
  _db->initialize(FLAGS_rocksdb_path, _real_local_shard_num);
  VLOG(0) << "initalize SSDSparseTable succ";
  VLOG(0) << "SSD FLAGS_pserver_print_missed_key_num_every_push:"
          << FLAGS_pserver_print_missed_key_num_every_push;
  return 0;
}

int32_t SSDSparseTable::InitializeShard() { return 0; }

int32_t SSDSparseTable::Pull(TableContext& context) {
  CHECK(context.value_type == Sparse);
  if (context.use_ptr) {
    char** pull_values = context.pull_context.ptr_values;
    const uint64_t* keys = context.pull_context.keys;
    return PullSparsePtr(
        context.shard_id, pull_values, keys, context.num, context.pass_id);
  } else {
    float* pull_values = context.pull_context.values;
    const PullSparseValue& pull_value = context.pull_context.pull_value;
    return PullSparse(pull_values, pull_value.feasigns_, pull_value.numel_);
  }
}

int32_t SSDSparseTable::Push(TableContext& context) {
  CHECK(context.value_type == Sparse);
  if (context.use_ptr) {
    return PushSparse(context.push_context.keys,
                      context.push_context.ptr_values,
                      context.num);
  } else {
    const uint64_t* keys = context.push_context.keys;
    const float* values = context.push_context.values;
    size_t num = context.num;
    return PushSparse(keys, values, num);
  }
}

int32_t SSDSparseTable::PullSparse(float* pull_values,
                                   const uint64_t* keys,
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
      task_keys[shard_id].emplace_back(keys[i], i);
    }

    std::atomic<uint32_t> missed_keys{0};
    for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
      tasks[shard_id] =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this,
               shard_id,
               &task_keys,
               value_size,
               mf_value_size,
               select_value_size,
               pull_values,
               keys,
               &missed_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_size];  // NOLINT
                float* data_buffer_ptr = data_buffer;
                for (size_t i = 0; i < keys.size(); ++i) {
                  uint64_t key = keys[i].first;
                  auto itr = local_shard.find(key);
                  size_t data_size = value_size - mf_value_size;
                  if (itr == local_shard.end()) {
                    // pull rocksdb
                    std::string tmp_string("");
                    if (_db->get(shard_id,
                                 reinterpret_cast<char*>(&key),
                                 sizeof(uint64_t),
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
                        memcpy(data_ptr,
                               data_buffer_ptr,
                               data_size * sizeof(float));
                      }
                    } else {
                      data_size = tmp_string.size() / sizeof(float);
                      memcpy(data_buffer_ptr,
                             ::paddle::string::str_to_float(tmp_string),
                             data_size * sizeof(float));
                      // from rocksdb to mem
                      auto& feature_value = local_shard[key];
                      feature_value.resize(data_size);
                      memcpy(const_cast<float*>(feature_value.data()),
                             data_buffer_ptr,
                             data_size * sizeof(float));
                      _db->del_data(shard_id,
                                    reinterpret_cast<char*>(&key),
                                    sizeof(uint64_t));
                    }
                  } else {
                    data_size = itr.value().size();
                    memcpy(data_buffer_ptr,
                           itr.value().data(),
                           data_size * sizeof(float));
                  }
                  for (size_t mf_idx = data_size; mf_idx < value_size;
                       ++mf_idx) {
                    data_buffer[mf_idx] = 0.0;
                  }
                  int pull_data_idx = keys[i].second;
                  float* select_data =
                      pull_values + pull_data_idx * select_value_size;
                  _value_accesor->Select(
                      &select_data, (const float**)&data_buffer_ptr, 1);
                }
                return 0;
              });
    }
    for (int i = 0; i < _real_local_shard_num; ++i) {
      tasks[i].wait();
    }
    if (FLAGS_pserver_print_missed_key_num_every_push) {
      LOG(WARNING) << "total pull keys:" << num
                   << " missed_keys:" << missed_keys.load();
    }
  }
  return 0;
}

int32_t SSDSparseTable::PullSparsePtr(int shard_id,
                                      char** pull_values,
                                      const uint64_t* pull_keys,
                                      size_t num,
                                      uint16_t pass_id) {
  CostTimer timer("pserver_ssd_sparse_select_all");
  size_t value_size = _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);

  {  // 从table取值 or create
    RocksDBCtx context;
    std::vector<std::future<int>> tasks;
    RocksDBItem* cur_ctx = context.switch_item();
    cur_ctx->reset();
    FixedFeatureValue* ret = NULL;
    auto& local_shard = _local_shards[shard_id];
    float data_buffer[value_size];  // NOLINT
    float* data_buffer_ptr = data_buffer;

    for (size_t i = 0; i < num; ++i) {
      uint64_t key = pull_keys[i];
      auto itr = local_shard.find(key);
      if (itr == local_shard.end()) {
        cur_ctx->batch_index.push_back(i);
        cur_ctx->batch_keys.emplace_back(
            reinterpret_cast<const char*>(&(pull_keys[i])), sizeof(uint64_t));
        if (cur_ctx->batch_keys.size() == 1024) {
          cur_ctx->batch_values.resize(cur_ctx->batch_keys.size());
          cur_ctx->status.resize(cur_ctx->batch_keys.size());
          auto fut =
              _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
                  [this, shard_id, cur_ctx]() -> int {
                    _db->multi_get(shard_id,
                                   cur_ctx->batch_keys.size(),
                                   cur_ctx->batch_keys.data(),
                                   cur_ctx->batch_values.data(),
                                   cur_ctx->status.data());
                    return 0;
                  });
          cur_ctx = context.switch_item();
          for (size_t x = 0; x < tasks.size(); ++x) {
            tasks[x].wait();
            for (size_t idx = 0; idx < cur_ctx->status.size(); idx++) {
              uint64_t cur_key = *(reinterpret_cast<uint64_t*>(
                  const_cast<char*>(cur_ctx->batch_keys[idx].data())));
              if (cur_ctx->status[idx].IsNotFound()) {
                auto& feature_value = local_shard[cur_key];
                int init_size = value_size - mf_value_size;
                feature_value.resize(init_size);
                _value_accesor->Create(&data_buffer_ptr, 1);
                memcpy(const_cast<float*>(feature_value.data()),
                       data_buffer_ptr,
                       init_size * sizeof(float));
                ret = &feature_value;
              } else {
                int data_size =
                    cur_ctx->batch_values[idx].size() / sizeof(float);
                // from rocksdb to mem
                auto& feature_value = local_shard[cur_key];
                feature_value.resize(data_size);
                memcpy(const_cast<float*>(feature_value.data()),
                       ::paddle::string::str_to_float(
                           cur_ctx->batch_values[idx].data()),
                       data_size * sizeof(float));
                _db->del_data(shard_id,
                              reinterpret_cast<char*>(&cur_key),
                              sizeof(uint64_t));
                ret = &feature_value;
              }
              _value_accesor->UpdatePassId(ret->data(), pass_id);
              int pull_data_idx = cur_ctx->batch_index[idx];
              pull_values[pull_data_idx] = reinterpret_cast<char*>(ret);
            }
          }
          cur_ctx->reset();
          tasks.clear();
          tasks.push_back(std::move(fut));
        }
      } else {
        ret = itr.value_ptr();
        // int pull_data_idx = keys[i].second;
        _value_accesor->UpdatePassId(ret->data(), pass_id);
        pull_values[i] = reinterpret_cast<char*>(ret);
      }
    }
    if (!cur_ctx->batch_keys.empty()) {
      cur_ctx->batch_values.resize(cur_ctx->batch_keys.size());
      cur_ctx->status.resize(cur_ctx->batch_keys.size());
      auto fut =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this, shard_id, cur_ctx]() -> int {
                _db->multi_get(shard_id,
                               cur_ctx->batch_keys.size(),
                               cur_ctx->batch_keys.data(),
                               cur_ctx->batch_values.data(),
                               cur_ctx->status.data());
                return 0;
              });
      tasks.push_back(std::move(fut));
    }
    for (size_t x = 0; x < tasks.size(); ++x) {
      tasks[x].wait();
    }
    for (size_t x = 0; x < 2; x++) {
      cur_ctx = context.switch_item();
      for (size_t idx = 0; idx < cur_ctx->status.size(); idx++) {
        uint64_t cur_key = *(reinterpret_cast<uint64_t*>(
            const_cast<char*>(cur_ctx->batch_keys[idx].data())));
        if (cur_ctx->status[idx].IsNotFound()) {
          auto& feature_value = local_shard[cur_key];
          int init_size = value_size - mf_value_size;
          feature_value.resize(init_size);
          _value_accesor->Create(&data_buffer_ptr, 1);
          memcpy(const_cast<float*>(feature_value.data()),
                 data_buffer_ptr,
                 init_size * sizeof(float));
          ret = &feature_value;
        } else {
          int data_size = cur_ctx->batch_values[idx].size() / sizeof(float);
          // from rocksdb to mem
          auto& feature_value = local_shard[cur_key];
          feature_value.resize(data_size);
          memcpy(
              const_cast<float*>(feature_value.data()),
              ::paddle::string::str_to_float(cur_ctx->batch_values[idx].data()),
              data_size * sizeof(float));
          _db->del_data(
              shard_id, reinterpret_cast<char*>(&cur_key), sizeof(uint64_t));
          ret = &feature_value;
        }
        _value_accesor->UpdatePassId(ret->data(), pass_id);
        int pull_data_idx = cur_ctx->batch_index[idx];
        pull_values[pull_data_idx] = reinterpret_cast<char*>(ret);
      }
      cur_ctx->reset();
    }
  }
  return 0;
}

int32_t SSDSparseTable::PushSparse(const uint64_t* keys,
                                   const float* values,
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
      task_keys[shard_id].emplace_back(keys[i], i);
    }
    for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
      tasks[shard_id] =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this,
               shard_id,
               value_col,
               mf_value_col,
               update_value_col,
               values,
               &task_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_col];  // NOLINT
                float* data_buffer_ptr = data_buffer;
                for (size_t i = 0; i < keys.size(); ++i) {
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
                           data_buffer_ptr,
                           value_size * sizeof(float));
                    itr = local_shard.find(key);
                  }
                  auto& feature_value = itr.value();
                  float* value_data = const_cast<float*>(feature_value.data());
                  size_t value_size = feature_value.size();

                  if (value_size ==
                      value_col) {  // 已拓展到最大size, 则就地update
                    _value_accesor->Update(&value_data, &update_data, 1);
                  } else {
                    // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
                    memcpy(data_buffer_ptr,
                           value_data,
                           value_size * sizeof(float));
                    _value_accesor->Update(&data_buffer_ptr, &update_data, 1);
                    if (_value_accesor->NeedExtendMF(data_buffer)) {
                      feature_value.resize(value_col);
                      value_data = const_cast<float*>(feature_value.data());
                      _value_accesor->Create(&value_data, 1);
                    }
                    memcpy(value_data,
                           data_buffer_ptr,
                           value_size * sizeof(float));
                  }
                }
                return 0;
              });
    }
    for (int i = 0; i < _real_local_shard_num; ++i) {
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

int32_t SSDSparseTable::PushSparse(const uint64_t* keys,
                                   const float** values,
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
      task_keys[shard_id].emplace_back(keys[i], i);
    }
    for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
      tasks[shard_id] =
          _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
              [this,
               shard_id,
               value_col,
               mf_value_col,
               update_value_col,
               values,
               &task_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_col];  // NOLINT
                float* data_buffer_ptr = data_buffer;
                for (size_t i = 0; i < keys.size(); ++i) {
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
                    memcpy(const_cast<float*>(feature_value.data()),
                           data_buffer_ptr,
                           value_size * sizeof(float));
                    itr = local_shard.find(key);
                  }
                  auto& feature_value = itr.value();
                  float* value_data = const_cast<float*>(feature_value.data());
                  size_t value_size = feature_value.size();

                  if (value_size ==
                      value_col) {  // 已拓展到最大size, 则就地update
                    _value_accesor->Update(&value_data, &update_data, 1);
                  } else {
                    // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
                    memcpy(data_buffer_ptr,
                           value_data,
                           value_size * sizeof(float));
                    _value_accesor->Update(&data_buffer_ptr, &update_data, 1);
                    if (_value_accesor->NeedExtendMF(data_buffer)) {
                      feature_value.resize(value_col);
                      value_data = const_cast<float*>(feature_value.data());
                      _value_accesor->Create(&value_data, 1);
                    }
                    memcpy(value_data,
                           data_buffer_ptr,
                           value_size * sizeof(float));
                  }
                }
                return 0;
              });
    }
    for (int i = 0; i < _real_local_shard_num; ++i) {
      tasks[i].wait();
    }
  }
  return 0;
}

int32_t SSDSparseTable::Shrink(const std::string& param) {
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
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
              ::paddle::string::str_to_float(it->value().data()))) {
        _db->del_data(i, it->key().data(), it->key().size());
        ssd_count++;
      } else {
        _db->put(i,
                 it->key().data(),
                 it->key().size(),
                 it->value().data(),
                 it->value().size());
      }
    }
    delete it;
    LOG(INFO) << "SSDSparseTable shrink success. shard:" << i << " delete MEM["
              << mem_count << "] SSD[" << ssd_count << "]";
    // _db->flush(i);
  }
  return 0;
}

int32_t SSDSparseTable::UpdateTable() {
  int count = 0;
  for (int i = 0; i < _real_local_shard_num; ++i) {
    auto& shard = _local_shards[i];
    // from mem to ssd
    for (auto it = shard.begin(); it != shard.end();) {
      if (_value_accesor->SaveSSD(it.value().data())) {
        _db->put(i,
                 reinterpret_cast<const char*>(&it.key()),
                 sizeof(uint64_t),
                 reinterpret_cast<const char*>(it.value().data()),
                 it.value().size() * sizeof(float));
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
  for (int i = 0; i < _real_local_shard_num; ++i) {
    local_size += _local_shards[i].size();
  }
  return local_size;
}

int32_t SSDSparseTable::Save(const std::string& path,
                             const std::string& param) {
  std::lock_guard<std::mutex> guard(_table_mutex);
#ifdef PADDLE_WITH_HETERPS
  int save_param = atoi(param.c_str());
  int32_t ret = 0;
  if (save_param > 3) {
    ret = SaveWithStringMultiOutput(path, param);  // batch_model:4  xbox:5
  } else {
    ret = SaveWithBinary(path, param);  // batch_model:0  xbox:1
  }
  return ret;
#else
  // CPUPS PSCORE
  return SaveWithString(path, param);  // batch_model:0  xbox:1
#endif
}

// save shard_num 个文件
int32_t SSDSparseTable::SaveWithString(const std::string& path,
                                       const std::string& param) {
  std::lock_guard<std::mutex> guard(_table_mutex);
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
#ifdef PADDLE_WITH_HETERPS
  save_param -= 4;
#endif
  //    if (save_param == 5) {
  //        return save_patch(path, save_param);
  //    }

  // LOG(INFO) << "table cache rate is: " << _config.sparse_table_cache_rate();
  VLOG(0) << "table cache rate is: " << _config.sparse_table_cache_rate();
  VLOG(0) << "enable_sparse_table_cache: "
          << _config.enable_sparse_table_cache();
  VLOG(0) << "LocalSize: " << LocalSize();
  if (_config.enable_sparse_table_cache()) {
    VLOG(0) << "Enable sparse table cache, top n:" << _cache_tk_size;
  }
  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, _cache_tk_size);
  VLOG(0) << "TopkCalculator top n:" << _cache_tk_size;
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  std::string table_path = TableDir(path);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
#ifdef PADDLE_WITH_GPU_GRAPH
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
#endif

  // std::atomic<uint32_t> feasign_size;
  std::atomic<uint32_t> feasign_size_all{0};
  // feasign_size = 0;

  std::vector<
      ::paddle::framework::Channel<std::pair<uint64_t, std::vector<float>>>>
      fs_channel;
  for (int i = 0; i < _real_local_shard_num; i++) {
    fs_channel.push_back(::paddle::framework::MakeChannel<
                         std::pair<uint64_t, std::vector<float>>>(10240));
  }
  std::vector<std::thread> threads;
  threads.resize(_real_local_shard_num);

  auto save_func = [this,
                    &save_param,
                    &table_path,
                    &file_start_idx,
                    &fs_channel](int file_num) {
    int err_no = 0;
    FsChannelConfig channel_config;
    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path =
          ::paddle::string::format_string("%s/part-%03d-%05d.gz",
                                          table_path.c_str(),
                                          _shard_idx,
                                          file_start_idx + file_num);
    } else {
      channel_config.path =
          ::paddle::string::format_string("%s/part-%03d-%05d",
                                          table_path.c_str(),
                                          _shard_idx,
                                          file_start_idx + file_num);
    }
    channel_config.converter = _value_accesor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(save_param).deconverter;
    auto write_channel =
        _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
    ::paddle::framework::ChannelReader<std::pair<uint64_t, std::vector<float>>>
        reader(fs_channel[file_num].get());
    std::pair<uint64_t, std::vector<float>> out_str;
    while (reader >> out_str) {
      std::string format_value = _value_accesor->ParseToString(
          out_str.second.data(), out_str.second.size());
      if (0 != write_channel->write_line(::paddle::string::format_string(
                   "%lu %s", out_str.first, format_value.c_str()))) {
        LOG(FATAL) << "SSDSparseTable save failed, retry it! path:"
                   << channel_config.path;
      }
    }
    write_channel->close();
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(save_func, i);
  }

  std::vector<::paddle::framework::ChannelWriter<
      std::pair<uint64_t, std::vector<float>>>>
      writers(_real_local_shard_num);
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    int feasign_size = 0;
    auto& shard = _local_shards[i];
    auto& writer = writers[i];
    writer.Reset(fs_channel[i].get());
    {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2)) {
          // get_field get right decayed show
          tk.push(i, _value_accesor->GetField(it.value().data(), "show"));
        }
        if (_value_accesor->Save(it.value().data(), save_param)) {
          std::vector<float> feature_value;
          feature_value.resize(it.value().size());
          memcpy(const_cast<float*>(feature_value.data()),
                 it.value().data(),
                 it.value().size() * sizeof(float));
          writer << std::make_pair(it.key(), std::move(feature_value));
          ++feasign_size;
        }
      }
    }

    if (save_param != 1) {
      auto* it = _db->get_iterator(i);
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        bool need_save = _value_accesor->Save(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        _value_accesor->UpdateStatAfterSave(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        if (need_save) {
          std::vector<float> feature_value;
          feature_value.resize(it->value().size() / sizeof(float));
          memcpy(const_cast<float*>(feature_value.data()),
                 ::paddle::string::str_to_float(it->value().data()),
                 it->value().size());
          writer << std::make_pair(*(reinterpret_cast<uint64_t*>(
                                       const_cast<char*>(it->key().data()))),
                                   std::move(feature_value));
          ++feasign_size;
        }
      }
      delete it;
    }

    writer.Flush();
    fs_channel[i]->Close();
    feasign_size_all += feasign_size;
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      _value_accesor->UpdateStatAfterSave(it.value().data(), save_param);
    }
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  for (size_t i = 0; i < fs_channel.size(); i++) {
    fs_channel[i].reset();
  }
  fs_channel.clear();

  if (save_param == 3) {
    // UpdateTable();
    _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
    VLOG(0) << "SSDSparseTable update success.";
  }
  VLOG(0) << "SSDSparseTable save success, feasign size:" << feasign_size_all
          << ", path:"
          << ::paddle::string::format_string("%s/%03d/part-%03d-",
                                             path.c_str(),
                                             _config.table_id(),
                                             _shard_idx)
          << " from " << file_start_idx << " to "
          << file_start_idx + _real_local_shard_num - 1;
  _local_show_threshold = tk.top();
  VLOG(0) << "local cache threshold: " << _local_show_threshold;
  return 0;
}

// save shard_num * n 个文件, n由模型大小决定
int32_t SSDSparseTable::SaveWithStringMultiOutput(const std::string& path,
                                                  const std::string& param) {
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }
  int save_param = atoi(param.c_str());
#ifdef PADDLE_WITH_HETERPS
  save_param -= 4;
#endif
  VLOG(0) << "table cache rate is: " << _config.sparse_table_cache_rate();
  VLOG(0) << "enable_sparse_table_cache: "
          << _config.enable_sparse_table_cache();
  VLOG(0) << "LocalSize: " << LocalSize();
  if (_config.enable_sparse_table_cache()) {
    VLOG(0) << "Enable sparse table cache, top n:" << _cache_tk_size;
  }
  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, _cache_tk_size);
  VLOG(0) << "TopkCalculator top n:" << _cache_tk_size;
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  std::string table_path = TableDir(path);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
#ifdef PADDLE_WITH_GPU_GRAPH
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
#endif

  std::atomic<uint32_t> feasign_size_all{0};
  std::vector<::paddle::framework::Channel<std::shared_ptr<MemRegion>>>
      busy_channel;
  std::vector<::paddle::framework::Channel<std::shared_ptr<MemRegion>>>
      free_channel;
  std::vector<std::thread> threads;

  for (int i = 0; i < _real_local_shard_num; i++) {
    busy_channel.push_back(
        ::paddle::framework::MakeChannel<std::shared_ptr<MemRegion>>());
    free_channel.push_back(
        ::paddle::framework::MakeChannel<std::shared_ptr<MemRegion>>());
  }
  threads.resize(_real_local_shard_num);

  auto save_func = [this,
                    &save_param,
                    &table_path,
                    &file_start_idx,
                    &free_channel,
                    &busy_channel](int file_num) {
    int err_no = 0;
    int shard_num = file_num;
    int part_num = 0;
    shard_num = file_num;
    part_num = 0;
    FsChannelConfig channel_config;
    channel_config.converter = _value_accesor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(save_param).deconverter;

    auto get_filename = [](int compress,
                           int save_param,
                           const char* table_path,
                           int node_num,
                           int shard_num,
                           int part_num,
                           int split_num) {
      if (compress && (save_param == 0 || save_param == 3)) {
        // return
        // ::paddle::string::format_string("%s/part-%03d-%05d-%03d-%03d.gz",
        //     table_path, node_num, shard_num, part_num, split_num);
        return ::paddle::string::format_string(
            "%s/part-%05d-%03d.gz", table_path, shard_num, split_num);
      } else {
        // return ::paddle::string::format_string("%s/part-%03d-%05d-%03d-%03d",
        //     table_path, node_num,  shard_num, part_num, split_num);
        return ::paddle::string::format_string(
            "%s/part-%05d-%03d", table_path, shard_num, split_num);
      }
    };
    std::shared_ptr<MemRegion> region = nullptr;
    // std::shared_ptr<AfsWriter> afs_writer = nullptr;
    // std::shared_ptr<XboxConverter> xbox_converter = nullptr;
    std::string filename;
    int last_file_idx = -1;
    std::shared_ptr<FsWriteChannel> write_channel = nullptr;

    while (busy_channel[shard_num]->Get(region)) {
      if (region->_file_idx != last_file_idx) {
        filename = get_filename(_config.compress_in_save(),
                                save_param,
                                table_path.c_str(),
                                _shard_idx,
                                file_start_idx + shard_num,
                                part_num,
                                region->_file_idx);
        channel_config.path = filename;
        write_channel =
            _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
        // afs_writer = _api_wrapper.open_writer(filename);
        last_file_idx = region->_file_idx;
        // xbox_converter = std::make_shared<XboxConverter>(afs_writer);
      }
      char* cursor = region->_buf;
      int remain = region->_cur;
      while (remain) {
        uint32_t len = *reinterpret_cast<uint32_t*>(cursor);
        len -= sizeof(uint32_t);
        remain -= sizeof(uint32_t);
        cursor += sizeof(uint32_t);

        uint64_t k = *reinterpret_cast<uint64_t*>(cursor);
        cursor += sizeof(uint64_t);
        len -= sizeof(uint64_t);
        remain -= sizeof(uint64_t);

        float* value = reinterpret_cast<float*>(cursor);
        int dim = len / sizeof(float);

        std::string format_value = _value_accesor->ParseToString(value, dim);
        if (0 != write_channel->write_line(::paddle::string::format_string(
                     "%lu %s", k, format_value.c_str()))) {
          VLOG(0) << "SSDSparseTable save failed, retry it! path:"
                  << channel_config.path;
        }
        remain -= len;
        cursor += len;
      }
      region->reset();
      free_channel[shard_num]->Put(region);
    }
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(save_func, i);
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < static_cast<size_t>(_real_local_shard_num); ++i) {
    std::shared_ptr<MemRegion> region = nullptr;
    std::vector<std::shared_ptr<MemRegion>> regions;
    free_channel[i]->Put(std::make_shared<MemRegion>());
    free_channel[i]->Put(std::make_shared<MemRegion>());
    free_channel[i]->Get(region);
    int feasign_size = 0;
    auto& shard = _local_shards[i];
    int file_idx = 0;
    int switch_cnt = 0;
    region->_file_idx = 0;
    {
      // auto ssd_timer =
      // std::make_shared<CostTimer>("pslib_downpour_memtable_iterator_v2");
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2)) {
          // get_field get right decayed show
          tk.push(i, _value_accesor->GetField(it.value().data(), "show"));
        }
        if (_value_accesor->Save(it.value().data(), save_param)) {
          uint32_t len = sizeof(uint64_t) + it.value().size() * sizeof(float) +
                         sizeof(uint32_t);
          int region_idx = i;
          if (!region->buff_remain(len)) {
            busy_channel[region_idx]->Put(region);
            free_channel[region_idx]->Get(region);
            // region->_file_idx = 0;
            switch_cnt += 1;
            if (switch_cnt % 1024 == 0) {
              file_idx += 1;
            }
            region->_file_idx = file_idx;
          }
          int read_count = 0;
          char* buf = region->acquire(len);
          // CHECK(buf);
          *reinterpret_cast<uint32_t*>(buf + read_count) = len;
          read_count += sizeof(uint32_t);

          *reinterpret_cast<uint64_t*>(buf + read_count) = it.key();
          read_count += sizeof(uint64_t);

          memcpy(buf + read_count,
                 it.value().data(),
                 sizeof(float) * it.value().size());
          // if (save_param == 1 || save_param == 2) {
          //     _value_accesor->update_time_decay((float*)(buf + read_count),
          //     false);
          // }
          ++feasign_size;
        }
      }
    }
    // delta and cache is all in mem, base in rocksdb
    if (save_param != 1) {
      // int file_idx = 1;
      // int switch_cnt = 0;
      file_idx++;
      switch_cnt = 0;
      // ssd里的参数必须按key值升序, 而内存里的参数是乱序的,
      // 这里必须重新申请region
      busy_channel[i]->Put(region);
      free_channel[i]->Get(region);
      region->_file_idx = file_idx;
      auto* it = _db->get_iterator(i);
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        bool need_save = _value_accesor->Save(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        _value_accesor->UpdateStatAfterSave(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        if (need_save) {
          uint32_t len =
              sizeof(uint64_t) + it->value().size() + sizeof(uint32_t);
          int region_idx = i;
          uint64_t key = *(
              reinterpret_cast<uint64_t*>(const_cast<char*>(it->key().data())));
          if (!region->buff_remain(len)) {
            busy_channel[region_idx]->Put(region);
            free_channel[region_idx]->Get(region);
            switch_cnt += 1;
            if (switch_cnt % 1024 == 0) {
              // if (switch_cnt % 1 == 0) {
              file_idx += 1;
            }
            region->_file_idx = file_idx;
          }
          int read_count = 0;
          char* buf = region->acquire(len);
          *reinterpret_cast<uint32_t*>(buf + read_count) = len;
          read_count += sizeof(uint32_t);

          *reinterpret_cast<uint64_t*>(buf + read_count) = key;
          read_count += sizeof(uint64_t);

          memcpy(buf + read_count, it->value().data(), it->value().size());
          // if (save_param == 2) {
          //     _value_accesor->update_time_decay((float*)(buf + read_count),
          //     false);
          // }
          ++feasign_size;
        }
      }
      delete it;
    }
    if (region->_cur) {
      busy_channel[i]->Put(region);
    }
    feasign_size_all += feasign_size;
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      _value_accesor->UpdateStatAfterSave(it.value().data(), save_param);
    }
  }
  for (auto& channel : busy_channel) {
    channel->Close();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  for (size_t i = 0; i < busy_channel.size(); i++) {
    busy_channel[i].reset();
    free_channel[i].reset();
  }
  busy_channel.clear();
  free_channel.clear();
  if (save_param == 3) {
    //        update_table();
    uint64_t ssd_key_num = 0;
    _db->get_estimate_key_num(ssd_key_num);
    _cache_tk_size =
        (LocalSize() + ssd_key_num) * _config.sparse_table_cache_rate();
    VLOG(0) << "DownpourSparseSSDTable update success.";
  }
  VLOG(0) << "DownpourSparseSSDTable save success, feasign size:"
          << feasign_size_all << " ,path:"
          << ::paddle::string::format_string("%s/%03d/part-%03d-",
                                             path.c_str(),
                                             _config.table_id(),
                                             _shard_idx)
          << " from " << file_start_idx << " to "
          << file_start_idx + _real_local_shard_num - 1;
  if (_config.enable_sparse_table_cache()) {
    _local_show_threshold = tk.top();
    VLOG(0) << "local cache threshold: " << _local_show_threshold;
  }
  // int32 may overflow need to change return value
  return 0;
}

int32_t SSDSparseTable::SaveWithBinary(const std::string& path,
                                       const std::string& param) {
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }
  int save_param = atoi(param.c_str());
  VLOG(0) << "table cache rate is: " << _config.sparse_table_cache_rate();
  VLOG(0) << "enable_sparse_table_cache: "
          << _config.enable_sparse_table_cache();
  VLOG(0) << "LocalSize: " << LocalSize();
  if (_config.enable_sparse_table_cache()) {
    VLOG(0) << "Enable sparse table cache, top n:" << _cache_tk_size;
  }
  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, _cache_tk_size);
  VLOG(0) << "TopkCalculator top n:" << _cache_tk_size;
  size_t file_start_idx = _avg_local_shard_num * _shard_idx;
  std::string table_path = TableDir(path);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
#ifdef PADDLE_WITH_GPU_GRAPH
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
#endif

  std::atomic<uint32_t> feasign_size_all{0};
  std::vector<::paddle::framework::Channel<std::shared_ptr<MemRegion>>>
      busy_channel;
  std::vector<::paddle::framework::Channel<std::shared_ptr<MemRegion>>>
      free_channel;
  std::vector<std::thread> threads;

  for (int i = 0; i < _real_local_shard_num; i++) {
    busy_channel.push_back(
        ::paddle::framework::MakeChannel<std::shared_ptr<MemRegion>>());
    free_channel.push_back(
        ::paddle::framework::MakeChannel<std::shared_ptr<MemRegion>>());
  }
  threads.resize(_real_local_shard_num);

  auto save_func = [this,
                    &save_param,
                    &table_path,
                    &file_start_idx,
                    &free_channel,
                    &busy_channel](int file_num) {
    int err_no = 0;
    int shard_num = file_num;
    int part_num = 0;
    shard_num = file_num;
    part_num = 0;
    FsChannelConfig channel_config;
    channel_config.converter = _value_accesor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(save_param).deconverter;

    auto get_filename = [](int compress,
                           int save_param,
                           const char* table_path,
                           int node_num,
                           int shard_num,
                           int part_num,
                           int split_num) {
      if (compress && (save_param == 0 || save_param == 3)) {
        return ::paddle::string::format_string("%s/part-%03d-%05d-%03d-%03d.gz",
                                               table_path,
                                               node_num,
                                               shard_num,
                                               part_num,
                                               split_num);
      } else {
        return ::paddle::string::format_string("%s/part-%03d-%05d-%03d-%03d",
                                               table_path,
                                               node_num,
                                               shard_num,
                                               part_num,
                                               split_num);
      }
    };
    std::shared_ptr<MemRegion> region = nullptr;
    std::string filename;
    int last_file_idx = -1;
    std::shared_ptr<FsWriteChannel> write_channel = nullptr;
    if (save_param != 1 && save_param != 2) {
      while (busy_channel[shard_num]->Get(region)) {
        if (region->_file_idx != last_file_idx) {
          filename = get_filename(_config.compress_in_save(),
                                  save_param,
                                  table_path.c_str(),
                                  _shard_idx,
                                  file_start_idx + shard_num,
                                  part_num,
                                  region->_file_idx);
          channel_config.path = filename;
          write_channel =
              _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
          last_file_idx = region->_file_idx;
        }
        if (0 != write_channel->write(region->_buf, region->_cur)) {
          LOG(FATAL) << "DownpourSparseSSDTable save failed, retry it! path:"
                     << channel_config.path;
          CHECK(false);
        }
        region->reset();
        free_channel[shard_num]->Put(region);
      }
    } else {
      while (busy_channel[shard_num]->Get(region)) {
        if (region->_file_idx != last_file_idx) {
          filename = get_filename(_config.compress_in_save(),
                                  save_param,
                                  table_path.c_str(),
                                  _shard_idx,
                                  file_start_idx + shard_num,
                                  part_num,
                                  region->_file_idx);
          channel_config.path = filename;
          write_channel =
              _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
          last_file_idx = region->_file_idx;
        }
        char* cursor = region->_buf;
        int remain = region->_cur;
        while (remain) {
          uint32_t len = *reinterpret_cast<uint32_t*>(cursor);
          len -= sizeof(uint32_t);
          remain -= sizeof(uint32_t);
          cursor += sizeof(uint32_t);

          uint64_t k = *reinterpret_cast<uint64_t*>(cursor);
          cursor += sizeof(uint64_t);
          len -= sizeof(uint64_t);
          remain -= sizeof(uint64_t);

          float* value = reinterpret_cast<float*>(cursor);
          int dim = len / sizeof(float);

          std::string format_value = _value_accesor->ParseToString(value, dim);
          if (0 != write_channel->write_line(::paddle::string::format_string(
                       "%lu %s", k, format_value.c_str()))) {
            LOG(FATAL) << "SSDSparseTable save failed, retry it! path:"
                       << channel_config.path;
          }
          remain -= len;
          cursor += len;
        }
        region->reset();
        free_channel[shard_num]->Put(region);
      }
    }
    // write_channel->close();
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(save_func, i);
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < static_cast<size_t>(_real_local_shard_num); ++i) {
    std::shared_ptr<MemRegion> region = nullptr;
    std::vector<std::shared_ptr<MemRegion>> regions;
    free_channel[i]->Put(std::make_shared<MemRegion>());
    free_channel[i]->Put(std::make_shared<MemRegion>());
    free_channel[i]->Get(region);
    int feasign_size = 0;
    auto& shard = _local_shards[i];
    region->_file_idx = 0;
    {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2)) {
          // get_field get right decayed show
          tk.push(i, _value_accesor->GetField(it.value().data(), "show"));
        }
        if (_value_accesor->Save(it.value().data(), save_param)) {
          uint32_t len = sizeof(uint64_t) + it.value().size() * sizeof(float) +
                         sizeof(uint32_t);
          int region_idx = i;
          if (!region->buff_remain(len)) {
            busy_channel[region_idx]->Put(region);
            free_channel[region_idx]->Get(region);
            region->_file_idx = 0;
          }
          int read_count = 0;
          char* buf = region->acquire(len);
          // CHECK(buf);
          *reinterpret_cast<uint32_t*>(buf + read_count) = len;
          read_count += sizeof(uint32_t);

          *reinterpret_cast<uint64_t*>(buf + read_count) = it.key();
          read_count += sizeof(uint64_t);

          memcpy(buf + read_count,
                 it.value().data(),
                 sizeof(float) * it.value().size());
          ++feasign_size;
        }
      }
    }
    // delta and cache is all in mem, base in rocksdb
    if (save_param != 1) {
      int file_idx = 1;
      int switch_cnt = 0;
      busy_channel[i]->Put(region);
      free_channel[i]->Get(region);
      region->_file_idx = file_idx;
      auto* it = _db->get_iterator(i);
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        bool need_save = _value_accesor->Save(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        _value_accesor->UpdateStatAfterSave(
            ::paddle::string::str_to_float(it->value().data()), save_param);
        if (need_save) {
          uint32_t len =
              sizeof(uint64_t) + it->value().size() + sizeof(uint32_t);
          int region_idx = i;
          uint64_t key = *(
              reinterpret_cast<uint64_t*>(const_cast<char*>(it->key().data())));
          if (!region->buff_remain(len)) {
            busy_channel[region_idx]->Put(region);
            free_channel[region_idx]->Get(region);
            switch_cnt += 1;
            if (switch_cnt % 1024 == 0) {
              file_idx += 1;
            }
            region->_file_idx = file_idx;
          }
          int read_count = 0;
          char* buf = region->acquire(len);
          *reinterpret_cast<uint32_t*>(buf + read_count) = len;
          read_count += sizeof(uint32_t);

          *reinterpret_cast<uint64_t*>(buf + read_count) = key;
          read_count += sizeof(uint64_t);

          memcpy(buf + read_count, it->value().data(), it->value().size());
          // if (save_param == 2) {
          //     _value_accesor->update_time_decay((float*)(buf + read_count),
          //     false);
          // }
          ++feasign_size;
        }
      }
      delete it;
    }
    if (region->_cur) {
      busy_channel[i]->Put(region);
    }
    feasign_size_all += feasign_size;
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      _value_accesor->UpdateStatAfterSave(it.value().data(), save_param);
    }
  }
  for (auto& channel : busy_channel) {
    channel->Close();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  for (size_t i = 0; i < busy_channel.size(); i++) {
    busy_channel[i].reset();
    free_channel[i].reset();
  }

  busy_channel.clear();
  free_channel.clear();
  if (save_param == 3) {
    //        update_table();
    uint64_t ssd_key_num = 0;
    _db->get_estimate_key_num(ssd_key_num);
    _cache_tk_size =
        (LocalSize() + ssd_key_num) * _config.sparse_table_cache_rate();
    VLOG(0) << "DownpourSparseSSDTable update success.";
  }
  VLOG(0) << "DownpourSparseSSDTable save success, feasign size:"
          << feasign_size_all << " ,path:"
          << ::paddle::string::format_string("%s/%03d/part-%03d-",
                                             path.c_str(),
                                             _config.table_id(),
                                             _shard_idx)
          << " from " << file_start_idx << " to "
          << file_start_idx + _real_local_shard_num - 1;
  if (_config.enable_sparse_table_cache()) {
    _local_show_threshold = tk.top();
    VLOG(0) << "local cache threshold: " << _local_show_threshold;
  }
  // int32 may overflow need to change return value
  return 0;
}

int64_t SSDSparseTable::CacheShuffle(
    const std::string& path,
    const std::string& param,
    double cache_threshold,
    std::function<std::future<int32_t>(
        int msg_type, int to_pserver_id, std::string& msg)> send_msg_func,
    ::paddle::framework::Channel<std::pair<uint64_t, std::string>>&
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
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;

  std::vector<
      ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>>
      writers(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, std::string>>> datas(
      _real_local_shard_num);

  int feasign_size = 0;
  std::vector<::paddle::framework::Channel<std::pair<uint64_t, std::string>>>
      tmp_channels;
  for (int i = 0; i < _real_local_shard_num; ++i) {
    tmp_channels.push_back(
        ::paddle::framework::MakeChannel<std::pair<uint64_t, std::string>>());
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>&
        writer = writers[i];
    //    std::shared_ptr<::paddle::framework::ChannelObject<std::pair<uint64_t,
    //    std::string>>> tmp_chan =
    //        ::paddle::framework::MakeChannel<std::pair<uint64_t,
    //        std::string>>();
    writer.Reset(tmp_channels[i].get());

    auto& shard = _local_shards[i];
    for (auto it = shard.begin(); it != shard.end(); ++it) {
      if (_value_accesor->SaveCache(
              it.value().data(), save_param, cache_threshold)) {
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
  for (int idx_shard = 0; idx_shard < _real_local_shard_num; ++idx_shard) {
    ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>&
        writer = writers[idx_shard];
    auto channel = writer.channel();
    std::vector<std::pair<uint64_t, std::string>>& data = datas[idx_shard];
    std::vector<::paddle::framework::BinaryArchive> ars(shuffle_node_num);
    while (channel->Read(data)) {
      for (auto& t : data) {
        auto pserver_id =
            ::paddle::distributed::local_random_engine()() % shuffle_node_num;
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
      for (int index = 0; index < shuffle_node_num; ++index) {
        size_t i = send_index[index];
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
      ars = std::vector<::paddle::framework::BinaryArchive>(shuffle_node_num);
      data = std::vector<std::pair<uint64_t, std::string>>();
    }
  }
  shuffled_channel->Write(std::move(local_datas));
  LOG(INFO) << "cache shuffle finished";
  return 0;
}

int32_t SSDSparseTable::SaveCache(
    const std::string& path,
    const std::string& param,
    ::paddle::framework::Channel<std::pair<uint64_t, std::string>>&
        shuffled_channel) {
  if (_shard_idx >= _config.sparse_table_cache_file_num()) {
    return 0;
  }
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  std::string table_path = ::paddle::string::format_string(
      "%s/%03d_cache/", path.c_str(), _config.table_id());
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d", table_path.c_str(), _shard_idx));
  uint32_t feasign_size = 0;
  FsChannelConfig channel_config;
  // not compress cache model
  channel_config.path = ::paddle::string::format_string(
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
      if (0 != write_channel->write_line(::paddle::string::format_string(
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
  VLOG(0) << "LOAD FLAGS_rocksdb_path:" << FLAGS_rocksdb_path;
  std::string table_path = TableDir(path);
  auto file_list = _afs_client.list(table_path);

  // std::sort(file_list.begin(), file_list.end());
  for (auto file : file_list) {
    VLOG(1) << "SSDSparseTable::Load() file list: " << file;
  }

  int load_param = atoi(param.c_str());
  size_t expect_shard_num = _sparse_table_shard_num;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "SSDSparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.empty()) {
    LOG(WARNING) << "SSDSparseTable load file is empty, path:" << path;
    return -1;
  }
  if (load_param > 3) {
    size_t file_start_idx = _shard_idx * _avg_local_shard_num;
    return LoadWithString(file_start_idx,
                          file_start_idx + _real_local_shard_num,
                          file_list,
                          param);
  } else {
    return LoadWithBinary(table_path, load_param);
  }
}

int32_t SSDSparseTable::LoadWithString(
    size_t file_start_idx,
    size_t end_idx,
    const std::vector<std::string>& file_list,
    const std::string& param) {
  if (file_start_idx >= file_list.size()) {
    return 0;
  }
  int load_param = atoi(param.c_str());
#ifdef PADDLE_WITH_HETERPS
  load_param -= 4;
#endif
  size_t feature_value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);

#ifdef PADDLE_WITH_HETERPS
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 15 ? _real_local_shard_num : 15;
#endif

  for (int i = 0; i < _real_local_shard_num; i++) {
    _fs_channel.push_back(::paddle::framework::MakeChannel<std::string>(30000));
  }

  std::vector<std::thread> threads;
  threads.resize(thread_num);
  auto load_func = [this, &file_start_idx, &file_list, &load_param](
                       int file_num) {
    int err_no = 0;
    FsChannelConfig channel_config;
    channel_config.path = file_list[file_num + file_start_idx];
    VLOG(1) << "SSDSparseTable::load begin load " << channel_config.path
            << " into local shard " << file_num;
    channel_config.converter = _value_accesor->Converter(load_param).converter;
    channel_config.deconverter =
        _value_accesor->Converter(load_param).deconverter;

    std::string line_data;
    auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
    ::paddle::framework::ChannelWriter<std::string> writer(
        _fs_channel[file_num].get());
    while (read_channel->read_line(line_data) == 0 && line_data.size() > 1) {
      writer << line_data;
    }
    writer.Flush();
    read_channel->close();
    _fs_channel[file_num]->Close();
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(load_func, i);
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    std::vector<std::pair<char*, int>> ssd_keys;
    std::vector<std::pair<char*, int>> ssd_values;
    std::vector<uint64_t> tmp_key;
    ssd_keys.reserve(FLAGS_pserver_load_batch_size);
    ssd_values.reserve(FLAGS_pserver_load_batch_size);
    tmp_key.reserve(FLAGS_pserver_load_batch_size);
    ssd_keys.clear();
    ssd_values.clear();
    tmp_key.clear();
    std::string line_data;
    char* end = NULL;
    int local_shard_id = i % _avg_local_shard_num;
    auto& shard = _local_shards[local_shard_id];
    float data_buffer[FLAGS_pserver_load_batch_size *
                      feature_value_size];  // NOLINT
    float* data_buffer_ptr = data_buffer;
    uint64_t mem_count = 0;
    uint64_t ssd_count = 0;
    uint64_t mem_mf_count = 0;
    uint64_t ssd_mf_count = 0;
    uint64_t filtered_count = 0;
    uint64_t filter_time = 0;
    uint64_t filter_begin = 0;

    ::paddle::framework::ChannelReader<std::string> reader(
        _fs_channel[i].get());

    while (reader >> line_data) {
      uint64_t key = std::strtoul(line_data.data(), &end, 10);
      if (FLAGS_pserver_open_strict_check) {
        if (key % _sparse_table_shard_num != (i + file_start_idx)) {
          LOG(WARNING) << "SSDSparseTable key:" << key << " not match shard,"
                       << " file_idx:" << i
                       << " shard num:" << _sparse_table_shard_num;
          continue;
        }
      }
      size_t value_size =
          _value_accesor->ParseFromString(++end, data_buffer_ptr);
      filter_begin = butil::gettimeofday_ms();
      if (!_value_accesor->FilterSlot(data_buffer_ptr)) {
        filter_time += butil::gettimeofday_ms() - filter_begin;
        // ssd or mem
        if (_value_accesor->SaveSSD(data_buffer_ptr)) {
          tmp_key.emplace_back(key);
          ssd_keys.emplace_back(reinterpret_cast<char*>(&tmp_key.back()),
                                sizeof(uint64_t));
          ssd_values.emplace_back(reinterpret_cast<char*>(data_buffer_ptr),
                                  value_size * sizeof(float));
          data_buffer_ptr += feature_value_size;
          if (static_cast<int>(ssd_keys.size()) ==
              FLAGS_pserver_load_batch_size) {
            _db->put_batch(
                local_shard_id, ssd_keys, ssd_values, ssd_keys.size());
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
      } else {
        filter_time += butil::gettimeofday_ms() - filter_begin;
        filtered_count++;
      }
    }
    // last batch
    if (!ssd_keys.empty()) {
      _db->put_batch(local_shard_id, ssd_keys, ssd_values, ssd_keys.size());
    }

    _db->flush(local_shard_id);
    VLOG(0) << "Table>> load done. ALL[" << mem_count + ssd_count << "] MEM["
            << mem_count << "] MEM_MF[" << mem_mf_count << "] SSD[" << ssd_count
            << "] SSD_MF[" << ssd_mf_count << "] FILTERED[" << filtered_count
            << "] filter_time cost:" << filter_time / 1000 << " s";
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  for (size_t i = 0; i < _fs_channel.size(); i++) {
    _fs_channel[i].reset();
  }
  _fs_channel.clear();
  LOG(INFO) << "load num:" << LocalSize();
  LOG(INFO) << "SSDSparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + _real_local_shard_num - 1];

  _cache_tk_size = LocalSize() * _config.sparse_table_cache_rate();
  return 0;
}

int32_t SSDSparseTable::LoadWithBinary(const std::string& path, int param) {
  size_t feature_value_size =
      _value_accesor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accesor->GetAccessorInfo().mf_size / sizeof(float);
  // task pool _file_num_one_shard default 7
  auto task_pool = std::make_shared<::ThreadPool>(_real_local_shard_num * 7);
  auto filelists = _afs_client.list(::paddle::string::format_string(
      "%s/part-%03d*", path.c_str(), _shard_idx));
  // #pragma omp parallel for schedule(dynamic)
  std::vector<std::future<int>> tasks;

  for (int shard_idx = 0; shard_idx < _real_local_shard_num; shard_idx++) {
    // FsChannelConfig channel_config;
    // channel_config.converter = _value_accesor->Converter(param).converter;
    // channel_config.deconverter =
    // _value_accesor->Converter(param).deconverter;
    for (auto& filename : filelists) {
      std::vector<std::string> split_filename_string =
          ::paddle::string::split_string<std::string>(filename, "-");
      int file_split_idx =
          atoi(split_filename_string[split_filename_string.size() - 1].c_str());
      int file_shard_idx =
          atoi(split_filename_string[split_filename_string.size() - 3].c_str());
      if (file_shard_idx != shard_idx) {
        continue;
      }
      auto future = task_pool->enqueue([this,
                                        feature_value_size,
                                        mf_value_size,
                                        shard_idx,
                                        filename,
                                        file_split_idx,
                                        param]() -> int {
        // &channel_config]() -> int {
        FsChannelConfig channel_config;
        channel_config.converter = _value_accesor->Converter(param).converter;
        channel_config.deconverter =
            _value_accesor->Converter(param).deconverter;
        int err_no = 0;
        uint64_t mem_count = 0;
        uint64_t mem_mf_count = 0;
        uint64_t ssd_count = 0;
        uint64_t ssd_mf_count = 0;

        channel_config.path = filename;
        auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
        // auto reader = _api_wrapper.open_reader(filename);
        auto& shard = _local_shards[shard_idx];
        rocksdb::Options options;
        options.comparator = _db->get_comparator();
        rocksdb::BlockBasedTableOptions bbto;
        bbto.format_version = 5;
        bbto.use_delta_encoding = false;
        bbto.block_size = 4 * 1024;
        bbto.block_restart_interval = 6;
        bbto.cache_index_and_filter_blocks = false;
        bbto.filter_policy.reset(rocksdb::NewBloomFilterPolicy(15, false));
        bbto.whole_key_filtering = true;
        options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(bbto));
        options.OptimizeLevelStyleCompaction();
        options.keep_log_file_num = 100;
        options.max_log_file_size = 50 * 1024 * 1024;  // 50MB
        options.create_if_missing = true;
        options.use_direct_reads = true;
        options.write_buffer_size = 256 * 1024 * 1024;  // 256MB
        options.max_write_buffer_number = 8;
        options.max_bytes_for_level_base =
            options.max_write_buffer_number * options.write_buffer_size;
        options.min_write_buffer_number_to_merge = 1;
        options.target_file_size_base = 1024 * 1024 * 1024;  // 1024MB
        options.memtable_prefix_bloom_size_ratio = 0.02;
        options.num_levels = 4;
        options.max_open_files = -1;

        options.compression = rocksdb::kNoCompression;

        rocksdb::SstFileWriter sst_writer(rocksdb::EnvOptions(), options);
        int use_sst = 0;
        if (file_split_idx != 0) {
          std::string path =
              ::paddle::string::format_string("%s_%d/part-%03d.sst",
                                              FLAGS_rocksdb_path.c_str(),
                                              shard_idx,
                                              file_split_idx);
          rocksdb::Status status = sst_writer.Open(path);
          if (!status.ok()) {
            VLOG(0) << "sst writer open " << path << "failed";
            abort();
          }
          use_sst = 1;
        }
        uint64_t last_k = 0;
        int buf_len = 1024 * 1024 * 10;
        char* buf = reinterpret_cast<char*>(malloc(buf_len + 10));
        // used for cache converted line
        char* convert_buf = reinterpret_cast<char*>(malloc(buf_len + 10));
        int ret = 0;
        char* cursor = buf;
        char* convert_cursor = convert_buf;
        int remain = 0;
        while (1) {
          remain = ret;
          cursor = buf + remain;
          ret = read_channel->read(cursor, buf_len - remain);
          // ret = reader->read(cursor, buf_len - remain);
          if (ret <= 0) {
            break;
          }
          cursor = buf;
          convert_cursor = convert_buf;
          ret += remain;
          do {
            if (ret >= static_cast<int>(sizeof(uint32_t))) {
              uint32_t len = *reinterpret_cast<uint32_t*>(cursor);
              if (ret >= static_cast<int>(len)) {
                ret -= sizeof(uint32_t);
                len -= sizeof(uint32_t);
                cursor += sizeof(uint32_t);

                uint64_t k = *reinterpret_cast<uint64_t*>(cursor);
                cursor += sizeof(uint64_t);
                ret -= sizeof(uint64_t);
                len -= sizeof(uint64_t);

                float* value = reinterpret_cast<float*>(cursor);
                size_t dim = len / sizeof(float);

                // copy value to convert_buf
                memcpy(convert_cursor, cursor, len);
                float* convert_value = reinterpret_cast<float*>(convert_cursor);

                if (use_sst) {
                  if (last_k >= k) {
                    VLOG(0) << "[last_k: " << last_k << "][k: " << k
                            << "][shard_idx: " << shard_idx
                            << "][file_split_idx: " << file_split_idx << "]"
                            << value[0];
                    abort();
                  }
                  last_k = k;
                  _value_accesor->UpdatePassId(convert_value, 0);
                  rocksdb::Status status = sst_writer.Put(
                      rocksdb::Slice(reinterpret_cast<char*>(&k),
                                     sizeof(uint64_t)),
                      rocksdb::Slice(reinterpret_cast<char*>(convert_value),
                                     dim * sizeof(float)));
                  if (!status.ok()) {
                    VLOG(0) << "fatal in Put file: " << filename;
                    abort();
                  }
                  ssd_count += 1;
                  if (dim > feature_value_size - mf_value_size) {
                    ssd_mf_count++;
                  }
                } else {
                  auto& feature_value = shard[k];
                  _value_accesor->UpdatePassId(convert_value, 0);
                  feature_value.resize(dim);
                  memcpy(const_cast<float*>(feature_value.data()),
                         convert_value,
                         dim * sizeof(float));
                  mem_count += 1;
                  if (dim > feature_value_size - mf_value_size) {
                    mem_mf_count++;
                  }
                }
                cursor += len;
                convert_cursor += dim * sizeof(float);
                ret -= len;
              } else {
                memcpy(buf, cursor, ret);
                break;
              }
            } else {
              memcpy(buf, cursor, ret);
              break;
            }
          } while (ret);
        }
        if (use_sst) {
          rocksdb::Status status = sst_writer.Finish();
          if (!status.ok()) {
            VLOG(0) << "fatal in finish file: " << filename << ", "
                    << status.getState();
            abort();
          }
        }
        free(buf);
        free(convert_buf);
        // read_channel->close();
        // VLOG(0) << "[last_k: " << last_k << "][remain: " << remain
        //         << "][shard_idx: " << shard_idx
        //         << "][file_split_idx: " << file_split_idx << "]";
        VLOG(0) << "Table " << filename << " load done. ALL["
                << mem_count + ssd_count << "] MEM[" << mem_count << "] MEM_MF["
                << mem_mf_count << "] SSD[" << ssd_count << "] SSD_MF["
                << ssd_mf_count << "].";
        return 0;
      });
      tasks.push_back(std::move(future));
    }
  }
  for (auto& fut : tasks) {
    fut.wait();
  }
  tasks.clear();
  for (int shard_idx = 0; shard_idx < _real_local_shard_num; shard_idx++) {
    auto sst_filelist = _afs_client.list(::paddle::string::format_string(
        "%s_%d/part-*", FLAGS_rocksdb_path.c_str(), shard_idx));
    if (!sst_filelist.empty()) {
      int ret = _db->ingest_externel_file(shard_idx, sst_filelist);
      if (ret) {
        VLOG(0) << "ingest file failed";
        abort();
      }
    }
  }
  uint64_t ssd_key_num = 0;
  _db->get_estimate_key_num(ssd_key_num);
  _cache_tk_size =
      (LocalSize() + ssd_key_num) * _config.sparse_table_cache_rate();
  return 0;
}

std::pair<int64_t, int64_t> SSDSparseTable::PrintTableStat() {
  int64_t feasign_size = LocalSize();
  return {feasign_size, -1};
}

int32_t SSDSparseTable::CacheTable(uint16_t pass_id) {
  std::lock_guard<std::mutex> guard(_table_mutex);
  VLOG(0) << "cache_table";
  std::atomic<uint32_t> count{0};
  std::vector<std::future<int>> tasks;

  double show_threshold = 10000000;

  // 保证cache数据不被淘汰掉
  if (_config.enable_sparse_table_cache()) {
    if (_local_show_threshold < show_threshold) {
      show_threshold = _local_show_threshold;
    }
  }

  if (show_threshold < 500) {
    show_threshold = 500;
  }
  VLOG(0) << " show_threshold:" << show_threshold
          << " ; local_show_threshold:" << _local_show_threshold;
  VLOG(0) << "Table>> origin mem feasign size:" << LocalSize();
  static int cache_table_count = 0;
  ++cache_table_count;
  for (size_t shard_id = 0;
       shard_id < static_cast<size_t>(_real_local_shard_num);
       ++shard_id) {
    // from mem to ssd
    auto fut = _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
        [shard_id, this, &count, show_threshold, pass_id]() -> int {
          rocksdb::Options options;
          options.comparator = _db->get_comparator();
          rocksdb::BlockBasedTableOptions bbto;
          bbto.format_version = 5;
          bbto.use_delta_encoding = false;
          bbto.block_size = 4 * 1024;
          bbto.block_restart_interval = 6;
          bbto.cache_index_and_filter_blocks = false;
          bbto.filter_policy.reset(rocksdb::NewBloomFilterPolicy(15, false));
          bbto.whole_key_filtering = true;
          options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(bbto));
          options.OptimizeLevelStyleCompaction();
          options.keep_log_file_num = 100;
          options.max_log_file_size = 50 * 1024 * 1024;  // 50MB
          options.create_if_missing = true;
          options.use_direct_reads = true;
          options.write_buffer_size = 64 * 1024 * 1024;  // 256MB
          options.max_write_buffer_number = 4;
          options.max_bytes_for_level_base =
              options.max_write_buffer_number * options.write_buffer_size;
          options.min_write_buffer_number_to_merge = 1;
          options.target_file_size_base = 1024 * 1024 * 1024;  // 1024MB
          options.memtable_prefix_bloom_size_ratio = 0.02;
          options.num_levels = 4;
          options.max_open_files = -1;

          options.compression = rocksdb::kNoCompression;

          auto& shard = _local_shards[shard_id];
          if (1) {
            using DataType = shard_type::map_type::iterator;
            std::vector<DataType> datas;
            datas.reserve(shard.size() * 0.8);
            for (auto it = shard.begin(); it != shard.end(); ++it) {
              if (!_value_accesor->SaveMemCache(
                      it.value().data(), 0, show_threshold, pass_id)) {
                datas.emplace_back(it.it);
              }
            }
            count.fetch_add(datas.size(), std::memory_order_relaxed);
            VLOG(0) << "datas size:  " << datas.size();
            {
              // sst文件写入必须有序
              uint64_t show_begin = butil::gettimeofday_ms();
              std::sort(datas.begin(),
                        datas.end(),
                        [](const DataType& a, const DataType& b) {
                          return a->first < b->first;
                        });
              VLOG(0) << "sort shard " << shard_id << ": "
                      << butil::gettimeofday_ms() - show_begin
                      << " ms, num: " << datas.size();
            }

            // 必须做空判断，否则sst_writer.Finish会core掉
            if (!datas.empty()) {
              rocksdb::SstFileWriter sst_writer(rocksdb::EnvOptions(), options);
              std::string filename =
                  ::paddle::string::format_string("%s_%d/cache-%05d.sst",
                                                  FLAGS_rocksdb_path.c_str(),
                                                  shard_id,
                                                  cache_table_count);
              rocksdb::Status status = sst_writer.Open(filename);
              if (!status.ok()) {
                VLOG(0) << "sst writer open " << filename << "failed"
                        << ", " << status.getState();
                abort();
              }
              VLOG(0) << "sst writer open " << filename;

              uint64_t show_begin = butil::gettimeofday_ms();
              for (auto& data : datas) {
                uint64_t tmp_key = data->first;
                FixedFeatureValue& tmp_value =
                    *((FixedFeatureValue*)(void*)(data->second));  // NOLINT
                status = sst_writer.Put(
                    rocksdb::Slice(reinterpret_cast<char*>(&(tmp_key)),
                                   sizeof(uint64_t)),
                    rocksdb::Slice(reinterpret_cast<char*>(tmp_value.data()),
                                   tmp_value.size() * sizeof(float)));
                if (!status.ok()) {
                  VLOG(0) << "fatal in Put file: " << filename << ", "
                          << status.getState();
                  abort();
                }
              }
              status = sst_writer.Finish();
              if (!status.ok()) {
                VLOG(0) << "fatal in finish file: " << filename << ", "
                        << status.getState();
                abort();
              }
              VLOG(0) << "write sst_file shard " << shard_id << ": "
                      << butil::gettimeofday_ms() - show_begin << " ms";
              int ret = _db->ingest_externel_file(shard_id, {filename});
              if (ret) {
                VLOG(0) << "ingest file failed"
                        << ", " << status.getState();
                abort();
              }
            }

            for (auto it = shard.begin(); it != shard.end();) {
              if (!_value_accesor->SaveMemCache(
                      it.value().data(), 0, show_threshold, pass_id)) {
                it = shard.erase(it);
              } else {
                ++it;
              }
            }
          }
          return 0;
        });
    tasks.push_back(std::move(fut));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  tasks.clear();

  VLOG(0) << "Table>> cache ssd count: " << count.load();
  VLOG(0) << "Table>> after update, mem feasign size:" << LocalSize();
  return 0;
}

}  // namespace distributed
}  // namespace paddle
