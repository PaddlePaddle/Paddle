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

#include "paddle/fluid/distributed/ps/table/memory_sparse_geo_table.h"

namespace paddle {
namespace distributed {

int32_t MemorySparseGeoTable::push_sparse_param(const uint64_t* keys,
                                                const float* values,
                                                size_t num) {
  VLOG(5) << "DEBUG MemorySparseGeoTable::push_sparse_param begin "
             "push_sparse_param "
          << num;
  auto shard_num = _task_pool_size;
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(shard_num);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % shard_num;
    offset_bucket[y].push_back(x);
    if (x < 10) {
      VLOG(5) << "DEBUG MemorySparseGeoTable::push_sparse_param key: "
              << keys[x] << " shard: " << y;
    }
  }

  std::vector<std::future<int>> tasks(shard_num);

  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &values]() -> int {
          auto& local_shard = _local_shards[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            auto& feature_value = local_shard[id];
            feature_value.resize(_dim);
            std::copy_n(values + _dim * offset, _dim, feature_value.data());
            if (i < 10) {
              VLOG(5) << "MemorySparseGeoTable::push_sparse_param "
                         "push_sparse_param key "
                      << id << " value[0]: " << (values + _dim * offset)[0]
                      << " data: " << feature_value.data()[0]
                      << " value[-1]: " << (values + _dim * offset)[_dim - 1]
                      << " data: " << feature_value.data()[_dim - 1];
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

int32_t MemorySparseGeoTable::pull_geo_param(const uint32_t trainer_id,
                                             std::vector<float>* values,
                                             std::vector<uint64_t>* ids) {
  _geo_recorder->GetAndClear(trainer_id, ids);
  VLOG(5)
      << "DEBUG MemorySparseGeoTable::pull_geo_param pull_geo_param trainer_id "
      << trainer_id << " id_num: " << ids->size();

  std::vector<uint32_t> frequencies;
  frequencies.resize(ids->size(), 1);

  auto pull_value = PullSparseValue(ids->size(), _dim);
  pull_value.is_training_ = true;
  pull_value.feasigns_ = ids->data();
  pull_value.frequencies_ = frequencies.data();

  values->resize(ids->size() * _dim);
  pull_sparse(values->data(), pull_value);
  return 0;
}

int32_t MemorySparseGeoTable::push_sparse(const uint64_t* keys,
                                          const float* values, size_t num) {
  VLOG(5) << "DEBUG MemorySparseGeoTable::push_sparse keys[0]" << keys[0]
          << " key_num: " << num;
  std::vector<uint64_t> ids;
  ids.resize(num);
  std::copy_n(keys, num, ids.begin());
  _geo_recorder->Update(ids);
  _push_sparse(keys, values, num);
  return 0;
}

int32_t MemorySparseGeoTable::initialize() {
  if (!_geo_recorder) {
    auto trainers = _config.common().trainer_num();
    _geo_recorder = std::make_shared<GeoRecorder>(trainers);
  }

  _dim = _config.common().dims()[0];
  _shards_task_pool.resize(_task_pool_size);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  _local_shards.reset(new shard_type[_task_pool_size]);
  return 0;
}

int32_t MemorySparseGeoTable::pull_sparse(float* pull_values,
                                          const PullSparseValue& pull_value) {
  auto shard_num = _task_pool_size;
  std::vector<std::future<int>> tasks(shard_num);

  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(shard_num);
  size_t num = pull_value.numel_;
  for (size_t i = 0; i < num; ++i) {
    int shard_id = pull_value.feasigns_[i] % shard_num;
    task_keys[shard_id].push_back({pull_value.feasigns_[i], i});
  }

  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &task_keys, pull_values]() -> int {
          auto& local_shard = _local_shards[shard_id];
          auto& keys = task_keys[shard_id];
          for (size_t i = 0; i < keys.size(); i++) {
            uint64_t key = keys[i].first;
            auto offset = keys[i].second;
            float* select_data = pull_values + _dim * offset;

            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              // ++missed_keys;
              auto& feature_value = local_shard[key];
              feature_value.resize(_dim);
              memset(feature_value.data(), 0, sizeof(float) * _dim);
              VLOG(0) << "MemorySparseGeoTable pull_sparse key not found!!! "
                      << key;
              itr = local_shard.find(key);
            }
            memcpy(select_data, itr.value().data(), _dim * sizeof(float));

            VLOG(5) << "DEBUG MemorySparseGeoTable::pull_sparse key: " << key
                    << " select_data[0] " << select_data[0]
                    << " value[0]: " << itr.value().data()[0];
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }

  return 0;
}

int32_t MemorySparseGeoTable::_push_sparse(const uint64_t* keys,
                                           const float* values, size_t num) {
  auto shard_num = _task_pool_size;
  std::vector<std::future<int>> tasks(shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = keys[i] % shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }

  for (size_t shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, values, &task_keys]() -> int {
          auto& keys = task_keys[shard_id];
          auto& local_shard = _local_shards[shard_id];
          auto blas = GetBlas<float>();

          for (int i = 0; i < keys.size(); ++i) {
            uint64_t key = keys[i].first;
            uint64_t push_data_idx = keys[i].second;
            const float* update_data = values + push_data_idx * _dim;
            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              VLOG(0) << "sparse geo table push not found key!!! " << key;
              auto& feature_value = local_shard[key];
              feature_value.resize(_dim);
              memset(feature_value.data(), 0, sizeof(float) * _dim);
              itr = local_shard.find(key);
            }

            auto& feature_value = itr.value();
            float* value_data = feature_value.data();
            VLOG(5) << "DEBUG MemorySparseGeoTable::_push_sparse before key: "
                    << key << " update_data[0] " << update_data[0]
                    << " value[0]: " << value_data[0];
            blas.VADD(_dim, update_data, value_data, value_data);
            VLOG(5) << "DEBUG MemorySparseGeoTable::_push_sparse after key: "
                    << key << " value[0]: " << value_data[0];
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

}  // namespace distributed
}  // namespace paddle
