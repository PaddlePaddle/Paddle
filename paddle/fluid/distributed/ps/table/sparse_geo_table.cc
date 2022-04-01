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

#include "paddle/fluid/distributed/ps/table/sparse_geo_table.h"

namespace paddle {
namespace distributed {

int32_t SparseGeoTable::pull_geo_param(const uint32_t trainer_id,
                                       std::vector<float>* values,
                                       std::vector<uint64_t>* ids) {
  geo_recorder->GetAndClear(trainer_id, ids);
  auto dim = _config.common().dims()[0];

  std::vector<uint32_t> frequencies;
  frequencies.resize(ids->size(), 1);

  auto pull_value = PullSparseValue(ids->size(), dim);
  pull_value.is_training_ = true;
  pull_value.feasigns_ = ids->data();
  pull_value.frequencies_ = frequencies.data();

  values->resize(ids->size() * dim);
  CommonSparseTable::pull_sparse(values->data(), pull_value);
  return 0;
}

int32_t SparseGeoTable::push_sparse(const uint64_t* keys, const float* values,
                                    size_t num) {
  std::vector<uint64_t> ids;
  ids.resize(num);
  std::copy_n(keys, num, ids.begin());
  geo_recorder->Update(ids);
  CommonSparseTable::push_sparse(keys, values, num);
  return 0;
}

int32_t SparseGeoTable::initialize_value() {
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

  VLOG(3) << "has " << feasigns.size() << " ids need to be pre inited";

  auto buckets = bucket(feasigns.size(), 10);
  for (int x = 0; x < 10; ++x) {
    auto bucket_feasigns = buckets[x + 1] - buckets[x];
    std::vector<uint64_t> ids(bucket_feasigns);
    std::copy(feasigns.begin() + buckets[x], feasigns.begin() + buckets[x + 1],
              ids.begin());

    std::vector<uint32_t> fres;
    fres.resize(ids.size(), 1);

    auto pull_value = PullSparseValue(ids, fres, param_dim_);
    std::vector<float> pulls;
    pulls.resize(bucket_feasigns * param_dim_);
    pull_sparse(pulls.data(), pull_value);
  }
  return 0;
}

}  // namespace distributed
}  // namespace paddle
