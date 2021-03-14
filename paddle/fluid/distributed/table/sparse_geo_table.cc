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

#include "paddle/fluid/distributed/table/sparse_geo_table.h"

namespace paddle {
namespace distributed {

int32_t SparseGeoTable::pull_geo_param(const uint32_t trainer_id,
                                       std::vector<float>* values,
                                       std::vector<uint64_t>* ids) {
  geo_recorder->GetAndClear(trainer_id, ids);
  auto dim = _config.common().dims()[0];
  values->resize(ids->size() * dim);
  CommonSparseTable::pull_sparse(values->data(), ids->data(), ids->size());
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

}  // namespace distributed
}  // namespace paddle
