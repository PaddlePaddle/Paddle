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

#pragma once

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_sparse_table.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/distributed/table/depends/geo_recorder.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/distributed/table/depends/sparse.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class GeoRecorder;

class SparseGeoTable : public CommonSparseTable {
 public:
  explicit SparseGeoTable() : CommonSparseTable() { geo_recorder = nullptr; }
  virtual ~SparseGeoTable() {}

  virtual int32_t initialize_value();

  int32_t pull_geo_param(const uint32_t trainer_id, std::vector<float>* values,
                         std::vector<uint64_t>* keys);

  int32_t push_sparse(const uint64_t* keys, const float* values,
                      size_t num) override;

  virtual int32_t initialize_recorder() {
    if (!geo_recorder) {
      auto trainers = _config.common().trainer_num();
      geo_recorder = std::make_shared<GeoRecorder>(trainers);
    }
    return 0;
  }

 private:
  std::shared_ptr<GeoRecorder> geo_recorder;
};

}  // namespace distributed
}  // namespace paddle
