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

#include <ThreadPool.h>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#include "butil/object_pool.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
#include "paddle/fluid/distributed/thirdparty/round_robin.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {


static const int DOWNPOUR_DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS = 6;
static const size_t DOWNPOUR_SPARSE_SHARD_BUCKET_NUM = (size_t)1
                                              << DOWNPOUR_DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS;

class DownpourFixedFeatureValue {
  public:
    DownpourFixedFeatureValue() {}
    ~DownpourFixedFeatureValue() {}
    float* data() {
      return data_.data();
    }
    size_t size() {
      return data_.size();
    }
    void resize(size_t size) {
      data_.resize(size);
    }
    void shrink_to_fit() {
      data_.shrink_to_fit();
    }
  private:
    std::vector<float> data_;
};


class DownpourValueBlock {
 public:
  typedef typename robin_hood::unordered_map<uint64_t, DownpourFixedFeatureValue *> map_type;
  ValueBlock() {}
  ~ValueBlock() {}

  DownpourFixedFeatureValue *Init(const uint64_t &id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];
    
    DownpourFixedFeatureValue *value = nullptr;
    value = butil::get_object<DownpourFixedFeatureValue>();
    table[id] = value;
    return value;
  }

  // dont judge if (has(id))
  float *Get(const uint64_t &id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];

    // auto &value = table.at(id);
    // return value->data_.data();
    auto res = table.find(id);
    DownpourFixedFeatureValue *value = res->second;
    return value->data();
  }

  // TODO: whether need this?
  // for load, to reset count, unseen_days
  DownpourFixedFeatureValue *GetValue(const uint64_t &id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);

    auto &table = values_[bucket];
    auto res = table.find(id);
    return res->second;
  }

  void erase(uint64_t feasign) {
    size_t hash = _hasher(feasign);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];

    auto iter = table.find(feasign);
    if (iter != table.end()) {
      butil::return_object(iter->second);
      iter = table.erase(iter);
    }
  }

  // TODO: whether need this?
  void clear() {
  }

  size_t compute_bucket(size_t hash) {
    if (DOWNPOUR_SPARSE_SHARD_BUCKET_NUM == 1) {
      return 0;
    } else {
      return hash >> (sizeof(size_t) * 8 - DOWNPOUR_DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS);
    }
  }
  
  map_type::iterator end() {
    return values_[DOWNPOUR_SPARSE_SHARD_BUCKET_NUM - 1].end();
  }

  map_type::iterator Find(uint64_t id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];

    auto got = table.find(id);
    if (got == table.end()) {
      return end();
    } else {
      return got;
    }
  }

 private:
  bool Has(const uint64_t id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];

    auto got = table.find(id);
    if (got == table.end()) {
      return false;
    } else {
      return true;
    }
  }

 public:
  map_type values_[DOWNPOUR_SPARSE_SHARD_BUCKET_NUM];
  std::hash<uint64_t> _hasher;

};

}  // namespace distributed
}  // namespace paddle
