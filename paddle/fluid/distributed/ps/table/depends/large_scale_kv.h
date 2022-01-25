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
#include "paddle/fluid/distributed/ps/table/depends/initializers.h"
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/pten/backends/dynload/port.h"
#include "paddle/pten/core/utils/rw_lock.h"

namespace paddle {
namespace distributed {

enum Mode { training, infer };

static const int SPARSE_SHARD_BUCKET_NUM_BITS = 6;
static const size_t SPARSE_SHARD_BUCKET_NUM = (size_t)1
                                              << SPARSE_SHARD_BUCKET_NUM_BITS;

struct VALUE {
  explicit VALUE(size_t length)
      : length_(length),
        count_(0),
        unseen_days_(0),
        need_save_(false),
        is_entry_(false) {
    data_.resize(length);
    memset(data_.data(), 0, sizeof(float) * length);
  }

  size_t length_;
  std::vector<float> data_;
  int count_;
  int unseen_days_;  // use to check knock-out
  bool need_save_;   // whether need to save
  bool is_entry_;    // whether knock-in
};

inline bool count_entry(VALUE *value, int threshold) {
  return value->count_ >= threshold;
}

inline bool probility_entry(VALUE *value, float threshold) {
  UniformInitializer uniform = UniformInitializer({"uniform", "0", "0", "1"});
  return uniform.GetValue() >= threshold;
}

class ValueBlock {
 public:
  typedef typename robin_hood::unordered_map<uint64_t, VALUE *> map_type;
  explicit ValueBlock(const std::vector<std::string> &value_names,
                      const std::vector<int> &value_dims,
                      const std::vector<int> &value_offsets,
                      const std::unordered_map<std::string, int> &value_idx,
                      const std::vector<std::string> &init_attrs,
                      const std::string &entry_attr)
      : value_names_(value_names),
        value_dims_(value_dims),
        value_offsets_(value_offsets),
        value_idx_(value_idx) {
    for (size_t x = 0; x < value_dims.size(); ++x) {
      value_length_ += value_dims[x];
    }

    // for Entry
    {
      auto slices = string::split_string<std::string>(entry_attr, ":");
      if (slices[0] == "none") {
        entry_func_ = std::bind(&count_entry, std::placeholders::_1, 0);
        threshold_ = 0;
      } else if (slices[0] == "count_filter_entry") {
        threshold_ = std::stoi(slices[1]);
        entry_func_ =
            std::bind(&count_entry, std::placeholders::_1, threshold_);
      } else if (slices[0] == "probability_entry") {
        threshold_ = std::stof(slices[1]);
        entry_func_ =
            std::bind(&probility_entry, std::placeholders::_1, threshold_);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Not supported Entry Type : %s, Only support [CountFilterEntry, "
            "ProbabilityEntry]",
            slices[0]));
      }
    }

    // for Initializer
    {
      for (auto &attr : init_attrs) {
        auto slices = string::split_string<std::string>(attr, "&");

        if (slices[0] == "gaussian_random") {
          initializers_.emplace_back(
              std::make_shared<GaussianInitializer>(slices));
        } else if (slices[0] == "fill_constant") {
          initializers_.emplace_back(
              std::make_shared<FillConstantInitializer>(slices));
        } else if (slices[0] == "uniform_random") {
          initializers_.emplace_back(
              std::make_shared<UniformInitializer>(slices));
        } else if (slices[0] == "truncated_gaussian_random") {
          initializers_.emplace_back(
              std::make_shared<TruncatedGaussianInitializer>(slices));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s can not be supported", attr));
        }
      }
    }
  }

  ~ValueBlock() {}

  std::vector<float *> Get(const uint64_t &id,
                           const std::vector<std::string> &value_names,
                           const std::vector<int> &value_dims) {
    auto pts = std::vector<float *>();
    pts.reserve(value_names.size());
    auto values = GetValue(id);
    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      PADDLE_ENFORCE_EQ(
          value_dims[i], value_dims_[i],
          platform::errors::InvalidArgument("value dims is not match"));
      pts.push_back(values->data_.data() +
                    value_offsets_.at(value_idx_.at(value_names[i])));
    }
    return pts;
  }

  // pull
  float *Init(const uint64_t &id, const bool with_update = true,
              const int counter = 1) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);

    auto &table = values_[bucket];
    auto res = table.find(id);

    VALUE *value = nullptr;
    if (res == table.end()) {
      value = butil::get_object<VALUE>(value_length_);

      table[id] = value;

    } else {
      value = res->second;
    }

    if (with_update) {
      AttrUpdate(value, counter);
    }
    return value->data_.data();
  }

  VALUE *InitGet(const uint64_t &id, const bool with_update = true,
                 const int counter = 1) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);

    auto &table = values_[bucket];
    auto res = table.find(id);

    VALUE *value = nullptr;
    if (res == table.end()) {
      value = butil::get_object<VALUE>(value_length_);
      // value = _alloc.acquire(value_length_);
      table[id] = value;
    } else {
      value = (VALUE *)(void *)(res->second);  // NOLINT
    }
    return value;
  }

  void AttrUpdate(VALUE *value, const int counter) {
    // update state
    value->unseen_days_ = 0;
    value->count_ += counter;

    if (!value->is_entry_) {
      value->is_entry_ = entry_func_(value);
      if (value->is_entry_) {
        // initialize
        for (size_t x = 0; x < value_names_.size(); ++x) {
          initializers_[x]->GetValue(value->data_.data() + value_offsets_[x],
                                     value_dims_[x]);
        }
        value->need_save_ = true;
      }
    } else {
      value->need_save_ = true;
    }

    return;
  }

  // dont jude if (has(id))
  float *Get(const uint64_t &id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);
    auto &table = values_[bucket];

    // auto &value = table.at(id);
    // return value->data_.data();
    auto res = table.find(id);
    VALUE *value = res->second;
    return value->data_.data();
  }

  // for load, to reset count, unseen_days
  VALUE *GetValue(const uint64_t &id) {
    size_t hash = _hasher(id);
    size_t bucket = compute_bucket(hash);

    auto &table = values_[bucket];
    auto res = table.find(id);
    return res->second;
  }

  bool GetEntry(const uint64_t &id) {
    auto value = GetValue(id);
    return value->is_entry_;
  }

  void SetEntry(const uint64_t &id, const bool state) {
    auto value = GetValue(id);
    value->is_entry_ = state;
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

  void Shrink(const int threshold) {
    for (auto &table : values_) {
      for (auto iter = table.begin(); iter != table.end();) {
        // VALUE* value = (VALUE*)(void*)(iter->second);
        VALUE *value = iter->second;
        value->unseen_days_++;
        if (value->unseen_days_ >= threshold) {
          butil::return_object(iter->second);
          // _alloc.release(iter->second);
          // _alloc.release(value);
          iter = table.erase(iter);
        } else {
          ++iter;
        }
      }
    }
    return;
  }

  float GetThreshold() { return threshold_; }
  size_t compute_bucket(size_t hash) {
    if (SPARSE_SHARD_BUCKET_NUM == 1) {
      return 0;
    } else {
      return hash >> (sizeof(size_t) * 8 - SPARSE_SHARD_BUCKET_NUM_BITS);
    }
  }

  map_type::iterator end() {
    return values_[SPARSE_SHARD_BUCKET_NUM - 1].end();
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
  map_type values_[SPARSE_SHARD_BUCKET_NUM];
  size_t value_length_ = 0;
  std::hash<uint64_t> _hasher;

 private:
  const std::vector<std::string> &value_names_;
  const std::vector<int> &value_dims_;
  const std::vector<int> &value_offsets_;
  const std::unordered_map<std::string, int> &value_idx_;

  std::function<bool(VALUE *)> entry_func_;
  std::vector<std::shared_ptr<Initializer>> initializers_;
  float threshold_;
};

}  // namespace distributed
}  // namespace paddle
