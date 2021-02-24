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
#include <gflags/gflags.h>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
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

enum Mode { training, infer };

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

inline bool count_entry(std::shared_ptr<VALUE> value, int threshold) {
  return value->count_ >= threshold;
}

inline bool probility_entry(std::shared_ptr<VALUE> value, float threshold) {
  UniformInitializer uniform = UniformInitializer({"uniform", "0", "0", "1"});
  return uniform.GetValue() >= threshold;
}

class ValueBlock {
 public:
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
    for (int x = 0; x < value_dims.size(); ++x) {
      value_length_ += value_dims[x];
    }

    // for Entry
    {
      auto slices = string::split_string<std::string>(entry_attr, ":");
      if (slices[0] == "none") {
        entry_func_ = std::bind(&count_entry, std::placeholders::_1, 0);
      } else if (slices[0] == "count_filter_entry") {
        int threshold = std::stoi(slices[1]);
        entry_func_ = std::bind(&count_entry, std::placeholders::_1, threshold);
      } else if (slices[0] == "probability_entry") {
        float threshold = std::stof(slices[1]);
        entry_func_ =
            std::bind(&probility_entry, std::placeholders::_1, threshold);
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
    auto &values = values_.at(id);
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
  float *Init(const uint64_t &id, const bool with_update = true) {
    if (!Has(id)) {
      values_[id] = std::make_shared<VALUE>(value_length_);
    }

    auto &value = values_.at(id);

    if (with_update) {
      AttrUpdate(value);
    }

    return value->data_.data();
  }

  void AttrUpdate(std::shared_ptr<VALUE> value) {
    // update state
    value->unseen_days_ = 0;
    ++value->count_;

    if (!value->is_entry_) {
      value->is_entry_ = entry_func_(value);
      if (value->is_entry_) {
        // initialize
        for (int x = 0; x < value_names_.size(); ++x) {
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
    auto &value = values_.at(id);
    return value->data_.data();
  }

  // for load, to reset count, unseen_days
  std::shared_ptr<VALUE> GetValue(const uint64_t &id) { return values_.at(id); }

  bool GetEntry(const uint64_t &id) {
    auto &value = values_.at(id);
    return value->is_entry_;
  }

  void SetEntry(const uint64_t &id, const bool state) {
    auto &value = values_.at(id);
    value->is_entry_ = state;
  }

  void Shrink(const int threshold) {
    for (auto iter = values_.begin(); iter != values_.end();) {
      auto &value = iter->second;
      value->unseen_days_++;
      if (value->unseen_days_ >= threshold) {
        iter = values_.erase(iter);
      } else {
        ++iter;
      }
    }
    return;
  }

 private:
  bool Has(const uint64_t id) {
    auto got = values_.find(id);
    if (got == values_.end()) {
      return false;
    } else {
      return true;
    }
  }

 public:
  std::unordered_map<uint64_t, std::shared_ptr<VALUE>> values_;
  size_t value_length_ = 0;

 private:
  const std::vector<std::string> &value_names_;
  const std::vector<int> &value_dims_;
  const std::vector<int> &value_offsets_;
  const std::unordered_map<std::string, int> &value_idx_;

  std::function<bool(std::shared_ptr<VALUE>)> entry_func_;
  std::vector<std::shared_ptr<Initializer>> initializers_;
};

}  // namespace distributed
}  // namespace paddle
