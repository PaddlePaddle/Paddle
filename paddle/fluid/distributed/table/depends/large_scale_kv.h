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

template <typename T>
inline bool entry(const int count, const T threshold);

template <>
inline bool entry<std::string>(const int count, const std::string threshold) {
  return true;
}

template <>
inline bool entry<int>(const int count, const int threshold) {
  return count >= threshold;
}

template <>
inline bool entry<float>(const int count, const float threshold) {
  UniformInitializer uniform = UniformInitializer({"0", "0", "1"});
  return uniform.GetValue() >= threshold;
}

struct VALUE {
  explicit VALUE(size_t length)
      : length_(length),
        count_(1),
        unseen_days_(0),
        seen_after_last_save_(true),
        is_entry_(true) {
    data_.resize(length);
  }

  size_t length_;
  std::vector<float> data_;
  int count_;
  int unseen_days_;
  bool seen_after_last_save_;
  bool is_entry_;
};

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
      if (entry_attr == "none") {
        has_entry_ = false;
        entry_func_ =
            std::bind(entry<std::string>, std::placeholders::_1, "none");
      } else {
        has_entry_ = true;
        auto slices = string::split_string<std::string>(entry_attr, "&");
        if (slices[0] == "count_filter") {
          int threshold = std::stoi(slices[1]);
          entry_func_ = std::bind(entry<int>, std::placeholders::_1, threshold);
        } else if (slices[0] == "probability") {
          float threshold = std::stof(slices[1]);
          entry_func_ =
              std::bind(entry<float>, std::placeholders::_1, threshold);
        }
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

  float *Init(const uint64_t &id) {
    auto value = std::make_shared<VALUE>(value_length_);
    for (int x = 0; x < value_names_.size(); ++x) {
      initializers_[x]->GetValue(value->data_.data() + value_offsets_[x],
                                 value_dims_[x]);
    }
    values_[id] = value;
    return value->data_.data();
  }

  std::vector<float *> Get(const uint64_t &id,
                           const std::vector<std::string> &value_names) {
    auto pts = std::vector<float *>();
    pts.reserve(value_names.size());
    auto &values = values_.at(id);
    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      pts.push_back(values->data_.data() +
                    value_offsets_.at(value_idx_.at(value_names[i])));
    }
    return pts;
  }

  float *Get(const uint64_t &id) {
    auto pts = std::vector<std::vector<float> *>();
    auto &values = values_.at(id);

    return values->data_.data();
  }

  float *InitFromInitializer(const uint64_t &id) {
    if (Has(id)) {
      if (has_entry_) {
        Update(id);
      }
      return Get(id);
    }
    return Init(id);
  }

  bool GetEntry(const uint64_t &id) {
    auto value = values_.at(id);
    return value->is_entry_;
  }

  void Update(const uint64_t id) {
    auto value = values_.at(id);
    value->unseen_days_ = 0;
    auto count = ++value->count_;

    if (!value->is_entry_) {
      value->is_entry_ = entry_func_(count);
    }
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

  bool has_entry_ = false;
  std::function<bool(uint64_t)> entry_func_;
  std::vector<std::shared_ptr<Initializer>> initializers_;
};

}  // namespace distributed
}  // namespace paddle
