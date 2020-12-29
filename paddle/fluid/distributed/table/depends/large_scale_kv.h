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
      : data_(new float[length]),
        length_(length),
        count_(1),
        unseen_days_(0),
        seen_after_last_save_(true),
        is_entry_(true) {}

  ~VALUE() {
    if (data_) {
      delete[] static_cast<float *>(data_);
      data_ = nullptr;
      length_ = 0;
      count_ = 0;
      unseen_days_ = 0;
      seen_after_last_save_ = true;
      is_entry_ = true;
    }
  }

  float *data_;
  size_t length_;
  int count_;
  int unseen_days_;
  bool seen_after_last_save_;
  bool is_entry_;
};

class ValueBlock {
 public:
  explicit ValueBlock(
      const CommonAccessorParameter &common,
      std::unordered_map<std::string, Initializer *> *initializers) {
    initializers_ = initializers;
    int size = static_cast<int>(common.params().size());

    value_length_ = 0;
    for (int x = 0; x < size; ++x) {
      auto varname = common.params()[x];
      auto dim = common.dims()[x];

      value_names_.push_back(varname);
      value_dims_.push_back(dim);
      value_offsets_.push_back(value_length_);
      value_length_ += dim;
      value_idx_[varname] = x;
    }

    for (auto &name : value_names_) {
      initializer_list_.emplace_back(initializers_->at(name));
    }

    // for Entry
    {
      // entry will add later
      std::string entry_attr = "none";
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
  }

  ~ValueBlock() {}

  void Init(const uint64_t &id) {
    if (Has(id)) {
      PADDLE_THROW(platform::errors::AlreadyExists("id already exist, error"));
    }

    auto value = std::make_shared<VALUE>(value_length_);

    for (int x = 0; x < value_names_.size(); ++x) {
      initializer_list_[x]->GetValue(value->data_ + value_offsets_[x],
                                     value_dims_[x]);
    }
    values_[id] = value;
  }

  std::vector<float *> Get(const uint64_t &id,
                           const std::vector<std::string> &value_names) {
    auto pts = std::vector<float *>();
    pts.reserve(value_names.size());
    auto &values = values_.at(id);
    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      pts.push_back(values->data_ + value_idx_.at(value_names[i]));
    }
    return pts;
  }

  float *Get(const uint64_t &id) {
    auto pts = std::vector<std::vector<float> *>();
    auto &values = values_.at(id);

    return values->data_;
  }

  void InitFromInitializer(const uint64_t &id) {
    if (Has(id)) {
      if (has_entry) {
        Update(id);
      }
      return;
    }

    Init(id);
  }

  bool GetEntry(const uint64_t &id) {
    auto value = values_.at(id);
    return value->is_entry_;
  }

  void Update(const uint64_t id) {
    auto value = values_.at(id);
    value->reset_unseen_days();
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
  std::unordered_map<uint64_t, std::shard_ptr<VALUE>> values_;
  size_t value_length_ = 0;

 private:
  std::vector<std::string> value_names_;
  std::vector<int> value_dims_;
  std::vector<int> value_offsets_;
  std::unordered_map<std::string, int> value_idx_;

  bool has_entry_ = false;
  std::function<bool(uint64_t)> entry_func_;

  std::unordered_map<std::string, Initializer *> *initializers_;
  std::vector<Initializer *> initializer_list_;
};

}  // namespace distributed
}  // namespace paddle
