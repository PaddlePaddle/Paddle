// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <future>  // NOLINT
#include <memory>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace distributed {

class ConcurrentSet {
 public:
  ConcurrentSet() : pool_(new ::ThreadPool(1)) {}
  ~ConcurrentSet() {}

  std::future<void> Update(const std::vector<uint64_t>& rows) {
    auto task = [this, rows] {
      for (auto row : rows) {
        set_.insert(row);
      }
    };
    return pool_->enqueue(std::move(task));
  }

  std::future<void> GetAndClear(std::vector<uint64_t>* result) {
    auto task = [this, &result] {
      result->clear();
      for (auto& id : set_) {
        result->push_back(id);
      }
      set_.clear();
    };
    return pool_->enqueue(std::move(task));
  }

 private:
  std::unordered_set<uint64_t> set_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
};

class GeoRecorder {
 public:
  explicit GeoRecorder(int trainer_num) : trainer_num_(trainer_num) {
    trainer_rows_.reserve(trainer_num);
    for (auto i = 0; i < trainer_num; ++i) {
      trainer_rows_.emplace_back(new ConcurrentSet());
    }
  }

  ~GeoRecorder() = default;

  void Update(const std::vector<uint64_t>& update_rows) {
    VLOG(3) << " row size: " << update_rows.size();

    std::vector<std::future<void>> fs;
    for (auto& set : trainer_rows_) {
      fs.push_back(set->Update(update_rows));
    }
    for (auto& f : fs) {
      f.wait();
    }
  }

  void GetAndClear(uint32_t trainer_id, std::vector<uint64_t>* result) {
    VLOG(3) << "GetAndClear for trainer: " << trainer_id;
    trainer_rows_.at(trainer_id)->GetAndClear(result).wait();
  }

 private:
  const int trainer_num_;
  std::vector<std::unique_ptr<ConcurrentSet>> trainer_rows_;
};

}  // namespace distributed
}  // namespace paddle
