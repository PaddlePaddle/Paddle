/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>         // NOLINT
#include <unordered_map>  // NOLINT
#include <unordered_set>  // NOLINT
#include <utility>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/platform/timer.h"
namespace paddle {
namespace framework {
class DataFeed;
enum HeterTaskState { PULL_SPARSE, OP_RUN, XPU, OP_RUN_END, PUSH_GRAD, DONE };

class HeterTask {
 public:
  HeterTask() {}
  virtual ~HeterTask() {}

  void Update() {
    if (state_ == PULL_SPARSE) {
      state_ = OP_RUN;
    } else if (state_ == OP_RUN) {
      state_ = XPU;
      // state_ = PUSH_GRAD;
      // state_ = PUSH_GRAD;
    } else if (state_ == XPU) {
      state_ = OP_RUN_END;
    } else if (state_ == OP_RUN_END) {
      state_ = PUSH_GRAD;
    } else if (state_ == PUSH_GRAD) {
      state_ = DONE;
    }
  }
  void Reset() {
    total_time = 0;
    read_time = 0;
    pack_time = 0;
    pull_sparse_local_time = 0;
    op_all_time = 0;
    xpu_op_time = 0;
    xpu_wait_time = 0;
    cpu_op_time = 0;
    collect_label_time = 0;
    fill_sparse_time = 0;
    push_sparse_time = 0;
    gpu_2_cpu_time = 0;
    cpu_2_gpu_time = 0;
    timeline.Reset();
  }
  void Show() {
    std::cout << "features size " << features_.size() << std::endl;
    for (size_t i = 0; i < features_.size(); ++i) {
      std::cout << "features[" << i << "] size " << features_[i].size()
                << std::endl;
    }
  }
  void PackTask(Scope* scope,
                int taskid,
                DataFeed* reader,
                int cur_batch,
                const ProgramDesc& program);
  void PackGpuTask(Scope* thread_scope,
                   DataFeed* reader,
                   const ProgramDesc& program);

  Scope* scope_{nullptr};
  int taskid_;
  int cur_batch_;
  HeterTaskState state_;
  // cache
  std::map<uint64_t, std::vector<uint64_t>> features_;
  std::map<uint64_t, std::vector<float>> feature_labels_;
  std::map<uint64_t, std::vector<std::vector<float>>> feature_values_;
  std::map<uint64_t, std::vector<std::vector<float>>> feature_grads_;
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;
  double total_time{0};
  double read_time{0};
  double pack_time{0};
  double pull_sparse_local_time{0};
  double op_all_time{0};
  double xpu_op_time{0};
  double xpu_wait_time{0};
  double cpu_op_time{0};
  double collect_label_time{0};
  double fill_sparse_time{0};
  double push_sparse_time{0};
  double gpu_2_cpu_time{0};
  double cpu_2_gpu_time{0};
  platform::Timer timeline;
};

template <class T>
class HeterObjectPool {
 public:
  HeterObjectPool() {}
  virtual ~HeterObjectPool() {}
  std::shared_ptr<T> Get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) {
      num_ += 1;
      return std::make_shared<T>();
    } else {
      auto ret = pool_.back();
      pool_.pop_back();
      return ret;
    }
  }
  void Push(std::shared_ptr<T> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(std::move(data));
  }
  int Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
  }
  bool Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.empty();
  }
  std::shared_ptr<T>& GetElement(int i) { return pool_[i]; }

 private:
  std::vector<std::shared_ptr<T>> pool_;
  std::mutex mutex_;
  int num_{0};
};

}  // namespace framework
}  // namespace paddle
