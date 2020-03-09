//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>
#include "glog/logging.h"

namespace paddle {
namespace platform {
class Monitor {
 public:
  Monitor() {}
  virtual ~Monitor() {}

  static Monitor* Instance() {
    if (nullptr == instance) {
      VLOG(0) << "Init Monitor Instance";
      instance = new Monitor();
    }
    return instance;
  }

  void LogCudaMalloc(int device_id, size_t size) {
    std::lock_guard<std::mutex> g(mutex_);
    stats_.gpu_mem_size_[device_id] += size;
  }

  void LogCudaFree(int device_id, size_t size) {
    std::lock_guard<std::mutex> g(mutex_);
    stats_.gpu_mem_size_[device_id] -= size;
  }

 public:
  struct MonitorStats {
    MonitorStats() {
      gpu_mem_size_.resize(8, 0);  // more than 8 GPUs in a node?
      nets_in_ = 0;
      nets_out_ = 0;
      total_feasign_num_in_mem_ = 0;
    }
    // System
    std::vector<size_t> gpu_mem_size_;
    std::atomic<size_t> nets_in_;
    std::atomic<size_t> nets_out_;

    // Application
    std::atomic<size_t> total_feasign_num_in_mem_;
  };

  MonitorStats stats_;

 private:
  std::mutex mutex_;
  static Monitor* instance;
};

}  // namespace platform
}  // namespace paddle
