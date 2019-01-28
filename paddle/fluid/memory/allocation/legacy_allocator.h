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
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace memory {
namespace allocation {

struct LegacyMemMonitor {
  // used to store the GPU memory usage of each devices
  using MemUsage =
      std::unordered_map</*device id*/ int,
                         std::pair</*current memory usage*/ uint64_t,
                                   /*peak memory usage*/ uint64_t>>;

  MemUsage GetMemUsageInfo() { return gpu_mem_info_; }

  void Add(int device, size_t size) {
    gpu_mem_info_[device].first += size;
    if (gpu_mem_info_[device].first > gpu_mem_info_[device].second) {
      gpu_mem_info_[device].second = gpu_mem_info_[device].first;
      VLOG(3) << "device: " << device
              << " peak memory usage : " << (gpu_mem_info_[device].second >> 20)
              << " MiB";
    }
  }

  void Minus(int device, size_t size) { gpu_mem_info_[device].first -= size; }

  uint64_t GetMemUsage(int device) { return gpu_mem_info_[device].second; }

  void PrintMemUsage() {
    std::vector<int> devices;
    for (const auto &item : gpu_mem_info_) {
      devices.emplace_back(item.first);
    }
    std::sort(devices.begin(), devices.end());
    for (const auto &device : devices) {
      std::cout << "Device : " << device << " Peak Memory Usage : "
                << (gpu_mem_info_[device].second >> 20) << " MiB" << std::endl;
    }
  }

 private:
  MemUsage gpu_mem_info_;
};

extern LegacyMemMonitor GPUMemMonitor;

class LegacyAllocatorPrivate;
class LegacyAllocator : public Allocator {
 public:
  explicit LegacyAllocator(const platform::Place &p) : place_(p) {}

 protected:
  Allocation *AllocateImpl(size_t size, Allocator::Attr attr) override;
  void Free(Allocation *allocation) override;

 private:
  platform::Place place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
