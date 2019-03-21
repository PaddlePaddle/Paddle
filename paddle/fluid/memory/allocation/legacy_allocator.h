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
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace memory {
namespace allocation {

class MemInfo {
 public:
  MemInfo() : usage_(0), peak_usage_(0) {}

  // return a flag to indicate current operation will create a peak point or not
  bool Add(const size_t &);
  void Minus(const size_t &);

  uint64_t GetPeakUsage() const;

 private:
  /* current memory usage*/
  uint64_t usage_;
  uint64_t peak_usage_;
  std::mutex mutex_;

  DISABLE_COPY_AND_ASSIGN(MemInfo);
};

class LegacyMemMonitor {
 public:
  // used to store the GPU memory usage of each devices
  using MemUsage = std::unordered_map</*device id*/ int,
                                      /*mem usage info node*/ MemInfo *>;

  MemUsage GetMemUsageInfo() { return gpu_mem_info_; }
  ~LegacyMemMonitor();

  void Initialize(const int &);
  void Add(const int &, const size_t &);
  void Minus(const int &, const size_t &);

  uint64_t GetMemUsage(const int &) const;

  void PrintMemUsage();

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
