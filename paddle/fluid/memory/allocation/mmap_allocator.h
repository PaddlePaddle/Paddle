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

#ifndef _WIN32

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class MemoryMapWriterAllocation : public Allocation {
 public:
  explicit MemoryMapWriterAllocation(void *ptr, size_t size,
                                     std::string ipc_name)
      : Allocation(ptr, size, platform::CPUPlace()),
        ipc_name_(std::move(ipc_name)) {}

  inline const std::string &ipc_name() const { return ipc_name_; }

  ~MemoryMapWriterAllocation() override;

 private:
  std::string ipc_name_;
};

class MemoryMapReaderAllocation : public Allocation {
 public:
  explicit MemoryMapReaderAllocation(void *ptr, size_t size,
                                     std::string ipc_name)
      : Allocation(ptr, size, platform::CPUPlace()),
        ipc_name_(std::move(ipc_name)) {}

  inline const std::string &ipc_name() const { return ipc_name_; }

  ~MemoryMapReaderAllocation() override;

 private:
  std::string ipc_name_;
};

std::shared_ptr<MemoryMapWriterAllocation> AllocateMemoryMapWriterAllocation(
    size_t size);

std::shared_ptr<MemoryMapReaderAllocation> RebuildMemoryMapReaderAllocation(
    const std::string &ipc_name, size_t size);

class MemoryMapFdSet {
 public:
  static MemoryMapFdSet &Instance();  // NOLINT

  void Insert(const std::string &ipc_name);

  void Remove(const std::string &ipc_name);

  void Clear();

  ~MemoryMapFdSet();

 private:
  MemoryMapFdSet() = default;

  std::unordered_set<std::string> fd_set_;
  std::mutex mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
