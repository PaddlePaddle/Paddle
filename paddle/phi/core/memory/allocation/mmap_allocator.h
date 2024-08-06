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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/phi/core/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

std::string GetIPCName();

static constexpr int64_t mmap_alignment = 64;

enum MappedModes {
  MAPPED_SHAREDMEM = 1,
  MAPPED_EXCLUSIVE = 2,
  MAPPED_NOCREATE = 4,
  MAPPED_KEEPFD = 8,
  MAPPED_FROMFD = 16,
  MAPPED_UNLINK = 32
};

class MemoryMapAllocation : public Allocation {
 public:
  explicit MemoryMapAllocation(void *ptr,
                               size_t size,
                               std::string ipc_name,
                               int fd)
      : Allocation(ptr, size, phi::CPUPlace()),
        ipc_name_(std::move(ipc_name)),
        fd_(fd),
        map_ptr_(ptr),
        map_size_(size) {}
  explicit MemoryMapAllocation(
      void *ptr, size_t size, std::string ipc_name, int fd, int flags)
      : Allocation(ptr, size, phi::CPUPlace()),
        ipc_name_(std::move(ipc_name)),
        fd_(fd),
        flags_(flags),
        map_ptr_(ptr),
        map_size_(size) {}

  inline const std::string &ipc_name() const { return ipc_name_; }
  inline const int shared_fd() const { return fd_; }

  virtual void close();

  ~MemoryMapAllocation() override;

 protected:
  std::string ipc_name_;
  int fd_ = -1;
  int flags_ = 0;
  void *map_ptr_ = nullptr;
  size_t map_size_ = 0;
  bool closed_ = false;
  bool closed_fd_ = false;
};

class RefcountedMemoryMapAllocation : public MemoryMapAllocation {
 public:
  RefcountedMemoryMapAllocation(void *ptr,
                                size_t size,
                                std::string ipc_name,
                                int flags,
                                int fd,
                                int buffer_id = -1);

  void incref();
  int decref();
  void close() override;
  virtual ~RefcountedMemoryMapAllocation() { close(); }

 protected:
  int buffer_id_ = -1;
  void initializeRefercount();
  void resetBaseptr();
};

void AllocateMemoryMap(std::string filename,
                       int *shared_fd,
                       int flags,
                       size_t size,
                       void **base_ptr_);

std::shared_ptr<RefcountedMemoryMapAllocation>
AllocateRefcountedMemoryMapAllocation(std::string filename,
                                      int shared_fd,
                                      int flags,
                                      size_t size,
                                      int buffer_id = -1);

class MemoryMapWriterAllocation : public Allocation {
 public:
  explicit MemoryMapWriterAllocation(void *ptr,
                                     size_t size,
                                     std::string ipc_name)
      : Allocation(ptr, size, phi::CPUPlace()),
        ipc_name_(std::move(ipc_name)) {}

  inline const std::string &ipc_name() const { return ipc_name_; }
  inline const int shared_fd() const { return fd_; }

  ~MemoryMapWriterAllocation() override;

 private:
  std::string ipc_name_;
  int fd_ = -1;
};

class MemoryMapReaderAllocation : public Allocation {
 public:
  explicit MemoryMapReaderAllocation(void *ptr,
                                     size_t size,
                                     std::string ipc_name)
      : Allocation(ptr, size, phi::CPUPlace()),
        ipc_name_(std::move(ipc_name)) {}

  inline const std::string &ipc_name() const { return ipc_name_; }
  inline const int shared_fd() const { return fd_; }

  ~MemoryMapReaderAllocation() override;

 private:
  std::string ipc_name_;
  int fd_ = -1;
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

class MemoryMapInfo {
 public:
  explicit MemoryMapInfo(int flags,
                         size_t data_size,
                         std::string file_name,
                         void *mmap_ptr)
      : flags_(flags),
        data_size_(data_size),
        file_name_(file_name),
        mmap_ptr_(mmap_ptr) {}

  int flags_ = 0;
  size_t data_size_ = 0;
  std::string file_name_;
  void *mmap_ptr_ = nullptr;
};

/* Note(zhangbo):
MemoryMapAllocationPool is used to cache and reuse shm, thus reducing munmap in
dataloader. The munmap(shm_mmap_ptr) instruction in
RefcountedMemoryMapAllocation::close() function may block other threads of the
process. Therefore, the logic of shm cache and reuse is designed: the shm
created by the _share_filename process will be cached and reused according to
the data_size of shm, thus eliminating the problem of munmap blocking other
threads
*/
class MemoryMapAllocationPool {
 public:
  static MemoryMapAllocationPool &Instance() {
    if (pool_ == nullptr) {
      pool_ = new MemoryMapAllocationPool();
    }
    return *pool_;
  }

  void Insert(const MemoryMapInfo &memory_map);

  int FindFromCache(const int &flag,
                    const size_t &data_size,
                    const std::string &file_name = "",
                    bool check_refcount = true);

  const MemoryMapInfo &GetById(int id);

  size_t BufferSize() { return memory_map_allocations_.size(); }

  void Clear();

  void SetMaxPoolSize(const int &size);

  int MaxPoolSize() { return max_pool_size_; }

  ~MemoryMapAllocationPool();

 private:
  MemoryMapAllocationPool() = default;
  static MemoryMapAllocationPool *pool_;
  std::vector<MemoryMapInfo> memory_map_allocations_;
  int max_pool_size_ = 0;
  std::mutex mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
