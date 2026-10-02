// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/store/store.h"

namespace phi {
namespace distributed {

// SymmetricMemory represents symmetric allocations across a group of devices.
// The allocations are accessible by all devices in the group via P2P.
// This class provides:
// - Buffer access APIs (get_buffer, get_signal_pad)
// - Synchronization primitives (barrier, put_signal, wait_signal)
// - Low-level memory operations (stream_write_value32, memset32)
class SymmetricMemory {
 public:
  SymmetricMemory(int rank,
                  int world_size,
                  std::vector<void*> buffer_ptrs,
                  std::vector<void*> signal_pad_ptrs,
                  void** buffer_ptrs_dev,
                  void** signal_pad_ptrs_dev,
                  size_t buffer_size,
                  size_t signal_pad_size,
                  int device_id);

  ~SymmetricMemory();

  // Properties
  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  size_t buffer_size() const { return buffer_size_; }
  size_t signal_pad_size() const { return signal_pad_size_; }
  int device_id() const { return device_id_; }

  // Pointer access
  const std::vector<void*>& buffer_ptrs() const { return buffer_ptrs_; }
  const std::vector<void*>& signal_pad_ptrs() const { return signal_pad_ptrs_; }
  void** buffer_ptrs_dev() const { return buffer_ptrs_dev_; }
  void** signal_pad_ptrs_dev() const { return signal_pad_ptrs_dev_; }

  // Get a tensor view of a peer's buffer
  DenseTensor get_buffer(int rank,
                         const std::vector<int64_t>& sizes,
                         DataType dtype,
                         int64_t storage_offset = 0);

  // Get a tensor view of a peer's signal pad
  DenseTensor get_signal_pad(int rank,
                             const std::vector<int64_t>& sizes = {},
                             DataType dtype = DataType::UINT32,
                             int64_t storage_offset = 0);

  // Synchronization primitives
  void barrier(int channel = 0, size_t timeout_ms = 0);
  void put_signal(int dst_rank, int channel = 0, size_t timeout_ms = 0);
  void wait_signal(int src_rank, int channel = 0, size_t timeout_ms = 0);

 private:
  int rank_;
  int world_size_;
  std::vector<void*> buffer_ptrs_;
  std::vector<void*> signal_pad_ptrs_;
  void** buffer_ptrs_dev_;
  void** signal_pad_ptrs_dev_;
  size_t buffer_size_;
  size_t signal_pad_size_;
  int device_id_;
};

// SymmetricMemoryAllocator manages symmetric memory allocations and
// rendezvous across processes.
class SymmetricMemoryAllocator {
 public:
  static SymmetricMemoryAllocator& Instance();

  // Allocate symmetric memory (P2P accessible)
  DenseTensor alloc(size_t size,
                    int device_id,
                    const std::string& group_name);

  // Allocate with persistent ID (same alloc_id returns same memory)
  DenseTensor alloc_persistent(size_t size,
                               int device_id,
                               const std::string& group_name,
                               int64_t alloc_id);

  // Perform rendezvous to establish cross-rank buffer association
  std::shared_ptr<SymmetricMemory> rendezvous(const DenseTensor& tensor);

  // Check if a tensor is backed by symmetric memory
  bool is_symm_mem_tensor(const DenseTensor& tensor) const;

  // Set group info for symmetric memory
  void set_group_info(const std::string& group_name,
                      int rank,
                      int world_size,
                      std::shared_ptr<phi::distributed::Store> store);

  // Signal pad size management
  static size_t get_signal_pad_size();
  static void set_signal_pad_size(size_t size);

  // Low-level memory operations
  static void stream_write_value32(const DenseTensor& tensor,
                                   int64_t offset,
                                   int64_t val);
  static void memset32(const DenseTensor& tensor,
                       int64_t offset,
                       int64_t val,
                       int64_t count);

 private:
  SymmetricMemoryAllocator() = default;

  struct AllocInfo {
    void* ptr;
    size_t size;
    int device_id;
    std::string group_name;
    void* ipc_handle;
  };

  struct GroupInfo {
    int rank;
    int world_size;
    std::shared_ptr<phi::distributed::Store> store;
  };

  std::unordered_map<void*, AllocInfo> alloc_map_;
  std::unordered_map<void*, std::shared_ptr<SymmetricMemory>> rendezvous_map_;
  std::unordered_map<std::string, GroupInfo> group_info_map_;
  std::unordered_map<int64_t, void*> persistent_allocs_;

  static size_t signal_pad_size_;
};

}  // namespace distributed
}  // namespace phi
