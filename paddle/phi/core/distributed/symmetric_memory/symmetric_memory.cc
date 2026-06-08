// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/symmetric_memory/symmetric_memory.h"

#include <cuda_runtime.h>

#include <cstring>
#include <mutex>
#include <stdexcept>

#include "paddle/phi/core/distributed/store/store.h"

#define CUDA_CHECK(cmd)                                              \
  do {                                                               \
    cudaError_t e = cmd;                                             \
    if (e != cudaSuccess) {                                          \
      throw std::runtime_error(                                      \
          std::string("CUDA error: ") + cudaGetErrorString(e) +      \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
    }                                                                \
  } while (0)

namespace phi {
namespace distributed {

// Default signal pad size: 64KB
size_t SymmetricMemoryAllocator::signal_pad_size_ = 65536;

// CUDA kernels for synchronization (defined in .cu file)
extern void launch_barrier_kernel(
    void** signal_pad_ptrs_dev, int rank, int world_size,
    int channel, size_t timeout_ms, cudaStream_t stream);

extern void launch_put_signal_kernel(
    void** signal_pad_ptrs_dev, int rank, int dst_rank,
    int channel, size_t timeout_ms, cudaStream_t stream);

extern void launch_wait_signal_kernel(
    void** signal_pad_ptrs_dev, int rank, int src_rank,
    int channel, size_t timeout_ms, cudaStream_t stream);

// Kernel for stream_write_value32 (defined in .cu file)
extern void launch_stream_write_value32(
    uint32_t* ptr, int64_t offset, uint32_t val, cudaStream_t stream);

// Kernel for memset32 (defined in .cu file)
extern void launch_memset32(
    uint32_t* ptr, int64_t offset, uint32_t val, int64_t count,
    cudaStream_t stream);

// SymmetricMemory implementation
SymmetricMemory::SymmetricMemory(int rank,
                                 int world_size,
                                 std::vector<void*> buffer_ptrs,
                                 std::vector<void*> signal_pad_ptrs,
                                 void** buffer_ptrs_dev,
                                 void** signal_pad_ptrs_dev,
                                 size_t buffer_size,
                                 size_t signal_pad_size,
                                 int device_id)
    : rank_(rank),
      world_size_(world_size),
      buffer_ptrs_(std::move(buffer_ptrs)),
      signal_pad_ptrs_(std::move(signal_pad_ptrs)),
      buffer_ptrs_dev_(buffer_ptrs_dev),
      signal_pad_ptrs_dev_(signal_pad_ptrs_dev),
      buffer_size_(buffer_size),
      signal_pad_size_(signal_pad_size),
      device_id_(device_id) {}

SymmetricMemory::~SymmetricMemory() = default;

DenseTensor SymmetricMemory::get_buffer(int rank,
                                        const std::vector<int64_t>& sizes,
                                        DataType dtype,
                                        int64_t storage_offset) {
  if (rank < 0 || rank >= world_size_) {
    throw std::runtime_error("Invalid rank for get_buffer");
  }

  void* ptr = buffer_ptrs_[rank];
  size_t dtype_size = phi::SizeOf(dtype);
  size_t offset_bytes = storage_offset * dtype_size;

  auto alloc = std::make_shared<phi::Allocation>(
      static_cast<uint8_t*>(ptr) + offset_bytes,
      buffer_size_ - offset_bytes,
      phi::GPUPlace(device_id_));

  DenseTensorMeta meta(dtype, common::make_ddim(sizes));
  DenseTensor tensor(alloc, meta);
  return tensor;
}

DenseTensor SymmetricMemory::get_signal_pad(int rank,
                                            const std::vector<int64_t>& sizes,
                                            DataType dtype,
                                            int64_t storage_offset) {
  if (rank < 0 || rank >= world_size_) {
    throw std::runtime_error("Invalid rank for get_signal_pad");
  }

  void* ptr = signal_pad_ptrs_[rank];
  size_t dtype_size = phi::SizeOf(dtype);
  size_t offset_bytes = storage_offset * dtype_size;

  std::vector<int64_t> actual_sizes = sizes;
  if (actual_sizes.empty()) {
    int64_t numel = static_cast<int64_t>(signal_pad_size_ / dtype_size);
    actual_sizes = {numel};
  }

  auto alloc = std::make_shared<phi::Allocation>(
      static_cast<uint8_t*>(ptr) + offset_bytes,
      signal_pad_size_ - offset_bytes,
      phi::GPUPlace(device_id_));

  DenseTensorMeta meta(dtype, common::make_ddim(actual_sizes));
  DenseTensor tensor(alloc, meta);
  return tensor;
}

void SymmetricMemory::barrier(int channel, size_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device_id_));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_barrier_kernel(
      signal_pad_ptrs_dev_, rank_, world_size_, channel, timeout_ms, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void SymmetricMemory::put_signal(int dst_rank, int channel,
                                 size_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device_id_));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_put_signal_kernel(
      signal_pad_ptrs_dev_, rank_, dst_rank, channel, timeout_ms, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void SymmetricMemory::wait_signal(int src_rank, int channel,
                                  size_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device_id_));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_wait_signal_kernel(
      signal_pad_ptrs_dev_, rank_, src_rank, channel, timeout_ms, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// SymmetricMemoryAllocator implementation
SymmetricMemoryAllocator& SymmetricMemoryAllocator::Instance() {
  static SymmetricMemoryAllocator instance;
  return instance;
}

size_t SymmetricMemoryAllocator::get_signal_pad_size() {
  return signal_pad_size_;
}

void SymmetricMemoryAllocator::set_signal_pad_size(size_t size) {
  signal_pad_size_ = size;
}

DenseTensor SymmetricMemoryAllocator::alloc(size_t size,
                                            int device_id,
                                            const std::string& group_name) {
  CUDA_CHECK(cudaSetDevice(device_id));

  // Allocate buffer + signal pad
  size_t total_size = size + signal_pad_size_;
  void* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, total_size));
  CUDA_CHECK(cudaMemset(ptr, 0, total_size));

  AllocInfo info;
  info.ptr = ptr;
  info.size = size;
  info.device_id = device_id;
  info.group_name = group_name;
  info.ipc_handle = nullptr;

  alloc_map_[ptr] = info;

  auto alloc = std::make_shared<phi::Allocation>(
      ptr, size, phi::GPUPlace(device_id));

  DenseTensorMeta meta(DataType::UINT8, {static_cast<int64_t>(size)});
  DenseTensor tensor(alloc, meta);
  return tensor;
}

DenseTensor SymmetricMemoryAllocator::alloc_persistent(
    size_t size, int device_id, const std::string& group_name,
    int64_t alloc_id) {
  auto it = persistent_allocs_.find(alloc_id);
  if (it != persistent_allocs_.end()) {
    // Check if still active
    if (alloc_map_.count(it->second)) {
      throw std::runtime_error(
          "Persistent allocation with alloc_id=" + std::to_string(alloc_id) +
          " already exists and is active");
    }
    // Reuse the pointer
    void* ptr = it->second;
    auto alloc = std::make_shared<phi::Allocation>(
        ptr, size, phi::GPUPlace(device_id));
    DenseTensorMeta meta(DataType::UINT8, {static_cast<int64_t>(size)});
    DenseTensor tensor(alloc, meta);
    return tensor;
  }

  DenseTensor tensor = alloc(size, device_id, group_name);
  persistent_allocs_[alloc_id] = const_cast<void*>(tensor.data());
  return tensor;
}

std::shared_ptr<SymmetricMemory> SymmetricMemoryAllocator::rendezvous(
    const DenseTensor& tensor) {
  void* ptr = const_cast<void*>(tensor.data());

  // Check if already rendezvous'd
  auto it = rendezvous_map_.find(ptr);
  if (it != rendezvous_map_.end()) {
    return it->second;
  }

  // Find allocation info
  auto alloc_it = alloc_map_.find(ptr);
  if (alloc_it == alloc_map_.end()) {
    return nullptr;
  }

  const AllocInfo& info = alloc_it->second;
  const std::string& group_name = info.group_name;

  auto group_it = group_info_map_.find(group_name);
  if (group_it == group_info_map_.end()) {
    throw std::runtime_error(
        "Group info not found for group: " + group_name);
  }

  const GroupInfo& group = group_it->second;
  int rank = group.rank;
  int world_size = group.world_size;
  auto* store = group.store.get();

  int device_id = info.device_id;
  CUDA_CHECK(cudaSetDevice(device_id));

  // Get IPC handle for our buffer
  cudaIpcMemHandle_t ipc_handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handle, ptr));

  // Exchange IPC handles via store
  std::string key_prefix = "symm_mem_" + group_name + "_" +
                           std::to_string(info.size);

  // Put our handle
  std::vector<uint8_t> handle_data(sizeof(cudaIpcMemHandle_t));
  std::memcpy(handle_data.data(), &ipc_handle, sizeof(cudaIpcMemHandle_t));
  store->set(key_prefix + "_rank_" + std::to_string(rank), handle_data);

  // Collect all handles
  std::vector<void*> buffer_ptrs(world_size, nullptr);
  std::vector<void*> signal_pad_ptrs(world_size, nullptr);
  buffer_ptrs[rank] = ptr;
  signal_pad_ptrs[rank] = static_cast<uint8_t*>(ptr) + info.size;

  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;

    std::string peer_key = key_prefix + "_rank_" + std::to_string(i);
    std::vector<uint8_t> peer_data = store->get(peer_key);

    cudaIpcMemHandle_t peer_handle;
    std::memcpy(&peer_handle, peer_data.data(), sizeof(cudaIpcMemHandle_t));

    void* peer_ptr = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(
        &peer_ptr, peer_handle, cudaIpcMemLazyEnablePeerAccess));
    buffer_ptrs[i] = peer_ptr;
    signal_pad_ptrs[i] = static_cast<uint8_t*>(peer_ptr) + info.size;
  }

  // Allocate device arrays for pointers
  void** buffer_ptrs_dev = nullptr;
  void** signal_pad_ptrs_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&buffer_ptrs_dev, world_size * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&signal_pad_ptrs_dev, world_size * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(buffer_ptrs_dev, buffer_ptrs.data(),
                        world_size * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(signal_pad_ptrs_dev, signal_pad_ptrs.data(),
                        world_size * sizeof(void*), cudaMemcpyHostToDevice));

  auto symm_mem = std::make_shared<SymmetricMemory>(
      rank, world_size, buffer_ptrs, signal_pad_ptrs,
      buffer_ptrs_dev, signal_pad_ptrs_dev,
      info.size, signal_pad_size_, device_id);

  rendezvous_map_[ptr] = symm_mem;
  return symm_mem;
}

bool SymmetricMemoryAllocator::is_symm_mem_tensor(
    const DenseTensor& tensor) const {
  const void* ptr = tensor.data();
  return alloc_map_.count(const_cast<void*>(ptr)) > 0;
}

void SymmetricMemoryAllocator::set_group_info(
    const std::string& group_name, int rank, int world_size,
    std::shared_ptr<phi::distributed::Store> store) {
  group_info_map_[group_name] = {rank, world_size, store};
}

void SymmetricMemoryAllocator::stream_write_value32(
    const DenseTensor& tensor, int64_t offset, int64_t val) {
  if (offset < 0) {
    throw std::runtime_error("offset must be greater than or equal to 0");
  }
  if (val < 0 || val > 4294967295LL) {
    throw std::runtime_error(
        "val must be in the range of [0, 4294967295] (uint32_t)");
  }

  uint32_t* ptr = static_cast<uint32_t*>(const_cast<void*>(tensor.data()));
  uint32_t value = static_cast<uint32_t>(val);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_stream_write_value32(ptr, offset, value, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void SymmetricMemoryAllocator::memset32(const DenseTensor& tensor,
                                        int64_t offset,
                                        int64_t val,
                                        int64_t count) {
  // Validate input
  auto dims = tensor.dims();
  if (dims.size() != 1 || tensor.dtype() != DataType::UINT32) {
    throw std::runtime_error(
        "input must be a flat, contiguous uint32 tensor");
  }

  if (offset < 0) {
    throw std::runtime_error("offset must be greater than or equal to 0");
  }
  if (val < 0 || val > 4294967295LL) {
    throw std::runtime_error(
        "val must be in the range of [0, 4294967295] (uint32_t)");
  }
  if (count <= 0) {
    throw std::runtime_error("count must be a positive integer");
  }

  int64_t numel = tensor.numel();
  if (offset + count > numel) {
    throw std::runtime_error(
        "offset + count (" + std::to_string(offset + count) +
        ") exceeded the numel of the input (" + std::to_string(numel) + ")");
  }

  uint32_t* ptr = static_cast<uint32_t*>(const_cast<void*>(tensor.data()));
  uint32_t value = static_cast<uint32_t>(val);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_memset32(ptr, offset, value, count, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace distributed
}  // namespace phi
