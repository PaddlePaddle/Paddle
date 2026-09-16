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

#include <cuda_runtime.h>
#include <cstdint>

namespace phi {
namespace distributed {

// ========== Synchronization Kernels ==========

// Barrier kernel: all ranks signal each other and wait
__global__ void barrier_kernel(void** signal_pad_ptrs,
                               int rank,
                               int world_size,
                               int channel,
                               size_t timeout_ms) {
  // Each rank has signal_pad_size bytes. We use uint32 slots.
  // Slot layout: [world_size entries per channel]
  // barrier uses channel*world_size + peer_rank as the slot index
  int base_offset = channel * world_size;

  // Step 1: Signal all peers (write 1 to our slot in their pad)
  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;
    volatile uint32_t* peer_pad =
        static_cast<volatile uint32_t*>(signal_pad_ptrs[i]);
    atomicAdd(const_cast<uint32_t*>(&peer_pad[base_offset + rank]), 1u);
  }

  // Step 2: Wait for all peers to signal us
  volatile uint32_t* my_pad =
      static_cast<volatile uint32_t*>(signal_pad_ptrs[rank]);

  unsigned long long start = clock64();
  // Approximate timeout: assume 1.5 GHz clock
  unsigned long long timeout_cycles = timeout_ms > 0
      ? (unsigned long long)timeout_ms * 1500000ULL
      : 0;

  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;
    while (my_pad[base_offset + i] == 0) {
      if (timeout_ms > 0 && (clock64() - start) > timeout_cycles) {
        __trap();
        return;
      }
    }
  }

  // Step 3: Reset our signals
  __threadfence_system();
  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;
    const_cast<uint32_t*>(
        const_cast<volatile uint32_t*>(&my_pad[base_offset + i]))[0] = 0;
  }
  __threadfence_system();
}

// Put signal: write a signal to dst_rank's signal pad
__global__ void put_signal_kernel(void** signal_pad_ptrs,
                                  int rank,
                                  int dst_rank,
                                  int channel,
                                  size_t timeout_ms) {
  int base_offset = channel * 8 + rank;  // max 8 ranks per channel group

  volatile uint32_t* dst_pad =
      static_cast<volatile uint32_t*>(signal_pad_ptrs[dst_rank]);

  // Wait until previous signal was consumed (slot == 0)
  unsigned long long start = clock64();
  unsigned long long timeout_cycles = timeout_ms > 0
      ? (unsigned long long)timeout_ms * 1500000ULL
      : 0;

  while (dst_pad[base_offset] != 0) {
    if (timeout_ms > 0 && (clock64() - start) > timeout_cycles) {
      __trap();
      return;
    }
  }

  // Write the signal
  __threadfence_system();
  atomicExch(const_cast<uint32_t*>(&dst_pad[base_offset]), 1u);
  __threadfence_system();
}

// Wait signal: wait for a signal from src_rank
__global__ void wait_signal_kernel(void** signal_pad_ptrs,
                                   int rank,
                                   int src_rank,
                                   int channel,
                                   size_t timeout_ms) {
  int base_offset = channel * 8 + src_rank;  // max 8 ranks per channel group

  volatile uint32_t* my_pad =
      static_cast<volatile uint32_t*>(signal_pad_ptrs[rank]);

  // Wait until signal arrives
  unsigned long long start = clock64();
  unsigned long long timeout_cycles = timeout_ms > 0
      ? (unsigned long long)timeout_ms * 1500000ULL
      : 0;

  while (my_pad[base_offset] == 0) {
    if (timeout_ms > 0 && (clock64() - start) > timeout_cycles) {
      __trap();
      return;
    }
  }

  // Consume the signal (reset to 0)
  __threadfence_system();
  const_cast<uint32_t*>(
      const_cast<volatile uint32_t*>(&my_pad[base_offset]))[0] = 0;
  __threadfence_system();
}

// ========== Memory Operation Kernels ==========

// Write a single uint32 value at offset
__global__ void stream_write_value32_kernel(uint32_t* ptr,
                                            int64_t offset,
                                            uint32_t val) {
  ptr[offset] = val;
  __threadfence_system();
}

// Set count uint32 values starting at offset
__global__ void memset32_kernel(uint32_t* ptr,
                                int64_t offset,
                                uint32_t val,
                                int64_t count) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    ptr[offset + idx] = val;
  }
}

// ========== Host Launch Functions ==========

void launch_barrier_kernel(void** signal_pad_ptrs_dev,
                           int rank,
                           int world_size,
                           int channel,
                           size_t timeout_ms,
                           cudaStream_t stream) {
  barrier_kernel<<<1, 1, 0, stream>>>(
      signal_pad_ptrs_dev, rank, world_size, channel, timeout_ms);
}

void launch_put_signal_kernel(void** signal_pad_ptrs_dev,
                              int rank,
                              int dst_rank,
                              int channel,
                              size_t timeout_ms,
                              cudaStream_t stream) {
  put_signal_kernel<<<1, 1, 0, stream>>>(
      signal_pad_ptrs_dev, rank, dst_rank, channel, timeout_ms);
}

void launch_wait_signal_kernel(void** signal_pad_ptrs_dev,
                               int rank,
                               int src_rank,
                               int channel,
                               size_t timeout_ms,
                               cudaStream_t stream) {
  wait_signal_kernel<<<1, 1, 0, stream>>>(
      signal_pad_ptrs_dev, rank, src_rank, channel, timeout_ms);
}

void launch_stream_write_value32(uint32_t* ptr,
                                 int64_t offset,
                                 uint32_t val,
                                 cudaStream_t stream) {
  stream_write_value32_kernel<<<1, 1, 0, stream>>>(ptr, offset, val);
}

void launch_memset32(uint32_t* ptr,
                     int64_t offset,
                     uint32_t val,
                     int64_t count,
                     cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  memset32_kernel<<<blocks, threads, 0, stream>>>(ptr, offset, val, count);
}

}  // namespace distributed
}  // namespace phi
