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

// Fast hash for GPU tensors: D2H copy + CPU hash computation.
// Small tensors use plain cudaMemcpy; large tensors use pinned memory.

#include "paddle/utils/md5_gpu.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <xxhash.h>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/utils/md5.h"

namespace paddle {
namespace {

// Below this threshold, use malloc + sync cudaMemcpy (avoids cudaMallocHost
// overhead which dominates for small tensors).
static constexpr size_t kPinnedThreshold = 1 * 1024 * 1024;  // 1 MB

// Helper: D2H copy with size-adaptive strategy.
// Returns a host pointer that the caller must free via free_host_buffer.
struct HostBuffer {
  void* ptr;
  bool pinned;
};

static HostBuffer d2h_copy(const void* dev_data,
                           size_t len,
                           cudaStream_t stream) {
  HostBuffer buf;
  if (len < kPinnedThreshold) {
    buf.ptr = std::malloc(len);
    buf.pinned = false;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        buf.ptr, dev_data, len, cudaMemcpyDeviceToHost, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMallocHost(&buf.ptr, len));
    buf.pinned = true;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        buf.ptr, dev_data, len, cudaMemcpyDeviceToHost, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  }
  return buf;
}

static void free_host_buffer(HostBuffer* buf) {
  if (buf->pinned) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(buf->ptr));
  } else {
    std::free(buf->ptr);
  }
  buf->ptr = nullptr;
}

}  // anonymous namespace

std::string md5_gpu(const void* data, size_t len, void* stream) {
  if (len == 0) {
    return md5(data, 0);
  }
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  auto buf = d2h_copy(data, len, cuda_stream);
  std::string result = md5(buf.ptr, len);
  free_host_buffer(&buf);
  return result;
}

std::string xxhash64_gpu(const void* data, size_t len, void* stream) {
  auto cuda_stream = static_cast<cudaStream_t>(stream);

  uint64_t hash_val;
  if (len == 0) {
    hash_val = XXH64(nullptr, 0, 0);
  } else {
    auto buf = d2h_copy(data, len, cuda_stream);
    hash_val = XXH64(buf.ptr, len, 0);
    free_host_buffer(&buf);
  }

  // Convert uint64_t to 16-char hex string
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016" PRIx64, hash_val);
  return std::string(hex, 16);
}

}  // namespace paddle
