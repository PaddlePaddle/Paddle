/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/common/macros.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
namespace phi {
namespace funcs {

template <typename Context>
struct ForRange {
  ForRange(const Context& dev_ctx, size_t limit);

  template <typename Function>
  void operator()(Function func) const;
};

template <>
struct ForRange<phi::CPUContext> {
  ForRange(const phi::CPUContext& dev_ctx UNUSED, size_t limit)
      : limit_(limit) {}

  template <typename Function>
  void operator()(Function func) const {
    for (size_t i = 0; i < limit_; ++i) {
      func(i);
    }
  }

  size_t limit_;
};

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename Function>
__global__ static void ForRangeElemwiseOpGridIsOne(Function func) {
  size_t idx = static_cast<size_t>(threadIdx.x);
  func(idx);
}

template <typename Function>
__global__ static void ForRangeElemwiseOp(Function func, size_t limit) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < limit) {
    func(idx);
  }
}

template <>
struct ForRange<phi::GPUContext> {
  ForRange(const phi::GPUContext& dev_ctx, size_t limit)
      : dev_ctx_(dev_ctx), limit_(limit) {}

  template <typename Function>
  inline void operator()(Function func) const {
#if WITH_NV_JETSON
    // JETSON_NANO will throw core dump when threads > 128
    int num_thread = 256;
    backends::gpu::ChangeThreadNum(dev_ctx_, &num_thread, 128);
    const int num_threads = num_thread;
#else
    constexpr int num_threads = 1024;
#endif
    size_t block_size = limit_ <= num_threads ? limit_ : num_threads;
    size_t grid_size = (limit_ + num_threads - 1) / num_threads;

    if (grid_size == 1) {
      ForRangeElemwiseOpGridIsOne<<<1, block_size, 0, dev_ctx_.stream()>>>(
          func);
    } else {
      ForRangeElemwiseOp<<<grid_size, block_size, 0, dev_ctx_.stream()>>>(
          func, limit_);
    }
  }

  const phi::GPUContext& dev_ctx_;
  size_t limit_;
};

#endif

}  // namespace funcs
}  // namespace phi
