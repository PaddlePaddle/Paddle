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

#include <vector>

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

template <typename DeviceContext>
struct ForRangeIn {
  ForRangeIn(const DeviceContext& dev_ctx, std::vector<int64_t> range);

  template <typename Function>
  void operator()(Function func) const;
};

template <>
struct ForRangeIn<CPUDeviceContext> {
  ForRangeIn(const CPUDeviceContext& dev_ctx, std::vector<int64_t> range)
      : range_(range) {}

  template <typename Function>
  void operator()(Function func) const {
    for (auto i : range_) {
      func(i);
    }
  }

  std::vector<int64_t> range_;
};

template <typename DeviceContext>
struct ForRange {
  ForRange(const DeviceContext& dev_ctx, size_t limit);

  template <typename Function>
  void operator()(Function func) const;
};

template <>
struct ForRange<CPUDeviceContext> {
  ForRange(const CPUDeviceContext& dev_ctx, size_t limit) : limit_(limit) {}

  template <typename Function>
  void operator()(Function func) const {
    for (size_t i = 0; i < limit_; ++i) {
      func(i);
    }
  }

  size_t limit_;
};

#ifdef __NVCC__
template <typename Function>
__global__ static void ForRangeElemwiseOpGridIsOne(Function func) {
  size_t idx = static_cast<size_t>(threadIdx.x);
  func(idx);
}

template <typename Function>
__global__ static void ForRangeElemwiseOp(Function func, int limit) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < limit) {
    func(idx);
  }
}

template <>
struct ForRange<CUDADeviceContext> {
  ForRange(const CUDADeviceContext& dev_ctx, size_t limit)
      : dev_ctx_(dev_ctx), limit_(static_cast<int>(limit)) {}

  template <typename Function>
  inline void operator()(Function func) const {
    constexpr int num_threads = 1024;
    int block_size = limit_ <= num_threads ? limit_ : num_threads;
    int grid_size = (limit_ + num_threads - 1) / num_threads;

    if (grid_size == 1) {
      ForRangeElemwiseOpGridIsOne<<<1, block_size, 0, dev_ctx_.stream()>>>(
          func);
    } else {
      ForRangeElemwiseOp<<<grid_size, block_size, 0, dev_ctx_.stream()>>>(
          func, limit_);
    }
  }

  const CUDADeviceContext& dev_ctx_;
  int limit_;
};

template <typename T, typename Function>
__global__ static void ForRangeInElemwiseOp(Function func, T* vector,
                                            int vector_size) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < vector_size) {
    func(vector[idx]);
  }
}

template <>
struct ForRangeIn<CUDADeviceContext> {
  ForRangeIn(const CUDADeviceContext& dev_ctx, std::vector<int64_t> range)
      : dev_ctx_(dev_ctx), range_(range) {}

  template <typename Function>
  inline void operator()(Function func) const {
    constexpr int num_threads = 1024;
    int range_size = range_.size();
    int block_size = range_size <= num_threads ? range_size : num_threads;
    int grid_size = (range_.size() + num_threads - 1) / num_threads;

    ForRangeInElemwiseOp<<<grid_size, block_size, 0, dev_ctx_.stream()>>>(
        func, range_.CUDAData(dev_ctx_.GetPlace()), range_size);
  }

  const CUDADeviceContext& dev_ctx_;
  framework::Vector<int64_t> range_;
};

#endif

}  // namespace platform
}  // namespace paddle
