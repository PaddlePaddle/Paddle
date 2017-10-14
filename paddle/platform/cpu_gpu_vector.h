/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <algorithm>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

namespace paddle {
namespace platform {

#ifndef PADDLE_WITH_CUDA
template <typename T>
using CPUGPUVector = std::vector<T>;
#else
template <typename T>
using CPUGPUVector = thrust::host_vector<
    T, thrust::system::cuda::experimental::pinned_allocator<T>>;
#endif

template <typename It1, typename It2>
inline void cpu_gpu_copy(It1 in_begin, It1 in_end, It2 out_begin) {
#ifndef PADDLE_WITH_CUDA
  std::copy(in_begin, in_end, out_begin);
#else
  thrust::copy(in_begin, in_end, out_begin);
#endif
}

template <typename T>
std::vector<T> CopyCPUGPUVecToVector(const CPUGPUVector<T>& in) {
  std::vector<T> out;
  out.resize(in.size());
  cpu_gpu_copy(in.begin(), in.end(), out.begin());
  return out;
}

template <typename T>
CPUGPUVector<T> CopyVectorToCPUGPUVec(const std::vector<T>& in) {
  CPUGPUVector<T> out;
  out.resize(in.size());
  cpu_gpu_copy(in.begin(), in.end(), out.begin());
  return out;
}
}
}  // namespace paddle
