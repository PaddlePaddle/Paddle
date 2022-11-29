/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

namespace phi {
namespace funcs {
namespace sparse {

// brief: calculation the distance between start and end
template <typename T>
__global__ void DistanceKernel(const T* start, const T* end, T* distance) {
  if (threadIdx.x == 0) {
    *distance = end - start;
  }
}

inline __device__ bool SetBits(const int value, int* ptr) {
  const int index = value >> 5;
  const int mask = 1 << (value & 31);
  const int old = atomicOr(ptr + index, mask);
  return (mask & old) != 0;
}

inline __device__ bool TestBits(const int value, const int* ptr) {
  const int index = value >> 5;
  const int mask = 1 << (value & 31);
  return (mask & ptr[index]) != 0;
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
