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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/cuda/PhiloxCudaState.h>
#include <tuple>

namespace at::cuda::philox {

// In-kernel call to retrieve philox seed and offset from a PhiloxCudaState
// instance whether that instance was created with graph capture underway or
// not. See Note [CUDA Graph-safe RNG states].
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    at::PhiloxCudaState arg) {
#else
inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
#endif
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to
    // "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire
    // kernel. For most threads' reads it will hit in cache, so it shouldn't
    // hurt performance.
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

}  // namespace at::cuda::philox
