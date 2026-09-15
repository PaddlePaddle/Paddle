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

#pragma once

#ifdef __NVCC__
#include <cub/version.cuh>
// Thrust iterators that replace cub::TransformInputIterator /
// cub::CountingInputIterator (both removed in CCCL 3.0).
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

// CUB/CCCL 2.x vs 3.x: several reduction operators and warp helpers were
// removed in CCCL 3.0.  Aliases and wrappers below restore them in the cub::
// namespace so all call sites can remain unchanged.
// CUB_VERSION exists on both old and new CCCL; CCCL_VERSION is 3.x-only.
// Guarded by __NVCC__: cuda::ptx functions are device-only and not available
// when compiled by g++ for CPU-only .cc files that transitively include cub.h.
#if defined(__NVCC__) && defined(CUB_VERSION) && CUB_VERSION >= 300000
#include <cuda/functional>      // cuda::maximum
#include <cuda/ptx>             // cuda::ptx::get_sreg_*
#include <cuda/std/functional>  // cuda::std::plus, equal_to
namespace cub {
// Type aliases
using Sum = ::cuda::std::plus<>;
using Equality = ::cuda::std::equal_to<>;
using Max = ::cuda::maximum<>;
// Warp helper functions -- __forceinline__ implies inline, safe in multi-TU
// headers
__device__ __forceinline__ unsigned int LaneId() {
  return ::cuda::ptx::get_sreg_laneid();
}
__device__ __forceinline__ unsigned int LaneMaskLt() {
  return ::cuda::ptx::get_sreg_lanemask_lt();
}
}  // namespace cub
#endif
