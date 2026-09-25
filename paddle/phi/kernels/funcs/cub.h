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

// Thrust iterators that replace cub::TransformInputIterator /
// cub::CountingInputIterator (both removed in CCCL 3.0). These types are
// provided by thrust on CUDA and by rocThrust on HIP, so include them on both
// compilation paths.
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#endif

#ifdef __NVCC__
#include <cub/version.cuh>
#include "cub/cub.cuh"

// phi half/bfloat16 are registered with CUB (radix sort / top-k). CCCL 3.0
// requires cuda::std::numeric_limits to be specialized (BaseTraits asserts it)
// and classifies float types via cuda::is_floating_point; 2.x needs neither but
// its BaseTraits takes an extra bool template argument. These specializations
// must be visible in every TU that instantiates CUB primitives for phi half
// types, so keep them here instead of repeating them per kernel.
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

#if defined(CUB_VERSION) && CUB_VERSION >= 300000
#include <cuda/std/limits>   // cuda::std::numeric_limits
#include <cuda/type_traits>  // cuda::is_floating_point_v
namespace cuda {
template <>
inline constexpr bool is_floating_point_v<phi::dtype::float16> = true;
template <>
inline constexpr bool is_floating_point_v<phi::dtype::bfloat16> = true;
}  // namespace cuda
template <>
class cuda::std::numeric_limits<phi::dtype::float16> {
 public:
  static constexpr bool is_specialized = true;
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::float16 max() {
    return phi::dtype::raw_uint16_to_float16(0x7bff);
  }
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::float16 min() {
    return phi::dtype::raw_uint16_to_float16(0x0400);
  }
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::float16 lowest() {
    return phi::dtype::raw_uint16_to_float16(0xfbff);
  }
};
template <>
class cuda::std::numeric_limits<phi::dtype::bfloat16> {
 public:
  static constexpr bool is_specialized = true;
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::bfloat16 max() {
    return phi::dtype::raw_uint16_to_bfloat16(0x7f7f);
  }
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::bfloat16 min() {
    return phi::dtype::raw_uint16_to_bfloat16(0x0080);
  }
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE phi::dtype::bfloat16 lowest() {
    return phi::dtype::raw_uint16_to_bfloat16(0xff7f);
  }
};
#endif  // CUB_VERSION >= 300000

// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::dtype::float16>
#if defined(CUB_VERSION) && CUB_VERSION >= 300000
    : BaseTraits<FLOATING_POINT, true, uint16_t, phi::dtype::float16> {
};
#else
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::float16> {
};
#endif
template <>
struct NumericTraits<phi::dtype::bfloat16>
#if defined(CUB_VERSION) && CUB_VERSION >= 300000
    : BaseTraits<FLOATING_POINT, true, uint16_t, phi::dtype::bfloat16> {
};
#else
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::bfloat16> {
};
#endif
}  // namespace cub
#endif  // __NVCC__
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
// Iterator aliases: cub::TransformInputIterator / cub::CountingInputIterator
// were removed in CCCL 3.0. Re-express them on top of thrust so call sites keep
// the cub:: spelling, which also lets the HIP path (cub == hipcub) use the
// rocPRIM-compatible hipcub iterators instead of thrust ones.
template <typename ValueType, typename ConversionOp, typename InputIteratorT>
using TransformInputIterator =
    ::thrust::transform_iterator<ConversionOp, InputIteratorT>;
template <typename ValueType>
using CountingInputIterator = ::thrust::counting_iterator<ValueType>;
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
