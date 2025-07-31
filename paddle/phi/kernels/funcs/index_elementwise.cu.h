/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/index_elementwise_utils.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace funcs {

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template <int nt, int vt, typename func_t>
__global__ void index_elementwise_with_tensor_kernel(const int64_t N,
                                                     const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename T, typename func_t>
__global__ void index_elementwise_kernel(const int64_t N,
                                         T value_T,
                                         const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx, value_T);
      idx += nt;
    }
  }
}

template <typename T>
struct DivMod {
  T div, mod;

  __host__ __device__ DivMod(T div, T mod) : div(div), mod(mod) {}
};

template <typename T>
struct IntDivider {
  IntDivider() = default;
  explicit IntDivider(T d) : divisor(d) {}

  __host__ __device__ inline T div(T n) const { return n / divisor; }
  __host__ __device__ inline T mod(T n) const { return n % divisor; }
  __host__ __device__ inline DivMod<T> divmod(T n) const {
    return DivMod<T>(n / divisor, n % divisor);
  }
  T divisor;
};

template <>
struct IntDivider<unsigned int> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() = default;

  explicit IntDivider(unsigned int d) : divisor(d) {
    for (shift = 0; shift < 32; shift++)
      if ((1U << shift) >= divisor) break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
  }

  __host__ __device__ inline unsigned int div(unsigned int n) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
#else
    uint64_t t = ((uint64_t)n * m1) >> 32;
    return (t + n) >> shift;
#endif
  }

  __host__ __device__ inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  __host__ __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;
  unsigned int m1;
  unsigned int shift;
};

template <int NARGS, typename INDEX_T = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  using stride_t =
      std::conditional_t<signed_strides, std::make_signed_t<INDEX_T>, INDEX_T>;
  using offset_type = std::array<stride_t, std::max<int>(NARGS, 1)>;

  OffsetCalculator(int dims,
                   const int64_t* shape,
                   const int64_t* const* strides,
                   const int64_t* element_sizes = nullptr)
      : dims(dims) {
    PADDLE_ENFORCE_LE(
        dims,
        MAX_DIMS,
        common::errors::InvalidArgument(
            "Tensor has too many dims. Maximum dim is d%.", MAX_DIMS));
    for (int i = 0; i < dims; i++) {
      shape_[i] = IntDivider<INDEX_T>(shape[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  __host__ __device__ offset_type get(INDEX_T linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = shape_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<INDEX_T> shape_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator_put(
    std::vector<int64_t> desired_shape, std::array<int64_t*, N> strides_array) {
  return OffsetCalculator<N, uint32_t, signed_strides>(
      desired_shape.size(), desired_shape.data(), strides_array.data());
}

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(
    int ndim,
    const int64_t* shape,
    const std::vector<std::vector<int64_t>>& strides) {
  std::array<const int64_t*, N> strides_array;
  for (int i = 0; i < N; ++i) {
    strides_array[i] = strides[i].data();
  }

  return OffsetCalculator<N, uint32_t, signed_strides>(
      ndim, shape, strides_array.data());
}

}  // namespace funcs
}  // namespace phi
