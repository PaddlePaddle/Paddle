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

#include "paddle/phi/kernels/funcs/index_elementwise_utils.h"

namespace phi {
namespace funcs {

template <typename Value>
struct CPUDivMod {
  Value div, mod;

  CPUDivMod(Value div, Value mod) : div(div), mod(mod) {}
};

template <typename Value>
struct CPUIntDivider {
  CPUIntDivider() = default;
  explicit CPUIntDivider(Value d) : divisor(d) {}

  inline CPUDivMod<Value> cpu_divmod(Value n) const {
    return CPUDivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct CPUOffsetCalculator {
  using stride_t =
      std::conditional_t<signed_strides, std::make_signed_t<index_t>, index_t>;
  using offset_type = std::array<stride_t, std::max<int>(NARGS, 1)>;

  CPUOffsetCalculator(int dims,
                      const int64_t* sizes,
                      const int64_t* const* strides,
                      const int64_t* element_sizes = nullptr)
      : dims(dims) {
    PADDLE_ENFORCE_LE(
        dims,
        MAX_DIMS,
        common::errors::InvalidArgument(
            "Tensor has too many dims. Maximum dim is d%.", MAX_DIMS));
    for (int i = 0; i < dims; i++) {
      sizes_[i] = CPUIntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  offset_type cpu_get(index_t linear_idx) const {
    offset_type offsets;
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].cpu_divmod(linear_idx);
      linear_idx = divmod.div;

      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  CPUIntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <int N, bool signed_strides = false>
static CPUOffsetCalculator<N, uint32_t, signed_strides>
CPUmake_offset_calculator_put(std::vector<int64_t> desired_shape,
                              std::array<int64_t*, N> strides_array) {
  return CPUOffsetCalculator<N, uint32_t, signed_strides>(
      desired_shape.size(), desired_shape.data(), strides_array.data());
}

template <int N, bool signed_strides = false>
static CPUOffsetCalculator<N, uint32_t, signed_strides>
CPUmake_offset_calculator(int ndim,
                          const int64_t* shape,
                          const std::vector<std::vector<int64_t>>& strides) {
  std::array<const int64_t*, N> strides_array;
  for (int i = 0; i < N; ++i) {
    strides_array[i] = strides[i].data();
  }

  return CPUOffsetCalculator<N, uint32_t, signed_strides>(
      ndim, shape, strides_array.data());
}

}  // namespace funcs
}  // namespace phi
