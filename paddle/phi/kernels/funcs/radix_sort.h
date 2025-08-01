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
#ifdef PADDLE_WITH_CUDA
#include <cub/cub.cuh>
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

#ifdef PADDLE_WITH_CUDA
template <int kValueSize>
struct OpaqueTypeRadix {
  uint8_t data[kValueSize];
  __device__ __host__ OpaqueTypeRadix() = default;
};

template <typename key_t, int kValueSize>
void RadixSortPairsImpl(const phi::GPUContext& dev_ctx,
                        const key_t* keys_in,
                        key_t* keys_out,
                        const OpaqueTypeRadix<kValueSize>* values_in,
                        OpaqueTypeRadix<kValueSize>* values_out,
                        int64_t n,
                        bool descending = false,
                        int64_t begin_bit = 0,
                        int64_t end_bit = sizeof(key_t) * 8);

template <typename key_t, typename value_t>
void RadixSortPairs(const phi::GPUContext& dev_ctx,
                    const key_t* keys_in,
                    key_t* keys_out,
                    const value_t* values_in,
                    value_t* values_out,
                    int64_t n,
                    bool descending = false,
                    int64_t begin_bit = 0,
                    int64_t end_bit = sizeof(key_t) * 8) {
  PADDLE_ENFORCE_EQ(
      std::is_trivially_copyable<value_t>::value,
      true,
      phi::errors::InvalidArgument(
          "RadixSortPairs value type must be trivially copyable"));

  using opaque_t = OpaqueTypeRadix<sizeof(value_t)>;
  PADDLE_ENFORCE_EQ(
      sizeof(value_t) <= 8 && (sizeof(value_t) & (sizeof(value_t) - 1)) == 0,
      true,
      phi::errors::InvalidArgument(
          "Unsupported value_t size (must be 1, 2, 4, or 8 bytes)"));
  PADDLE_ENFORCE_EQ(
      sizeof(value_t),
      alignof(value_t),
      phi::errors::InvalidArgument("Expected value_t to be size-aligned"));

  RadixSortPairsImpl<key_t, sizeof(value_t)>(
      dev_ctx,
      keys_in,
      keys_out,
      reinterpret_cast<const opaque_t*>(values_in),
      reinterpret_cast<opaque_t*>(values_out),
      n,
      descending,
      begin_bit,
      end_bit);
}

#endif
}  // namespace funcs
}  // namespace phi
