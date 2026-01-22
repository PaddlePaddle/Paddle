// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __HIPCC__

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include <rocprim/config.hpp>

// ROCm 7.0+ uses a new traits system based on rocprim::traits::define
// This header provides trait definitions for phi::float16 and phi::bfloat16
// to enable radix sort and other rocprim algorithms on these types.

#if defined(ROCPRIM_VERSION) && ROCPRIM_VERSION >= 400000
// ROCm 7.0+ (rocprim 4.0.0+)
namespace rocprim {
namespace traits {

template <>
struct define<phi::float16> {
  // float16: sign=0x8000, exponent=0x7C00, mantissa=0x03FF
  using float_bit_mask =
      float_bit_mask::values<uint16_t, 0x8000, 0x7C00, 0x03FF>;
};

template <>
struct define<phi::bfloat16> {
  // bfloat16: sign=0x8000, exponent=0x7F80, mantissa=0x007F
  using float_bit_mask =
      float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};

}  // namespace traits
}  // namespace rocprim

#else
// ROCm < 7.0 uses the old traits system
namespace rocprim {
namespace detail {

template <>
struct radix_key_codec_base<phi::float16>
    : radix_key_codec_integral<phi::float16, uint16_t> {};

template <>
struct radix_key_codec_base<phi::bfloat16>
    : radix_key_codec_integral<phi::bfloat16, uint16_t> {};

#if HIP_VERSION >= 50400000
template <>
struct float_bit_mask<phi::float16> : float_bit_mask<rocprim::half> {};

template <>
struct float_bit_mask<phi::bfloat16> : float_bit_mask<rocprim::bfloat16> {};
#endif

}  // namespace detail
}  // namespace rocprim

#endif  // ROCPRIM_VERSION

#endif  // __HIPCC__

