// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "cutlass/platform/platform.h"
namespace cutlass {
namespace platform {

// template< class To, class From >
// constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& from ) noexcept;

// template <class To, class From>
// constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& src) noexcept
// {
//   static_assert(sizeof(To) == sizeof(From), "sizes must match");
//   return reinterpret_cast<To const &>(src);
// }

// template <>
// struct numeric_limits<float> {
//   CUTLASS_HOST_DEVICE
//   static constexpr float infinity() noexcept { return bit_cast<float,
//   int32_t>(0x7f800000);} static constexpr bool is_integer = false; static
//   constexpr bool has_infinity = true;
// };

// template <>
// struct numeric_limits<cutlass::half_t> {
//   CUTLASS_HOST_DEVICE
//   static const cutlass::half_t infinity() noexcept { return
//   bit_cast<cutlass::half_t, int16_t>(0x7800);} static constexpr bool
//   is_integer = false; static constexpr bool has_infinity = true;
// };

}  // namespace platform
}  // namespace cutlass
