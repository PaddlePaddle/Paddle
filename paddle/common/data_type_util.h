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

#include <cstdint>

namespace phi::dtype {

union Bits {
  float f;
  uint32_t ui;
};

// refer to
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_fnuz_cvt.h

/*
 * Convert a 8-bit floating-point number in either f8 E4M3FNUZ or bf8
 * E5M2FNUZ format, in bit representation, to a 32-bit floating-point
 * number.
 */
template <uint32_t we, uint32_t wm>
inline HOSTDEVICE float fp8_fnuz_to_fp32_value(uint8_t x) {
  static_assert((we == 4 && wm == 3) || (we == 5 && wm == 2));
  constexpr uint32_t weo = 8;
  constexpr uint32_t wmo = 23;

  if (x == 0) {
    return 0;
  }

  if (x == 0x80) {
    Bits ifNaN;
    ifNaN.ui = 0x7F800001;
    return ifNaN.f;
  }

  uint32_t mantissa = x & ((1 << wm) - 1);
  uint32_t exponent = (x & 0x7F) >> wm;

  // subnormal input
  if (exponent == 0) {
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
#if defined(__CUDA_ARCH__)
    uint32_t renorm_shift = __clz(mantissa);
#elif defined(_MSC_VER)
    unsigned long nonsign_bsr;                               // NOLINT
    _BitScanReverse(&nonsign_bsr, (unsigned long)mantissa);  // NOLINT
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
    uint32_t renorm_shift = __builtin_clz(mantissa);
#endif
    uint32_t sh = 1 + renorm_shift - (32 - wm);
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1 << wm) - 1);
  }

  const uint32_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1));
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  uint32_t sign = x >> 7;
  Bits retval;
  retval.ui = (sign << 31) | (exponent << 23) | mantissa;
  return retval.f;
}

// refer to
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fnuz.h#L64-L133
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FNUZ format, in bit representation.
 */
inline HOSTDEVICE uint8_t fp8e4m3fnuz_from_fp32_value(float f) {
  Bits fb, denorm_mask;

  /*
   * Binary representation of 256.0f, which is the first value not representable
   * (i.e. the first value which would overflow in to the sign bit, resulting in
   * a NaN) in fp8e4m3fnuz range:
   * 1 0000 000 - fp8e4m3fnuz
   * 0 10000111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
   * into denorm representation
   * magic number: ((127 - 8) + (23 - 3) + 1)
   */
  denorm_mask.ui = UINT32_C(0x8C) << 23;
  fb.f = f;

  uint32_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = fb.ui & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  fb.ui ^= sign;

  if (fb.ui >= fnuz_max) {
    // NaN -- sign bit set to 1, rest 0s.
    return 0x80;
  }

  if (fb.ui < (UINT32_C(0x78) << 23) /* 2^-7 in float32 */) {
    // Input exponent is less than -7, the smallest e4m3fnuz exponent, so the
    // number will become subnormal.
    fb.f = fb.f + denorm_mask.f;
    result = static_cast<uint8_t>(fb.ui - denorm_mask.ui);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (fb.ui >> 20) & 1;

    // update exponent, rounding bias part 1
    fb.ui += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

    // rounding bias part 2
    fb.ui += mant_odd;

    // take the bits!
    result = static_cast<uint8_t>(fb.ui >> 20);
  }

  result |= sign >> 24;
  return result;
}
// refer to
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e8m0fnu.h#L57-L112
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 e8m0fnu format, in bit representation.
 */
inline HOSTDEVICE uint8_t fp8e8m0fnu_from_fp32_value(float f) {
  Bits fb, denorm_mask;

  fb.f = f;

  // extract the exponent
  uint32_t exponent = (fb.ui >> 23) & 0b11111111;

  // special case float32 NaN and +-inf to map to e8m0 nan
  if (exponent == 0b11111111) {
    return exponent;
  }

  // next, we use guard, round, sticky bits and the LSB to implement round to
  // nearest, with ties to even

  // guard bit - bit 23, or 22 zero-indexed
  uint8_t g = (fb.ui & 0x400000) > 0;
  // round bit - bit 22, or 21 zero-indexed
  uint8_t r = (fb.ui & 0x200000) > 0;
  // sticky bit - bits 21 to 1, or 20 to 0 zero-indexed
  uint8_t s = (fb.ui & 0x1FFFFF) > 0;
  // in casting to e8m0, LSB is the implied mantissa bit. It equals to 0 if the
  // original float32 is denormal, and to 1 if the original float32 is normal.
  uint8_t lsb = exponent > 0;

  // implement the RNE logic
  bool round_up = false;

  // if g == 0, round down (no-op)
  if (g == 1) {
    if ((r == 1) || (s == 1)) {
      // round up
      round_up = true;
    } else {
      if (lsb == 1) {
        // round up
        round_up = true;
      }
      // if lsb == 0, round down (no-op)
    }
  }

  if (round_up) {
    // adjust exponent
    // note that if exponent was 255 we would have already returned earlier, so
    // we know we can add one safely without running out of bounds
    exponent++;
  }

  return exponent;
}

}  // namespace phi::dtype
