/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cute/config.hpp>

#include <cute/arch/mma.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
     defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUTE_ARCH_MMA_SM90_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMMA Descriptor and utilities

// GMMA enums and utilities
namespace GMMA {

enum class LayoutType : uint8_t {
  INTERLEAVE = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

CUTE_HOST_DEVICE char const* to_string(LayoutType const& t) {
  switch (t) {
    case LayoutType::INTERLEAVE:
      return "INTERLEAVE";
    case LayoutType::B128:
      return "B128";
    case LayoutType::B64:
      return "B64";
    case LayoutType::B32:
      return "B32";
  }
  return nullptr;
}

// Output operator for all enums in this namespace
CUTE_HOST std::ostream& operator<<(std::ostream& os, LayoutType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

}  // end namespace GMMA

union GmmaDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4;  // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;  // 6 bits unused, 2 bits [6,8)
  };

  // Decay to a uint64_t
  CUTE_HOST_DEVICE constexpr operator uint64_t() const noexcept {
    return desc_;
  }

  // Printer
  CUTE_HOST_DEVICE friend void print(GmmaDescriptor const& t) {
    printf("GmmaDescriptor: 0x%016lx\n", t.desc_);
    printf("  start_addr :  0x%04x\n", t.start_address_);
    printf("  leading_off:  0x%04x (%d)\n",
           t.leading_byte_offset_,
           t.leading_byte_offset_);
    printf("  stride_off :  0x%04x (%d)\n",
           t.stride_byte_offset_,
           t.stride_byte_offset_);
    printf("  base_offset:  0x%01x\n", t.base_offset_);
    printf("  layout_type:  0x%01x (%s)\n",
           t.layout_type_,
           to_string(static_cast<GMMA::LayoutType>(t.layout_type_)));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////
