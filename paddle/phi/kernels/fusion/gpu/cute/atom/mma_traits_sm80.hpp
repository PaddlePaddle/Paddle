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

#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

#include <cute/numeric/integer_subbyte.hpp>

#include <cutlass/numeric_types.h>

namespace cute {

namespace {

// (T32,V1) -> (M8,N8)
using SM80_8x4 = Layout<Shape<Shape<_4, _8>, _1>, Stride<Stride<_8, _1>, _0>>;
// (T32,V2) -> (M8,N8)
using SM80_8x8_Row =
    Layout<Shape<Shape<_4, _8>, _2>, Stride<Stride<_16, _1>, _8>>;
// (T32,V4) -> (M8,N16)
using SM80_8x16_Row =
    Layout<Shape<Shape<_4, _8>, _4>, Stride<Stride<_32, _1>, _8>>;
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                             Stride<Stride<_32, _1>, Stride<_16, _8>>>;

}  // namespace

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN> {
  using ElementDVal = half_t;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = half_t;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID = Layout<_32>;
  using ALayout = SM80_16x8_Row;
  using BLayout = SM80_8x8_Row;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> {
  using ElementDVal = half_t;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = half_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  using CLayout = SM80_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = fp16 * fp16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>
    : MMA_Traits<SM80_16x8x8_F16F16F16F16_TN> {
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;
};

template <>
struct MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
    : MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> {
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = bf16 * bf16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F32BF16BF16F32_TN>
    : MMA_Traits<SM80_16x8x8_F16F16F16F16_TN> {
  using ElementDVal = float;
  using ElementAVal = bfloat16_t;
  using ElementBVal = bfloat16_t;
  using ElementCVal = float;
};

template <>
struct MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>
    : MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> {
  using ElementDVal = float;
  using ElementAVal = bfloat16_t;
  using ElementBVal = bfloat16_t;
  using ElementCVal = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = tf32 * tf32 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x4_F32TF32TF32F32_TN> {
  using ElementDVal = float;
  using ElementAVal = cutlass::tfloat32_t;
  using ElementBVal = cutlass::tfloat32_t;
  using ElementCVal = float;

  using Shape_MNK = Shape<_16, _8, _4>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, _2>, Stride<Stride<_16, _1>, _8>>;
  using BLayout = SM80_8x4;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN> {
  using ElementDVal = float;
  using ElementAVal = cutlass::tfloat32_t;
  using ElementBVal = cutlass::tfloat32_t;
  using ElementCVal = float;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, _2>, Stride<Stride<_8, _1>, _32>>;
  using CLayout = SM80_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp64 = fp64 * fp64 + fp64 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> {
  using ElementDVal = double;
  using ElementAVal = double;
  using ElementBVal = double;
  using ElementCVal = double;

  using Shape_MNK = Shape<_8, _8, _4>;
  using ThrID = Layout<_32>;
  using ALayout = SM80_8x4;
  using BLayout = SM80_8x4;
  using CLayout = SM80_8x8_Row;
};

// Custom complex fp64 MMA composed of 4 fp64 MMAs -- same layouts
template <>
struct MMA_Traits<SM80_8x8x4_C64C64C64C64_TN>
    : MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> {
  using ElementDVal = complex<double>;
  using ElementAVal = complex<double>;
  using ElementBVal = complex<double>;
  using ElementCVal = complex<double>;
};

// Custom complex fp64 MMA composed of 3 fp64 MMAs -- same layouts
template <>
struct MMA_Traits<SM80_8x8x4_GC64C64C64GC64_TN>
    : MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> {
  using ElementDVal = typename SM80_8x8x4_GC64C64C64GC64_TN::GaussComplex;
  using ElementAVal = complex<double>;
  using ElementBVal = complex<double>;
  using ElementCVal = typename SM80_8x8x4_GC64C64C64GC64_TN::GaussComplex;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;

  using Shape_MNK = Shape<_8, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = SM80_8x16_Row;
  using BLayout = SM80_8x16_Row;
  using CLayout = SM80_8x8_Row;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32S8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                         Stride<Stride<_64, _1>, Stride<_16, _8>>>;
  using BLayout = SM80_8x16_Row;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>,
                         Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                         Stride<Stride<_32, _1>, Stride<_8, _128>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32S8U8S32_TN>
    : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32S8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_8x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8U8S32_TN>
    : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8U8S32_TN>
    : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x32_S32S8U8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32U8S8S32_TN>
    : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32U8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_8x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8S8S32_TN>
    : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8S8S32_TN>
    : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8S8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x32_S32U8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32U8U8S32_TN>
    : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32U8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_8x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8U8S32_TN>
    : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8U8S32_TN>
    : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8U8S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x32_S32U8U8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = b1 ^ b1 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x256_S32U1U1S32_TN_XORPOPC> {
  using ElementDVal = int32_t;
  using ElementAVal = cute::uint1b_t;
  using ElementBVal = cute::uint1b_t;
  using ElementCVal = int32_t;

  using Shape_MNK = Shape<_16, _8, _256>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<_32, Shape<_8, _4, _2, _2>>,
                         Stride<_64, Stride<_64, _16, _8, _2048>>>;
  using BLayout =
      Layout<Shape<_32, Shape<_32, _2>>, Stride<_32, Stride<_1, _1024>>>;
  using CLayout = SM80_16x8_Row;
};
}  // end namespace cute
