/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
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

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

// Tag which triggers MMA which will trigger
struct OpMultiplyAddDequantizeInterleavedBToA;

/*
  Below we have extra tags to signal what kind of dequantization we want to do
  (per col, scale only fine grained, finegrained with zero). This still lets us
  the existing template infrastructure (incl. that in CUTLASS). However, we
  split out the template below into OpMultiplyAddDequantizeInterleavedBToA along
  with the quantization op before instantiating the GEMM pieces.

  Note that this is somewhat of a hack, but it SIGNIFICANTLY reduces the amount
  of code we need to duplicate.
 */
struct OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
struct OpMultiplyAddDequantizeInterleavedBToA_fine_grained_scale;

// The default just forwards the original operator
template <typename MmaOp, bool FineGrained>
struct TagOperator {
  using TaggedOperator = MmaOp;
};

// Specializations below attach more information to the operator
template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, false> {
  using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
};

template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, true> {
  using TaggedOperator =
      OpMultiplyAddDequantizeInterleavedBToA_fine_grained_scale;
};

// Here we instantiate some structs to "detag" the tagged operator. It splits it
// back to the original operator + the extra information. If no extra info was
// tagged, the dequant op per column scaling as a default.
template <typename TaggedMmaOp>
struct DetagOperator {
  using Operator = TaggedMmaOp;
  static constexpr bool FineGrained = false;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_percol_scale> {
  using Operator = OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr bool FineGrained = false;
};

template <>
struct DetagOperator<
    OpMultiplyAddDequantizeInterleavedBToA_fine_grained_scale> {
  using Operator = OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr bool FineGrained = true;
};

}  // namespace arch
}  // namespace cutlass
