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

#include <cute/arch/mma.hpp>

#include <cute/layout.hpp>

namespace cute {

template <class MMAOperation, class... MMAOpArgs>
struct MMA_Traits {
  static_assert(sizeof(MMAOperation) == 0,
                "MMA_Traits not implemented for this MMA_Operation.");
};

template <class D, class A, class B, class C>
struct MMA_Traits<UniversalFMA<D, A, B, C>> {
  using ElementDVal = D;
  using ElementAVal = A;
  using ElementBVal = B;
  using ElementCVal = C;

  // Logical shape of the MMA
  using Shape_MNK = Shape<_1, _1, _1>;

  // Logical thread id (tid) -> tidx
  using ThrID = Layout<_1>;

  // (Logical thread id (tid), Logical value id (vid)) -> coord

  // (tid,vid) -> (m,k)
  using ALayout = Layout<Shape<_1, _1>>;
  // (tid,vid) -> (n,k)
  using BLayout = Layout<Shape<_1, _1>>;
  // (tid,vid) -> (m,n)
  using CLayout = Layout<Shape<_1, _1>>;
};

}  // namespace cute
