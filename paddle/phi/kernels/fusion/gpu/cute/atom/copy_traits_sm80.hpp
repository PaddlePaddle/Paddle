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

#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/layout.hpp>

namespace cute {

template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S, D>> {
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<S, D>> {
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Element copy selector
template <class SrcTensor, class DstTensor>
CUTE_HOST_DEVICE constexpr auto select_elementwise_copy(SrcTensor const&,
                                                        DstTensor const&) {
  using SrcType = typename SrcTensor::value_type;
  using DstType = typename DstTensor::value_type;

#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  if constexpr (is_gmem<SrcTensor>::value && is_smem<DstTensor>::value &&
                sizeof(SrcType) == sizeof(DstType) &&
                (sizeof(SrcType) == 4 || sizeof(SrcType) == 8 ||
                 sizeof(SrcType) == 16)) {
    return SM80_CP_ASYNC_CACHEALWAYS<SrcType, DstType>{};
  } else {
    return UniversalCopy<SrcType, DstType>{};
  }

  CUTE_GCC_UNREACHABLE;
#else
  return UniversalCopy<SrcType, DstType>{};
#endif
}

}  // namespace cute
