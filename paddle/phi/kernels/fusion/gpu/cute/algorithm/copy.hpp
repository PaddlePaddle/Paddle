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

#include <cute/tensor.hpp>
#include <cute/tensor_predicate.hpp>

#include <cute/atom/copy_atom.hpp>

namespace cute {

//
// Accept mutable temporaries
//

template <class PrdTensor,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_if(PrdTensor const& pred,
                              Tensor<SrcEngine, SrcLayout> const& src,
                              Tensor<DstEngine, DstLayout>&& dst) {
  return copy_if(pred, src, dst);
}

template <class... CopyArgs,
          class PrdTensor,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_if(Copy_Atom<CopyArgs...> const& copy_atom,
                              PrdTensor const& pred,
                              Tensor<SrcEngine, SrcLayout> const& src,
                              Tensor<DstEngine, DstLayout>&& dst) {
  return copy_if(copy_atom, pred, src, dst);
}

template <class VecType,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
                               Tensor<DstEngine, DstLayout>&& dst) {
  return copy_vec<VecType>(src, dst);
}

template <class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void copy(Tensor<SrcEngine, SrcLayout> const& src,
                           Tensor<DstEngine, DstLayout>&& dst) {
  return copy(src, dst);
}

template <class... CopyArgs,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy(Copy_Atom<CopyArgs...> const& copy_atom,
                           Tensor<SrcEngine, SrcLayout> const& src,
                           Tensor<DstEngine, DstLayout>&& dst) {
  return copy(copy_atom, src, dst);
}

//
// copy_if -- Predicated Copy
//

template <class PrdTensor,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_if(PrdTensor const& pred,
                              Tensor<SrcEngine, SrcLayout> const& src,
                              Tensor<DstEngine, DstLayout>& dst) {
  auto copy_op = select_elementwise_copy(src, dst);

  CUTE_UNROLL
  for (int i = 0; i < size(src); ++i) {
    if (pred(i)) {
      copy_op.copy(src(i), dst(i));
    }
  }
}

//
// copy_if -- Predicated CopyAtom
//

template <class... CopyArgs,
          class PredTensor,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_if(
    Copy_Atom<CopyArgs...> const& copy_atom,
    PredTensor const& pred,                   // (Rest...)
    Tensor<SrcEngine, SrcLayout> const& src,  // (V,Rest...)
    Tensor<DstEngine, DstLayout>& dst)        // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
  if constexpr (SrcLayout::rank == 1) {  // Dispatch the copy
    copy_atom.call(src, dst);
  } else {  // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    auto src_v = group_modes<1, R>(src);
    auto dst_v = group_modes<1, R>(dst);
    CUTE_UNROLL
    for (int i = 0; i < size<1>(src_v); ++i) {
      if (pred(i)) {
        copy_atom.call(src_v(_, i), dst_v(_, i));
      }
    }
  }
}

//
// copy_vec -- attempt vectorized copy with VecType
//

template <class VecType,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
                               Tensor<DstEngine, DstLayout>& dst) {
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  if constexpr (sizeof(SrcType) == sizeof(DstType) &&
                sizeof(VecType) > sizeof(DstType)) {
    /* @pre  is_aligned<N>(src.data()) &&
     *       is_aligned<N>(dst.data())
     */
    auto src_v = recast<VecType const>(src);
    auto dst_v = recast<VecType>(dst);

#if 0
    if (thread0()) {
      print("copy_vec -- vectorizing copy from %3db to %3db\n", int(8*sizeof(SrcType)), int(8*sizeof(VecType)));
      print("   "); print(layout(src)); print(" => "); print(layout(src_v)); print("\n");
      print("   "); print(layout(dst)); print(" => "); print(layout(dst_v)); print("\n");
    }
#endif

    return copy_if(TrivialPredTensor{}, src_v, dst_v);
  } else {
#if 0
  if (thread0()) {
    print("copy_vec -- not vectorizing, copy with %3db and %3db\n", int(8*sizeof(SrcType)), int(8*sizeof(DstType)));
    print("   "); print(layout(src)); print("\n");
    print("   "); print(layout(dst)); print("\n");
  }
#endif

    return copy_if(TrivialPredTensor{}, src, dst);
  }
}

//
// copy -- auto-vectorizing copy
//

template <class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void copy(Tensor<SrcEngine, SrcLayout> const& src,
                           Tensor<DstEngine, DstLayout>& dst) {
  constexpr int N = decltype(max_common_vector(src, dst))::value;

#if 0
  if (thread0()) {
    print("copy -- found a max_common_vector of %d\n", N);
    print("   "); print(src.data()); print(" o "); print(layout(src)); print("\n");
    print("   "); print(dst.data()); print(" o "); print(layout(dst)); print("\n");
  }
#endif

  if constexpr (N <= 1) {
    return copy_if(TrivialPredTensor{}, src, dst);
  } else {
    constexpr int vec_bits =
        N * sizeof_bits<typename SrcEngine::value_type>::value;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;
    return copy_vec<VecType>(src, dst);
  }
}

//
// copy -- CopyAtom
//

template <class... CopyArgs,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy(Copy_Atom<CopyArgs...> const& copy_atom,
                           Tensor<SrcEngine, SrcLayout> const& src,
                           Tensor<DstEngine, DstLayout>& dst) {
  return copy_if(copy_atom, TrivialPredTensor{}, src, dst);
}

template <class... CopyArgs,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void copy(Copy_Atom<DefaultCopy, CopyArgs...> const&,
                           Tensor<SrcEngine, SrcLayout> const& src,
                           Tensor<DstEngine, DstLayout>& dst) {
  return copy(src, dst);
}

}  // end namespace cute
