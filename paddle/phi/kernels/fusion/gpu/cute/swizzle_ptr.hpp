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

#include <cute/arch/util.hpp>

#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>

#include <cute/container/array.hpp>
#include <cute/numeric/int.hpp>
#include <cute/pointer.hpp>

/* This implements a swizzle pointer of the form
 *   InvolutionFn o PtrAdd
 * where the InvolutionFn need not be linear.
 *
 * This differs subtly from swizzle_layout because the smem pointer is used
 * as the offset. That means that swizzle_layout will implement
 * position-independent swizzle layouts, while swizzle_ptr implements
 * position-dependent swizzle tensors. Arch chose to design hardware with
 * position-dependent swizzles.
 *
 * For clarity:
 *   NormalLayout  : DeRef <- PtrAdd <- [Layout]
 *   ComposedLayout: DeRef <- PtrAdd <- [Swizzle <- OffsetAdd <- Layout]
 *   SwizzlePtr    : [DeRef <- Swizzle <- PtrAdd] <- Layout
 *
 * Furthermore, for known swizzles, this pointer attempts to decay itself
 *    to a normal-pointer with a new layout containing dynamic or static
 * strides. This is possible by determining the subdomain of the InvolutionFn
 *    that is identity and testing if the Layout's codomain is contained
 *    within it.
 */

namespace cute {

template <class T, class Swizzle>
struct smem_ptr_swizzle {
  static_assert(std::is_empty<Swizzle>::value, "Swizzle can't have state.");

  CUTE_HOST_DEVICE constexpr T* get() const { return ptr_; }

  CUTE_HOST_DEVICE constexpr static Swizzle get_swizzle() { return {}; }

  CUTE_HOST_DEVICE constexpr static T* apply_swizzle(T* ptr) {
    return reinterpret_cast<T*>(
        Swizzle::apply(reinterpret_cast<std::uintptr_t>(ptr)));
  }

  CUTE_HOST_DEVICE constexpr T& operator*() const {
    return *apply_swizzle(get());
  }

  template <class Int>
  CUTE_HOST_DEVICE constexpr T& operator[](Int const& i) const {
    return *apply_swizzle(get() + i);
  }

  template <class Int>
  CUTE_HOST_DEVICE constexpr smem_ptr_swizzle operator+(Int const& i) const {
    return {ptr_ + i};
  }

  T* ptr_;
};

template <class T, class S>
struct is_smem<smem_ptr_swizzle<T, S>> : true_type {};

// Make a swizzle pointer
template <class T, class Swizzle>
CUTE_HOST_DEVICE constexpr auto make_smem_ptr(T* ptr, Swizzle const& swizzle) {
  return smem_ptr_swizzle<T, Swizzle>{ptr};
}

// A model of a nullptr smem_ptr<T> with B == sizeof_bits<T>::value
// That represents an unset pointer. This is a placeholder type that is waiting
// for an smem_ptr
template <int Bits>
struct smem_ptr_flag_bits : Int<0> {};

using smem_ptr_flag = smem_ptr_flag_bits<1>;

// A flagged construction method to transform ComposedLayout
// Make a swizzle pointer tensor and check that the intended type size matches
template <class T, class Swizzle, int B, class Layout>
CUTE_HOST_DEVICE constexpr auto make_tensor(
    smem_ptr<T> const& ptr,
    ComposedLayout<Swizzle, smem_ptr_flag_bits<B>, Layout> const& layout) {
  static_assert(B == sizeof_bits<T>::value, "Expected a B-bit pointer type.");
  return make_tensor(make_smem_ptr(ptr.get(), layout.swizzle_fn()),
                     layout.layout_fn());
}

// Specialization for immediate decay
template <class T, int M, int S, class LShape, class LStride>
CUTE_HOST_DEVICE constexpr auto make_tensor(
    smem_ptr_swizzle<T, Swizzle<0, M, S>>& p,
    Layout<LShape, LStride> const& layout) {
  return make_tensor(make_smem_ptr(p.ptr_), layout);
}

template <class T, int M, int S, class LShape, class LStride>
CUTE_HOST_DEVICE constexpr auto make_tensor(
    smem_ptr_swizzle<T, Swizzle<0, M, S>> const& p,
    Layout<LShape, LStride> const& layout) {
  return make_tensor(make_smem_ptr(p.ptr_), layout);
}

// NOTE: To preserve smem_ptr_flag_bits under recast ops
template <int N, class Swizzle, int B, class Layout>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Swizzle, smem_ptr_flag_bits<B>, Layout> const& layout) {
  return composition(layout.swizzle_fn(),
                     smem_ptr_flag_bits<B * N>{},
                     upcast<N>(layout.layout_fn()));
}

template <int N, class Swizzle, int B, class Layout>
CUTE_HOST_DEVICE constexpr auto downcast(
    ComposedLayout<Swizzle, smem_ptr_flag_bits<B>, Layout> const& layout) {
  return composition(layout.swizzle_fn(),
                     smem_ptr_flag_bits<B / N>{},
                     downcast<N>(layout.layout_fn()));
}

//
// Recast
//   Swizzle operates on the pointer address, so it doesn't care about the type
//

template <class NewT, class T, class Swizzle>
CUTE_HOST_DEVICE constexpr auto recast(
    smem_ptr_swizzle<T, Swizzle> const& ptr) {
  return smem_ptr_swizzle<NewT, Swizzle>{recast<NewT>(ptr.ptr_)};
}

template <class NewT, class T, class Swizzle>
CUTE_HOST_DEVICE constexpr auto recast(
    smem_ptr_swizzle<T const, Swizzle> const& ptr) {
  return smem_ptr_swizzle<NewT const, Swizzle>{recast<NewT const>(ptr.ptr_)};
}

//
// Conversion with swizzle_layout
//

template <class T, class Swizzle, int B, class Layout>
CUTE_HOST_DEVICE auto as_position_independent_swizzle_layout(
    ComposedLayout<Swizzle, smem_ptr_flag_bits<B>, Layout> const& layout) {
  return composition(recast<uint_bit_t<8>, uint_bit_t<B>>(layout.swizzle_fn()),
                     Int<0>{},
                     layout.layout_fn());
}

template <class T, class Swizzle, class Layout>
CUTE_HOST_DEVICE auto as_position_independent_swizzle_tensor(
    Tensor<ViewEngine<smem_ptr_swizzle<T, Swizzle>>, Layout> const& tensor) {
  {
    uint32_t address = cast_smem_ptr_to_uint(tensor.data().get());
    uint32_t mask =
        ((uint32_t(1) << Swizzle::num_base) - 1) & (Swizzle::swizzle_code);
    assert((address & mask) ==
           0);  // Alignment to the Base, Z, and Y of Swizzle
  }
  auto new_swizzle = recast<uint_bit_t<8>, uint_bit_t<sizeof_bits_v<T>>>(
      tensor.data().get_swizzle());
  return make_tensor(make_smem_ptr(tensor.data().get()),
                     composition(new_swizzle, Int<0>{}, tensor.layout()));
}

template <class T, class Swizzle, class Layout>
CUTE_HOST_DEVICE auto as_position_independent_swizzle_tensor(
    Tensor<ViewEngine<smem_ptr_swizzle<T, Swizzle>>, Layout>& tensor) {
  {
    uint32_t address = cast_smem_ptr_to_uint(tensor.data().get());
    uint32_t mask =
        ((uint32_t(1) << Swizzle::num_base) - 1) & (Swizzle::swizzle_code);
    assert((address & mask) ==
           0);  // Alignment to the Base, Z, and Y of Swizzle
  }
  auto new_swizzle = recast<uint_bit_t<8>, uint_bit_t<sizeof_bits_v<T>>>(
      tensor.data().get_swizzle());
  return make_tensor(make_smem_ptr(tensor.data().get()),
                     composition(new_swizzle, Int<0>{}, tensor.layout()));
}

template <class T, class Swizzle, class Layout>
CUTE_HOST_DEVICE auto as_position_independent_swizzle_tensor(
    Tensor<ViewEngine<smem_ptr_swizzle<T, Swizzle>>, Layout>&& tensor) {
  return as_position_independent_swizzle_tensor(tensor);
}

//
// Print
//

// Capture and cast smem_ptr_flag Layouts to offset-0 layouts
template <class Swizzle, int B, class Layout>
CUTE_HOST_DEVICE void print_latex(
    ComposedLayout<Swizzle, smem_ptr_flag_bits<B>, Layout> const& layout) {
  auto new_swizzle = recast<uint_bit_t<8>, uint_bit_t<B>>(layout.swizzle_fn());
  print_latex(composition(new_swizzle, Int<0>{}, layout.layout_fn()));
}

template <int B>
CUTE_HOST_DEVICE void print(smem_ptr_flag_bits<B> const& ptr) {
  printf("smem_ptr_%db(unset)", B);
}

template <class T, int B, int M, int S>
CUTE_HOST_DEVICE void print(smem_ptr_swizzle<T, Swizzle<B, M, S>> const& ptr) {
  printf(
      "smem_ptr_S<%d,%d,%d>_%db(%p)", B, M, S, int(8 * sizeof(T)), ptr.get());
}

template <class T, int B, int M, int S>
CUTE_HOST std::ostream& operator<<(
    std::ostream& os, smem_ptr_swizzle<T, Swizzle<B, M, S>> const&) {
  return os << "smem_ptr_S<" << B << "," << M << "," << S << ">_"
            << int(8 * sizeof(T)) << "b";
}

}  // end namespace cute
