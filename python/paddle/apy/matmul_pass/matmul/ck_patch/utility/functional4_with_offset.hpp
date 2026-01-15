// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_FUNCTIONAL4_VARIADIC_HPP
#define CK_FUNCTIONAL4_VARIADIC_HPP

#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/array.hpp"

namespace ck {

namespace detail {

template <typename Seq0, typename Seq1>
struct unpack2_impl_with_offset;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl_with_offset<Sequence<Is...>, Sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    __host__ __device__ constexpr auto operator()(F&& f, X&& x, Y&& y, const index_t batch, const index_t row_offset, const index_t col_offset) const
    {
        return std::forward<F>(f)(std::forward<X>(x).At(Number<Is>{})...,
                                  std::forward<Y>(y).At(Number<Js>{})..., batch, row_offset, col_offset);
    }
};

} // namespace detail

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto unpack2_with_offset(F&& f, X&& x, Y&& y, const index_t batch, const index_t row_offset, const index_t col_offset)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return detail::unpack2_impl_with_offset<typename arithmetic_sequence_gen<0, X_::Size(), 1>::type,
                                            typename arithmetic_sequence_gen<0, Y_::Size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y), batch, row_offset, col_offset);
}

} // namespace ck
#endif
