// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
