// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ir {
///
/// \brief Equivalent to boost::hash_combine.
///
std::size_t hash_combine(std::size_t lhs, std::size_t rhs);

///
/// \brief Aligned malloc and free functions.
///
void *aligned_malloc(size_t size, size_t alignment);

void aligned_free(void *mem_ptr);

///
/// \brief Type list template tools.
///
template <typename... Elements>
struct TypeList {
  static constexpr unsigned size = sizeof...(Elements);
};

// Get front element from a TypeList
template <typename List>
struct FrontT;

template <typename Front, typename... Tail>
struct FrontT<TypeList<Front, Tail...>> {
  using Type = Front;
};

template <typename List>
using Front = typename FrontT<List>::Type;

// Pop front element from TypeList
template <typename List>
struct PopFrontT;

template <typename Head, typename... Tail>
struct PopFrontT<TypeList<Head, Tail...>> {
  using Type = TypeList<Tail...>;
};

template <typename List>
using PopFront = typename PopFrontT<List>::Type;

// push element to TypeList front
template <typename NewElement, typename List>
struct PushFrontT;

template <typename NewElement, typename... Elements>
struct PushFrontT<NewElement, TypeList<Elements...>> {
  using Type = TypeList<NewElement, Elements...>;
};

template <typename NewElement, typename... Elements>
struct PushFrontT<TypeList<NewElement>, TypeList<Elements...>> {
  using Type = TypeList<NewElement, Elements...>;
};

template <typename NewElement, typename List>
using PushFront = typename PushFrontT<NewElement, List>::Type;

// Get Nth element from TypeList
template <typename List, unsigned N>
struct NthElementT : public NthElementT<PopFront<List>, N - 1> {};
// basis case:
template <typename List>
struct NthElementT<List, 0> : public FrontT<List> {};
template <typename List, unsigned N>
using NthElement = typename NthElementT<List, N>::Type;

// IsEmpty
template <typename List>
struct IsEmpty {
  static constexpr bool value = false;
};

template <>
struct IsEmpty<TypeList<>> {
  static constexpr bool value = true;
};

///
/// IfThenElseT
///
template <bool COND, typename TrueT, typename FalseT>
struct IfThenElseT {
  using Type = TrueT;
};

template <typename TrueT, typename FalseT>
struct IfThenElseT<false, TrueT, FalseT> {
  using Type = FalseT;
};

template <bool COND, typename TrueT, typename FalseT>
using IfThenElse = typename IfThenElseT<COND, TrueT, FalseT>::Type;

///
/// \brief Filter out all types inherited from BaseT from the type list.
///
template <typename BaseT, typename List, bool Empty = IsEmpty<List>::value>
struct Filter;

template <typename BaseT, typename List>
struct Filter<BaseT, List, false> {
 private:
  using Matched = IfThenElse<std::is_base_of<BaseT, Front<List>>::value,
                             TypeList<Front<List>>,
                             TypeList<>>;
  using Rest = typename Filter<BaseT, PopFront<List>>::Type;

 public:
  using Type =
      IfThenElse<IsEmpty<Matched>::value, Rest, PushFront<Matched, Rest>>;
};

// basis case:
template <typename BaseT, typename List>
struct Filter<BaseT, List, true> {
  using Type = TypeList<>;
};

}  // namespace ir
