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
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "paddle/pir/include/core/dll_decl.h"

namespace pir {
namespace detail {
///
/// \brief Equivalent to boost::hash_combine.
///
IR_API std::size_t hash_combine(std::size_t lhs, std::size_t rhs);

///
/// \brief Aligned malloc and free functions.
///
void *aligned_malloc(size_t size, size_t alignment);

void aligned_free(void *mem_ptr);

///
/// \brief Some template methods for manipulating std::tuple.
///
/// (1) Pop front element from Tuple
template <typename Tuple>
struct PopFrontT;

template <typename Head, typename... Tail>
struct PopFrontT<std::tuple<Head, Tail...>> {
 public:
  using Type = std::tuple<Tail...>;
};

template <typename Tuple>
using PopFront = typename PopFrontT<Tuple>::Type;

/// (2) Push front element to Tuple
template <typename NewElement, typename Tuple>
struct PushFrontT;

template <typename NewElement, typename... Elements>
struct PushFrontT<NewElement, std::tuple<Elements...>> {
 public:
  using Type = std::tuple<NewElement, Elements...>;
};

template <typename NewElement, typename... Elements>
struct PushFrontT<std::tuple<NewElement>, std::tuple<Elements...>> {
 public:
  using Type = std::tuple<NewElement, Elements...>;
};

template <typename NewElement, typename Tuple>
using PushFront = typename PushFrontT<NewElement, Tuple>::Type;

/// (3) IsEmpty
template <typename Tuple>
struct IsEmpty {
  static constexpr bool value = false;
};

template <>
struct IsEmpty<std::tuple<>> {
  static constexpr bool value = true;
};

/// (4) IfThenElseT
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

/// (5) Filter out all types inherited from BaseT from the tuple.
template <template <typename> class BaseT,
          typename Tuple,
          bool Empty = IsEmpty<Tuple>::value>
struct Filter;

template <template <typename> class BaseT, typename Tuple>
struct Filter<BaseT, Tuple, false> {
 private:
  using Matched =
      IfThenElse<std::is_base_of<BaseT<std::tuple_element_t<0, Tuple>>,
                                 std::tuple_element_t<0, Tuple>>::value,
                 std::tuple<std::tuple_element_t<0, Tuple>>,
                 std::tuple<>>;
  using Rest = typename Filter<BaseT, PopFront<Tuple>>::Type;

 public:
  using Type =
      IfThenElse<IsEmpty<Matched>::value, Rest, PushFront<Matched, Rest>>;
};

// basis case:
template <template <typename> class BaseT, typename Tuple>
struct Filter<BaseT, Tuple, true> {
  using Type = std::tuple<>;
};

template <typename ForwardIterator, typename UnaryFunctor, typename NullFunctor>
void PrintInterleave(ForwardIterator begin,
                     ForwardIterator end,
                     UnaryFunctor print_func,
                     NullFunctor between_func) {
  if (begin == end) return;
  print_func(*begin);
  begin++;
  for (; begin != end; begin++) {
    between_func();
    print_func(*begin);
  }
}

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;
template <class, template <class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
};

template <template <class...> class Op, class... Args>
struct detector<void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detector<void, Op, Args...>::value_t;

// Print content as follows.
// ===-------------------------------------------------------------------------===
//                                     header
// ===-------------------------------------------------------------------------===
void PrintHeader(const std::string &header, std::ostream &os);

}  // namespace detail
}  // namespace pir
