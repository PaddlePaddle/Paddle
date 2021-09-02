/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.1

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <tuple>
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace platform {

template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const> {
  typedef ReturnType result_type;

  template <size_t i>
  struct Arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type Type;
  };

  static const size_t arity = sizeof...(Args);
};

namespace details {

template <int Arity, typename InT, typename OutT, typename Functor>
struct ApplyImpl {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]);
};

template <typename InT, typename OutT, typename Functor>
struct ApplyImpl<1, InT, OutT, Functor> {
  using Traits = FunctionTraits<Functor>;
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    if (std::is_pointer<typename Traits::template Arg<0>::Type>::value) {
      return func(args);
    }
    return func(args[0]);
  }
};

template <typename InT, typename OutT, typename Functor>
struct ApplyImpl<2, InT, OutT, Functor> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args[0], args[1]);
  }
};

template <typename InT, typename OutT, typename Functor>
struct ApplyImpl<3, InT, OutT, Functor> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args[0], args[1], args[2]);
  }
};

}  // namespace details

template <typename InT, typename OutT, typename Functor>
HOSTDEVICE inline OutT Apply(Functor func, InT args[]) {
  using Traits = FunctionTraits<Functor>;
  static_assert(Traits::arity < 4,
                "Only functor whose's arity less than 4 is suuported.");

  return details::ApplyImpl<Traits::arity, InT, OutT, Functor>()(func, args);
}

}  // namespace platform
}  // namespace paddle
