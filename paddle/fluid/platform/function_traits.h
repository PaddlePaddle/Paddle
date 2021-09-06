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

// Declare a template class with a single template parameter.
template <typename>
struct FunctionTraits;

// A forwarding trait allowing functors (objects which have an operator())
// to be used with this traits class.
template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {};

// A partial specialization of FunctionTraits for pointers to member functions.
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const> {
  typedef ReturnType result_type;

  template <size_t i>
  struct Arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type Type;
  };

  static const size_t arity = sizeof...(Args);
  static const bool has_pointer_args =
      (arity == 1) &&
      (std::is_pointer<
          typename std::tuple_element<0, std::tuple<Args...>>::type>::value);
};

namespace details {

template <typename InT, typename OutT, typename Functor, int Arity,
          bool HasPointerArgs = false>
struct CallFunctorImpl {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]);
};

template <typename InT, typename OutT, typename Functor>
struct CallFunctorImpl<InT, OutT, Functor, 1, true> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args);
  }
};

template <typename InT, typename OutT, typename Functor>
struct CallFunctorImpl<InT, OutT, Functor, 1, false> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args[0]);
  }
};

template <typename InT, typename OutT, typename Functor, bool HasPointerArgs>
struct CallFunctorImpl<InT, OutT, Functor, 2, HasPointerArgs> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args[0], args[1]);
  }
};

template <typename InT, typename OutT, typename Functor, bool HasPointerArgs>
struct CallFunctorImpl<InT, OutT, Functor, 3, HasPointerArgs> {
  HOSTDEVICE inline OutT operator()(Functor func, InT args[]) {
    return func(args[0], args[1], args[2]);
  }
};

}  // namespace details

template <typename InT, typename OutT, typename Functor>
HOSTDEVICE inline OutT CallFunctor(Functor func, InT args[]) {
  using Traits = FunctionTraits<Functor>;

  return details::CallFunctorImpl<InT, OutT, Functor, Traits::arity,
                                  Traits::has_pointer_args>()(func, args);
}

}  // namespace platform
}  // namespace paddle
