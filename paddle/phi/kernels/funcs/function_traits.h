/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

namespace phi {
namespace funcs {
template <int Arity, typename... Args>
struct IsPointerArgs {
  static_assert(Arity == sizeof...(Args), "Arity and Args not match!");
  static const bool value = false;
};

template <typename... Args>
struct IsPointerArgs<1, Args...> {
  static_assert(1 == sizeof...(Args), "Arity and Args not match!");
  static const bool value = std::is_pointer<
      typename std::tuple_element<0, std::tuple<Args...>>::type>::value;
};

// Declare a template class with a single template parameter.
template <typename>
struct FunctionTraits;

// A forwarding trait allowing functors (objects which have an operator())
// to be used with this traits class.
template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {};

// A partial specialization of FunctionTraits for pointers to member functions
// and has const/non-const class member functions.
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const>
    : public FunctionTraits<ReturnType(Args...)> {};
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...)>
    : public FunctionTraits<ReturnType(Args...)> {};

// An implementation for common function.
template <typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType(Args...)> {
  static const size_t arity = sizeof...(Args);
  static const bool has_pointer_args = IsPointerArgs<arity, Args...>::value;
  using ArgsTuple = std::tuple<Args...>;
};

}  // namespace funcs
}  // namespace phi
