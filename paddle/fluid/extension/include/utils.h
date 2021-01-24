/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <array>
#include <functional>
#include <tuple>
#include <type_traits>

namespace paddle {
namespace detail {

// template <typename F>
// struct function_traits {
//   static_assert(!std::is_same<F, F>::value, "In function_traits<F>, F must be
//   a plain function type.");
// };

// template <typename ResultType, typename... Args>
// struct function_traits<ResultType (Args...)> {
//   using func_type = ResultType(Args...);
//   using return_type = ResultType;
//   using argument_types = typelist<Args...>;
//   static constexpr auto argument_num = sizeof...(Args);
// };

// template <typename ResultType, typename... Args>
// struct infer_function_traits<ResultType (*)(Args...)> {
//   using type = function_traits<ResultType(Args...)>;
// };

// template <typename ResultType, typename... Args>
// struct infer_function_traits<ResultType (Args...)> {
//   using type = function_traits<ResultType(Args...)>;
// };

// template <typename T>
// using infer_function_traits_t = typename infer_function_traits<T>::type;

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// function_traits<decltype(&s::f)> traits;
template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  typedef std::tuple<Args...> ArgsTuple;
  typedef ReturnType result_type;
  using Function = std::function<ReturnType(Args...)>;

  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    // the i-th argument is equivalent to the i-th tuple element of a tuple
    // composed of those arguments.
  };
};

// template <typename traits, std::size_t... INDEX>
// typename traits::ArgsTuple
// dereference_impl()

// template <typename traits>
// typename traits::ArgsTuple
// dereference()
}  // namespace detail
}  // namespace paddle
