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

#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/tensor.h"

namespace paddle {

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

namespace detail {

template <typename... Ts>
struct MakeVoid {
  using Type = void;
};

template <typename... Ts>
using Void = typename detail::MakeVoid<Ts...>::Type;

template <typename T, typename Enabled = void>
struct FunctionTraits;

template <typename Return_, typename... Args_>
struct FunctionTraits<Return_(Args_...)> {
  using Return = Return_;
  using Args = std::tuple<Args_...>;
  using Function = std::function<Return_(Args_...)>;

  enum : std::size_t { Arity = sizeof...(Args_) };

  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;
};

// Specialization for function pointer types.
template <typename Return, typename... Args>
struct FunctionTraits<Return (*)(Args...)> : FunctionTraits<Return(Args...)> {};

// Specialization for function reference types.
template <typename Return, typename... Args>
struct FunctionTraits<Return (&)(Args...)> : FunctionTraits<Return(Args...)> {};

// Specilization for method pointer types.
template <typename Class, typename Return, typename... Args>
struct FunctionTraits<Return (Class::*)(Args...)>
    : FunctionTraits<Return(Args...)> {};

// Specialization for const method pointer types.
template <typename Class, typename Return, typename... Args>
struct FunctionTraits<Return (Class::*)(Args...) const>
    : FunctionTraits<Return(Args...)> {};

// Specialization for functor types.
template <typename Op>
struct FunctionTraits<Op, Void<decltype(&Op::operator())>>
    : FunctionTraits<decltype(&Op::operator())> {};

}  // namespace detail

class TensorFunction {
 public:
  TensorFunction() = default;

  template <typename Func>
  void Wrap(Func&& func) {
    if (!func_.empty()) {
      throw std::runtime_error(
          "Repeat wrap error. The tensor function has contains function.");
    }
    func_ = std::move(func);
    func_type_ = std::type_index(typeid(Func));
  }

  template <typename Func>
  Func&& UnWrap() {
    try {
      return std::move(boost::any_cast<Func>(func_));
    } catch (boost::bad_any_cast&) {
      std::ostringstream err;
      err << "Unwrap TensorFunction error. Expected " << typeid(Func).name()
          << ", actual " << func_type_.name();
      throw std::runtime_error(err.str());
    }
  }

  template <typename Func>
  bool IsWrapped() const {
    return std::type_index(typeid(Func)) == func_type_;
  }

 private:
  boost::any func_;
  std::type_index func_type_ = std::type_index(typeid(void));
};

class OpFunction {
 public:
  OpFunction() = default;

  template <typename ForwardFunc>
  void SaveForwardFunc(ForwardFunc&& ff) {
    // 1. save args num
    using traits = detail::FunctionTraits<ForwardFunc>;
    using function_t = typename traits::Function;
    forward_in_num_ = traits::Arity;
    // 2. save func
    forward_func_ = TensorFunction();
    forward_func_.Wrap(static_cast<function_t>(std::forward<ForwardFunc>(ff)));
  }

  template <typename BackwardFunc>
  void SaveBackwardFunc(BackwardFunc&& bf) {
    // 1. save args num
    using traits = detail::FunctionTraits<BackwardFunc>;
    using function_t = typename traits::Function;
    backward_in_num_ = traits::Arity;
    // 2. save func
    backward_func_ = TensorFunction();
    backward_func_.Wrap(
        static_cast<function_t>(std::forward<BackwardFunc>(bf)));
  }

  size_t forward_in_num() const { return forward_in_num_; }
  size_t backward_in_num() const { return backward_in_num_; }

  const TensorFunction& forward_func() const { return forward_func_; }
  const TensorFunction& backward_func() const { return backward_func_; }

 private:
  size_t forward_in_num_;
  TensorFunction forward_func_;

  size_t backward_in_num_;
  TensorFunction backward_func_;

  // support infershape in the future
  TensorFunction infer_func_;
};

class OpFunctionMap {
 public:
  static OpFunctionMap& Instance() {
    static OpFunctionMap g_custom_op_function_holder;
    return g_custom_op_function_holder;
  }

  void Insert(const std::string& op_type, const OpFunction& funcs) {
    PADDLE_ENFORCE_NE(map_.find(op_type) != map_.end(), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", op_type));
    map_.insert({op_type, funcs});
  }

  const std::unordered_map<std::string, OpFunction>& map() { return map_; }

 private:
  OpFunctionMap() = default;

  std::unordered_map<std::string, OpFunction> map_;

  DISABLE_COPY_AND_ASSIGN(OpFunctionMap);
};

///////////////// Op Function Registrar ////////////////////////////

namespace detail {

template <bool at_end, size_t I, typename... FunctorTypes>
class OpFuncRegistrarFunctor;

// 0: forward functor
template <typename... FunctorTypes>
struct OpFuncRegistrarFunctor<false, 0, FunctorTypes...> {
  using ForwardFunctorType =
      typename std::tuple_element<0, std::tuple<FunctorTypes...>>::type;
  void operator()(const char* op_type, OpFunction* op_func) const {
    op_func->SaveForwardFunc(ForwardFunctorType());
    constexpr auto size = std::tuple_size<std::tuple<FunctorTypes...>>::value;
    OpFuncRegistrarFunctor<1 == size, 1, FunctorTypes...> func;
    func(op_type, op_func);
  }
};

// 1: backward functor
template <typename... FunctorTypes>
struct OpFuncRegistrarFunctor<false, 1, FunctorTypes...> {
  using BackwardFunctorType =
      typename std::tuple_element<1, std::tuple<FunctorTypes...>>::type;
  void operator()(const char* op_type, OpFunction* op_func) const {
    op_func->SaveBackwardFunc(BackwardFunctorType());
    constexpr auto size = std::tuple_size<std::tuple<FunctorTypes...>>::value;
    OpFuncRegistrarFunctor<2 == size, 2, FunctorTypes...> func;
    func(op_type, op_func);
  }
};

template <size_t I, typename... FunctorTypes>
struct OpFuncRegistrarFunctor<true, I, FunctorTypes...> {
  void operator()(const char* op_type, OpFunction* op_func) const {
    OpFunctionMap::Instance().Insert(op_type, *op_func);
  }
};

}  // namespace detail

class Registrar {
 public:
  void Touch() {}
};

template <typename... FunctorTypes>
struct CustomOperatorRegistrar : public Registrar {
  explicit CustomOperatorRegistrar(const char* op_type) {
    OpFunction op_func;
    detail::OpFuncRegistrarFunctor<false, 0, FunctorTypes...> func;
    func(op_type, &op_func);
  }
};

/////////////////////// Op register marco /////////////////////////

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_CUSTOM_OPERATOR(op_type, ...)                         \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_op__##op_type,                                             \
      "REGISTER_CUSTOM_OPERATOR must be called in global namespace."); \
  static ::paddle::CustomOperatorRegistrar<__VA_ARGS__>                \
      __custom_op_registrar_##op_type##__(#op_type);                   \
  int TouchOpRegistrar_##op_type() {                                   \
    __custom_op_registrar_##op_type##__.Touch();                       \
    return 0;                                                          \
  }

}  // namespace paddle
