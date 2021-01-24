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

#include "paddle/fluid/extension/include/utils.h"
#include "paddle/fluid/framework/tensor.h"

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

namespace paddle {

using Tensor = framework::Tensor;

using OpFunction = boost::variant<
    std::function<std::vector<Tensor>(const Tensor&)>,
    std::function<std::vector<Tensor>(const Tensor&, const Tensor&)>,
    std::function<std::vector<Tensor>(const Tensor&, const Tensor&,
                                      const Tensor&)>,
    std::function<std::vector<Tensor>(const Tensor&, const Tensor&,
                                      const Tensor&, const Tensor&)>>;

class OpFunctionMap {
 public:
  template <typename func_t>
  OpFunctionMap& AppendFunc(func_t&& func) {
    using traits = detail::function_traits<func_t>;
    using function_t = typename traits::Function;
    funcs_[typeid(function_t)] =
        static_cast<OpFunction>(std::forward<func_t>(func));
    return *this;
  }

  std::unordered_map<std::type_index, OpFunction>* mutable_funcs() {
    return &funcs_;
  }

 private:
  std::unordered_map<std::type_index, OpFunction> funcs_;
};

class OpFunctionHolder {
 public:
  static OpFunctionHolder& Instance() {
    static OpFunctionHolder g_custom_op_function_holder;
    return g_custom_op_function_holder;
  }

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& type, const OpFunctionMap& funcs) {
    PADDLE_ENFORCE_NE(Has(type), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", type));
    map_.insert({type, funcs});
  }

  const OpFunctionMap& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
    PADDLE_ENFORCE_NOT_NULL(
        op_info_ptr,
        platform::errors::NotFound("Operator (%s) is not registered.", type));
    return *op_info_ptr;
  }

  const OpFunctionMap* GetNullable(const std::string& type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  std::unordered_map<std::string, OpFunctionMap>* mutable_map() {
    return &map_;
  }

 private:
  OpFunctionHolder() = default;
  DISABLE_COPY_AND_ASSIGN(OpFunctionHolder);

 private:
  std::unordered_map<std::string, OpFunctionMap> map_;
};

namespace detail {

template <bool at_end, size_t I, typename... FunctorTypes>
class OpFuncRegistrarFunctor;

template <size_t I, typename... FunctorTypes>
struct OpFuncRegistrarFunctor<false, I, FunctorTypes...> {
  using FunctorType =
      typename std::tuple_element<I, std::tuple<FunctorTypes...>>::type;
  void operator()(const char* op_type, OpFunctionMap* op_funcs) const {
    op_funcs->AppendFunc(FunctorType());
    constexpr auto size = std::tuple_size<std::tuple<FunctorTypes...>>::value;
    OpFuncRegistrarFunctor<I + 1 == size, I + 1, FunctorTypes...> func;
    func(op_type, op_funcs);
  }
};

template <size_t I, typename... FunctorTypes>
struct OpFuncRegistrarFunctor<true, I, FunctorTypes...> {
  void operator()(const char* op_type, OpFunctionMap* op_funcs) const {
    OpFunctionHolder::Instance().Insert(op_type, *op_funcs);
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
    OpFunctionMap op_funcs;
    detail::OpFuncRegistrarFunctor<false, 0, FunctorTypes...> func;
    func(op_type, &op_funcs);
  }
};

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_CUSTOM_OPERATOR(op_type, ...)                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                     \
      __reg_op__##op_type,                                            \
      "REGISTER_CUSTOM_OPERATOR must be called in global namespace"); \
  static ::paddle::CustomOperatorRegistrar<__VA_ARGS__>               \
      __custom_op_registrar_##op_type##__(#op_type);                  \
  int TouchOpRegistrar_##op_type() {                                  \
    __custom_op_registrar_##op_type##__.Touch();                      \
    return 0;                                                         \
  }
}  // namespace paddle
