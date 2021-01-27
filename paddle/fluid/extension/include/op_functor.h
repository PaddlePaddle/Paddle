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

#include "paddle/fluid/framework/tensor.h"

namespace paddle {

using Tensor = framework::Tensor;

using FuncInfo = std::pair<size_t, size_t>;
using TraitsFunc = FuncInfo (*)();
using KernelFunc = std::vector<Tensor> (*)(std::vector<const Tensor*> inputs,
                                           std::vector<boost::any> attrs);
using KernelFuncList = std::vector<KernelFunc>;

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

//////////////////// Kernel Function traits (PD_TRAITS) ///////////////////

template <typename T, typename Enabled = void>
struct KernelFuncTraits;

template <typename Return, typename... Args>
struct KernelFuncTraits<Return (*)(Args...)> {
  static FuncInfo GetFuncInfo() {
    // TODO(chenweihang): parse tensor args num & attribute num
    return std::make_pair(Arity, 0);
  }

 private:
  using ReturnType = Return;
  using ArgsTuple = std::tuple<Args...>;

  enum : std::size_t { Arity = sizeof...(Args) };

  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, ArgsTuple>::type;
};

////////////////////// Kernel Function (PD_KERNEL) ////////////////////////

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct KernelFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelFuncImpl<Return (*)(Args...), impl_fn> {
  static Return Compute(std::vector<const Tensor*> inputs,
                        std::vector<boost::any> attrs) {
    return ComputeCallHelper<Args..., TypeTag<int>>::template Compute<0, 0>(
        inputs, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct ComputeCallHelper;

  template <typename... Tail>
  struct ComputeCallHelper<const Tensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<const Tensor*> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "Input tensor should appear before attributes.");
      const Tensor& arg = *(inputs[in_idx]);
      return ComputeCallHelper<Tail...>::template Compute<in_idx + 1, attr_idx>(
          inputs, attrs, pargs..., arg);
    }
  };

  template <typename... Tail>
  struct ComputeCallHelper<int, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<const Tensor*> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      try {
        int arg = boost::any_cast<int>(attrs[attr_idx]);
        return ComputeCallHelper<Tail...>::template Compute<in_idx,
                                                            attr_idx + 1>(
            inputs, attrs, pargs..., arg);
      } catch (boost::bad_any_cast&) {
        std::ostringstream err;
        err << "Attribute cast error in custom operator. Expected int value.";
        throw std::runtime_error(err.str());
      }
    }
  };

  template <typename T>
  struct ComputeCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<const Tensor*> inputs,
                          std::vector<boost::any> attrs, const Args&... args) {
      return impl_fn(args...);
    }
  };
};

////////////////////// Op Function //////////////////////

class OpFunction {
 public:
  OpFunction() = default;

  void SetForwardFuncs(KernelFuncList ffs) { forward_funcs_ = ffs; }
  const KernelFuncList& GetForwardFuncs() const { return forward_funcs_; }

  void SetBackwardFuncs(KernelFuncList bfs) { backward_funcs_ = bfs; }
  const KernelFuncList& GetBackwardFuncs() const { return backward_funcs_; }

  void SetNumTensorArgs(size_t num) { num_tensor_args_ = num; }
  size_t GetNumTensorArgs() const { return num_tensor_args_; }

  void SetNumAttributes(size_t num) { num_attributes_ = num; }
  size_t GetNumAttributes() const { return num_attributes_; }

 private:
  // 1. func member
  KernelFuncList forward_funcs_;
  KernelFuncList backward_funcs_;
  // support infershape func later

  // 2. func traits
  size_t num_tensor_args_;
  size_t num_attributes_;
};

//////////////////////// OpFunction Map //////////////////////////

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

  const std::unordered_map<std::string, OpFunction>& GetMap() { return map_; }

 private:
  OpFunctionMap() = default;

  std::unordered_map<std::string, OpFunction> map_;

  DISABLE_COPY_AND_ASSIGN(OpFunctionMap);
};

///////////////// Op Function Registrar ////////////////////////////

class Registrar {
 public:
  void Touch() {}
};

struct OperatorFunctionRegistrar : public Registrar {
  OperatorFunctionRegistrar(const char* op_type, TraitsFunc traits_func,
                            KernelFuncList forward_funcs,
                            KernelFuncList backward_funcs) {
    OpFunction op_func;
    FuncInfo func_info = traits_func();
    op_func.SetNumTensorArgs(func_info.first);
    op_func.SetNumAttributes(func_info.second);
    op_func.SetForwardFuncs(forward_funcs);
    op_func.SetBackwardFuncs(backward_funcs);
    OpFunctionMap::Instance().Insert(op_type, op_func);
  }
};

/////////////////////// Op register marco /////////////////////////

#define PD_KERNEL(...) \
  ::paddle::KernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define PD_TRAITS(...) \
  ::paddle::KernelFuncTraits<decltype(&__VA_ARGS__)>::GetFuncInfo

#define REGISTER_OP_FUNCTION(op_type, traits_func, forward_funcs, \
                             backward_funcs)                      \
  static ::paddle::OperatorFunctionRegistrar                      \
      __operator_function_registrar_##op_type##__(                \
          #op_type, traits_func, forward_funcs, backward_funcs);  \
  int TouchOpRegistrar_##op_type() {                              \
    __operator_function_registrar_##op_type##__.Touch();          \
    return 0;                                                     \
  }

}  // namespace paddle
