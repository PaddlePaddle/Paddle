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
#include "paddle/fluid/platform/place.h"

namespace paddle {

using Tensor = framework::Tensor;

using FuncInfo = std::pair<size_t, size_t>;
using TraitsFunc = FuncInfo (*)();
using ComputeFunc = std::vector<Tensor> (*)(std::vector<const Tensor*> inputs,
                                            std::vector<boost::any> attrs);
// key std::string means data type, replace by enum DataType later
using ComputeFuncMap = std::unordered_map<std::string, ComputeFunc>;

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

//////////////////// Compute Function traits (PD_TRAITS) ///////////////////

template <typename T, typename Enabled = void>
struct ComputeFuncTraits;

template <typename Return, typename... Args>
struct ComputeFuncTraits<Return (*)(Args...)> {
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

////////////////////// Compute Function (PD_KERNEL) ////////////////////////

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct ComputeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct ComputeFuncImpl<Return (*)(Args...), impl_fn> {
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

////////////////////// Kernel Execution Function //////////////////

class KernelExecFunction {
 public:
  KernelExecFunction() = delete;
  KernelExecFunction(const platform::Place& place, const std::string& lib_type,
                     const ComputeFuncMap& func_map)
      : place_(place), lib_type_(lib_type), func_map_(func_map) {}

  const platform::Place& GetPlace() const { return place_; }

  const std::string& GetLibType() const { return lib_type_; }

  const ComputeFuncMap GetComputeFuncMap() const { return func_map_; }

 private:
  // replace by enum Place later
  platform::Place place_;

  // replace by enum LibraryType later
  std::string lib_type_;

  // compute function
  ComputeFuncMap func_map_;
};

////////////////////// Op Execution Function //////////////////////

class OpExecFunction {
 public:
  OpExecFunction() = default;

  void AppendForwardFunc(const KernelExecFunction& ffs) {
    forward_funcs_.emplace_back(ffs);
  }
  const std::vector<KernelExecFunction>& GetForwardFuncList() const {
    return forward_funcs_;
  }

  void AppendBackwardFunc(const KernelExecFunction& bfs) {
    backward_funcs_.emplace_back(bfs);
  }
  const std::vector<KernelExecFunction>& GetBackwardFuncList() const {
    return backward_funcs_;
  }

  void SetNumTensorArgs(size_t num) { num_tensor_args_ = num; }
  size_t GetNumTensorArgs() const { return num_tensor_args_; }

  void SetNumAttributes(size_t num) { num_attributes_ = num; }
  size_t GetNumAttributes() const { return num_attributes_; }

 private:
  // 1. func member
  std::vector<KernelExecFunction> forward_funcs_;
  std::vector<KernelExecFunction> backward_funcs_;
  // support infershape func later

  // 2. func traits
  size_t num_tensor_args_;
  size_t num_attributes_;
};

//////////////////////// Op Execution Function Map //////////////////////////

class OpExecFunctionMap {
 public:
  static OpExecFunctionMap& Instance() {
    static OpExecFunctionMap g_custom_op_function_holder;
    return g_custom_op_function_holder;
  }

  void Insert(const std::string& op_type, const OpExecFunction& op_func) {
    PADDLE_ENFORCE_NE(map_.find(op_type) != map_.end(), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", op_type));
    map_.insert({op_type, op_func});
  }

  void InsertForwardKernelFunc(const std::string& op_type,
                               const KernelExecFunction& forward_func_) {
    auto it = map_.find(op_type);
    if (it == map_.end()) {
      throw std::runtime_error("op not exists");
    } else {
      it->second.AppendForwardFunc(forward_func_);
    }
  }

  void InsertBackwardKernelFunc(const std::string& op_type,
                                const KernelExecFunction& backward_func_) {
    auto it = map_.find(op_type);
    if (it == map_.end()) {
      throw std::runtime_error("op not exists");
    } else {
      it->second.AppendBackwardFunc(backward_func_);
    }
  }

  const std::unordered_map<std::string, OpExecFunction>& GetMap() {
    return map_;
  }

 private:
  OpExecFunctionMap() = default;

  std::unordered_map<std::string, OpExecFunction> map_;

  DISABLE_COPY_AND_ASSIGN(OpExecFunctionMap);
};

///////////////// Op Function Registrar ////////////////////////////

class Registrar {
 public:
  void Touch() {}
};

struct OperatorFunctionRegistrar : public Registrar {
  OperatorFunctionRegistrar(const char* op_type, TraitsFunc traits_func) {
    OpExecFunction op_func;
    FuncInfo func_info = traits_func();
    op_func.SetNumTensorArgs(func_info.first);
    op_func.SetNumAttributes(func_info.second);
    OpExecFunctionMap::Instance().Insert(op_type, op_func);
  }
};

template <typename Func, Func... funcs>
struct OpKernelFuncRegistrar : public Registrar {
  OpKernelFuncRegistrar(const char* op_type, bool is_forward,
                        const char* lib_type, platform::Place place,
                        const char* func_names) {
    // 1. build ComputeFuncMap
    ComputeFuncMap compute_func_map;
    std::vector<ComputeFunc> compute_funcs{{funcs...}};
    auto dtype_strs = ParseKernelDataTypeFromDeclaration(func_names);
    if (compute_funcs.size() != dtype_strs.size()) {
      throw std::runtime_error("kernel function number error.");
    }
    // use default dtype "float"
    for (size_t i = 0; i < compute_funcs.size(); ++i) {
      compute_func_map[dtype_strs[i]] = compute_funcs[i];
    }

    // 2. build KernelExecFunction
    auto kernel_func = KernelExecFunction(place, lib_type, compute_func_map);

    // 3. insert
    if (is_forward) {
      OpExecFunctionMap::Instance().InsertForwardKernelFunc(op_type,
                                                            kernel_func);
    } else {
      OpExecFunctionMap::Instance().InsertBackwardKernelFunc(op_type,
                                                             kernel_func);
    }
  }

 private:
  // only for demo test, polish later
  std::vector<std::string> ParseKernelDataTypeFromDeclaration(
      const std::string& func_names) {
    std::vector<std::string> res;
    // add others later
    std::vector<std::string> dtypes{"<float>", "<double>", "<int>",
                                    "<int64_t>"};
    for (auto& dtype : dtypes) {
      if (func_names.find(dtype) != std::string::npos) {
        VLOG(0) << "contains data type: " << dtype;
        res.emplace_back(dtype.substr(1, dtype.size() - 2));
      }
    }
    return res;
  }
};

/////////////////////// Op register marco /////////////////////////

#define PD_KERNEL(...) \
  ::paddle::ComputeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define OP_INFO(...) \
  ::paddle::ComputeFuncTraits<decltype(&__VA_ARGS__)>::GetFuncInfo

#define ADD_OPERATOR(op_type, traits_func)                                \
  static ::paddle::OperatorFunctionRegistrar                              \
      __operator_function_registrar_##op_type##__(#op_type, traits_func); \
  int TouchOperatorFunctionRegistrar_##op_type() {                        \
    __operator_function_registrar_##op_type##__.Touch();                  \
    return 0;                                                             \
  }

#define ADD_KERNEL(op_type, is_forward, lib_type, place, ...)                \
  static ::paddle::OpKernelFuncRegistrar<::paddle::ComputeFunc, __VA_ARGS__> \
      __op_kernel_func_registrar_##op_type##_##is_forward##_##lib_type##__(  \
          #op_type, is_forward, #lib_type, place, #__VA_ARGS__);             \
  int TouchOpKernelFuncRegistrar_##op_type_##is_forward##_##lib_type##__() { \
    __op_kernel_func_registrar_##op_type##_##is_forward##_##lib_type##__     \
        .Touch();                                                            \
    return 0;                                                                \
  }

#define ADD_FORWARD_CPU_KERNEL(op_type, ...) \
  ADD_KERNEL(op_type, true, CPU, ::paddle::platform::CPUPlace(), __VA_ARGS__)

#define ADD_BACKWARD_CPU_KERNEL(op_type, ...) \
  ADD_KERNEL(op_type, false, CPU, ::paddle::platform::CPUPlace(), __VA_ARGS__)

#define ADD_FORWARD_CUDA_KERNEL(op_type, ...) \
  ADD_KERNEL(op_type, true, CUDA, ::paddle::platform::CUDAPlace(), __VA_ARGS__)

#define ADD_BACKWARD_CUDA_KERNEL(op_type, ...) \
  ADD_KERNEL(op_type, false, CUDA, ::paddle::platform::CUDAPlace(), __VA_ARGS__)

}  // namespace paddle
