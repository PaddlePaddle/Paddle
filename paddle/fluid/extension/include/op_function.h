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
#include <sstream>
#include <boost/any.hpp>
#include "paddle/fluid/extension/include/tensor.h"
#include "paddle/fluid/extension/include/device.h"

/**
 * Op Function Related Define.
 *
 * Used to maintain operator core information independent of the framework.
 *
 */

namespace paddle {

using Tensor = paddle::Tensor;

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

//////////////////// Kernel Function traits (PD_TRAITS) ///////////////////

// Record forward function traits
using FuncInfo = std::pair<size_t, size_t>;
using TraitsFunc = FuncInfo (*)();

template <typename T, typename Enabled = void>
struct KernelFuncTraits;

template <typename Return, typename... Args>
struct KernelFuncTraits<Return (*)(Args...)> {
  static FuncInfo GetFuncInfo() {
    // TODO(chenweihang): parse tensor args num & attribute num
    // Now only Tensor is input by default
    return std::make_pair(Arity, 0);
  }

 private:
  using ReturnType = Return;
  using ArgsTuple = std::tuple<Args...>;

  enum : std::size_t { Arity = sizeof...(Args) };

  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, ArgsTuple>::type;
};

#define OP_INFO(...) \
  ::paddle::KernelFuncTraits<decltype(&__VA_ARGS__)>::GetFuncInfo

////////////////////// Kernel Function (PD_KERNEL) ////////////////////////

// Record Op kernel core function
using KernelFunc = std::vector<Tensor> (*)(std::vector<Tensor> inputs,
                                           std::vector<boost::any> attrs);

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct KernelFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelFuncImpl<Return (*)(Args...), impl_fn> {
  static Return Compute(std::vector<Tensor> inputs,
                        std::vector<boost::any> attrs) {
    return ComputeCallHelper<Args..., TypeTag<int>>::template Compute<0, 0>(
        inputs, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct ComputeCallHelper;

  // for Tensor input
  template <typename... Tail>
  struct ComputeCallHelper<const Tensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "Input tensor should appear before attributes.");
      const Tensor& arg = inputs[in_idx];
      return ComputeCallHelper<Tail...>::template Compute<in_idx + 1, attr_idx>(
          inputs, attrs, pargs..., arg);
    }
  };

  // for int attribute input (not used now)
  template <typename... Tail>
  struct ComputeCallHelper<int, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      try {
        int arg = boost::any_cast<int>(attrs[attr_idx]);
        return ComputeCallHelper<Tail...>::template Compute<in_idx,
                                                            attr_idx + 1>(
            inputs, attrs, pargs..., arg);
      } catch (boost::bad_any_cast&) {
        throw std::runtime_error(
            "Attribute cast error in custom operator. Expected int value.");
      }
    }
  };

  // end: base template
  template <typename T>
  struct ComputeCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs, const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_KERNEL(...) \
  ::paddle::KernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

/////////////// InferShape Function (PD_INFER_SHAPE) ///////////////

// Record Op infershape core function
using InferShapeFunc = std::vector<std::vector<int64_t>> (*)(
    std::vector<std::vector<int64_t>> input_shapes);

template <typename F, F f>
struct InferShapeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferShapeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferShape(std::vector<std::vector<int64_t>> input_shapes) {
    return InferShapeCallHelper<Args..., TypeTag<int>>::template InferShape<0>(
        input_shapes);
  }

 private:
  template <typename... RemainingArgs>
  struct InferShapeCallHelper;

  // only one type input: std::vector<int64_t>
  template <typename... Tail>
  struct InferShapeCallHelper<std::vector<int64_t>, Tail...> {
    template <int in_idx, typename... PreviousArgs>
    static Return InferShape(std::vector<std::vector<int64_t>> input_shapes,
                             const PreviousArgs&... pargs) {
      std::vector<int64_t> arg = input_shapes[in_idx];
      return InferShapeCallHelper<Tail...>::template InferShape<in_idx + 1>(
          input_shapes, pargs..., arg);
    }
  };

  // end: base template
  template <typename T>
  struct InferShapeCallHelper<TypeTag<T>> {
    template <int in_idx>
    static Return InferShape(std::vector<std::vector<int64_t>> input_shapes,
                             const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_SHAPE(...) \
  ::paddle::InferShapeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferShape

////////////////////// Op Execution Function //////////////////////

class OpFunction {
 public:
  OpFunction() = default;

  void SetForwardFunc(KernelFunc&& forward_func) {
    forward_func_ = forward_func;
  }
  const KernelFunc& GetForwardFunc() const { return forward_func_; }

  void SetBackwardFunc(KernelFunc&& backward_func) {
    backward_func_ = backward_func;
  }
  const KernelFunc& GetBackwardFunc() const { return backward_func_; }

  void SetInferShapeFunc(InferShapeFunc&& infer_shape_func) {
    infer_shape_func_ = infer_shape_func;
  }
  const InferShapeFunc& GetInferShapeFunc() const { return infer_shape_func_; }

  void SetNumTensorArgs(size_t num) { num_tensor_args_ = num; }
  size_t GetNumTensorArgs() const { return num_tensor_args_; }

  void SetNumAttributes(size_t num) { num_attributes_ = num; }
  size_t GetNumAttributes() const { return num_attributes_; }

 private:
  // 1. func member
  KernelFunc forward_func_;
  KernelFunc backward_func_;
  InferShapeFunc infer_shape_func_;

  // 2. func traits
  size_t num_tensor_args_;
  size_t num_attributes_;
};

//////////////////////// Op Execution Function Map //////////////////////////

class OpFunctionMap {
 public:
  static OpFunctionMap& Instance() {
    static OpFunctionMap g_custom_op_function_holder;
    return g_custom_op_function_holder;
  }

  void Insert(const std::string& op_type, const OpFunction& op_func) {
    if (map_.find(op_type) != map_.end()) {
      throw std::runtime_error("Operator `" + op_type +
                               "` has been registered.");
    }
    map_.insert({op_type, op_func});
  }

  const std::unordered_map<std::string, OpFunction>& GetMap() const {
    return map_;
  }

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
  OperatorFunctionRegistrar(const char* op_type, TraitsFunc&& traits_func,
                            KernelFunc&& forward_func,
                            KernelFunc&& backward_func,
                            InferShapeFunc&& infer_shape_func) {
    OpFunction op_func;
    FuncInfo func_info = traits_func();

    op_func.SetNumTensorArgs(func_info.first);
    op_func.SetNumAttributes(func_info.second);

    op_func.SetForwardFunc(std::forward<KernelFunc>(forward_func));
    op_func.SetBackwardFunc(std::forward<KernelFunc>(backward_func));
    op_func.SetInferShapeFunc(std::forward<InferShapeFunc>(infer_shape_func));

    OpFunctionMap::Instance().Insert(op_type, op_func);
  }
};

/////////////////////// Op register marco /////////////////////////

#define BUILD_OPERATOR(op_type, traits_func, forward_func, backward_func,      \
                       infer_shape_func)                                       \
  static ::paddle::OperatorFunctionRegistrar                                   \
      __operator_function_registrar_##op_type##__(#op_type, traits_func,       \
                                                  forward_func, backward_func, \
                                                  infer_shape_func);           \
  int TouchOperatorFunctionRegistrar_##op_type() {                             \
    __operator_function_registrar_##op_type##__.Touch();                       \
    return 0;                                                                  \
  }

}  // namespace paddle
