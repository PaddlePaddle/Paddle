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
#include <sstream>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/any.hpp>

#include "paddle/fluid/extension/include/tensor.h"

/**
 * Op Meta Info Related Define.
 *
 * Used to maintain operator core information.
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

/// If a variable's name has a certain suffix, it means that the
/// variable is the gradient of another variable.
/// e.g. Variable "x@GRAD" is the gradient of variable "x".
constexpr char kGradVarSuffix[] = "@GRAD";
constexpr size_t kGradVarSuffixSize = 5U;

inline std::string Grad(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

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

  // TODO(chenweihang): add support for attribute input
  // int attribute input (not used now)
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

/////////////// InferDataType Function (PD_INFER_DTYPE) ///////////////

// Record Op Infer dtype core function
using InferDtypeFunc =
    std::vector<DataType> (*)(std::vector<DataType> input_dtypes);

template <typename F, F f>
struct InferDtypeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferDtypeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferDtype(std::vector<DataType> input_dtypes) {
    return InferDtypeCallHelper<Args..., TypeTag<int>>::template InferDtype<0>(
        input_dtypes);
  }

 private:
  template <typename... RemainingArgs>
  struct InferDtypeCallHelper;

  // Only one type input now: DataType
  template <typename... Tail>
  struct InferDtypeCallHelper<DataType, Tail...> {
    template <int in_idx, typename... PreviousArgs>
    static Return InferDtype(std::vector<DataType> input_dtypes,
                             const PreviousArgs&... pargs) {
      DataType arg = input_dtypes[in_idx];
      return InferDtypeCallHelper<Tail...>::template InferDtype<in_idx + 1>(
          input_dtypes, pargs..., arg);
    }
  };

  // end: base template
  template <typename T>
  struct InferDtypeCallHelper<TypeTag<T>> {
    template <int in_idx>
    static Return InferDtype(std::vector<DataType> input_dtypes,
                             const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_DTYPE(...) \
  ::paddle::InferDtypeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferDtype

////////////////////// Op Meta Info //////////////////////

class OpMetaInfo {
 public:
  explicit OpMetaInfo(const std::string& op_name) { name_ = op_name; }
  OpMetaInfo& Inputs(std::vector<std::string>&& inputs) {
    inputs_ = inputs;
    return *this;
  }
  OpMetaInfo& Outputs(std::vector<std::string>&& outputs) {
    outputs_ = outputs;
    return *this;
  }
  OpMetaInfo& SetKernelFn(KernelFunc&& func) {
    kernel_fn_ = func;
    return *this;
  }
  OpMetaInfo& SetInferShapeFn(InferShapeFunc&& func) {
    infer_shape_fn_ = func;
    return *this;
  }
  OpMetaInfo& SetInferDtypeFn(InferDtypeFunc&& func) {
    infer_dtype_fn_ = func;
    return *this;
  }

  // Maybe need private
 public:
  const std::string& GetOpName() const { return name_; }
  const std::vector<std::string>& GetInputs() const { return inputs_; }
  const std::vector<std::string>& GetOutputs() const { return outputs_; }
  const std::vector<std::string>& GetAttrs() const { return attrs_; }
  const KernelFunc& GetKernelFn() const { return kernel_fn_; }
  const InferShapeFunc& GetInferShapeFn() const { return infer_shape_fn_; }
  const InferDtypeFunc& GetInferDtypeFn() const { return infer_dtype_fn_; }

 private:
  // 1. desc info
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;

  // 2. func info
  KernelFunc kernel_fn_;
  InferShapeFunc infer_shape_fn_;
  InferDtypeFunc infer_dtype_fn_;
};

class OpMetaInfoMap {
 public:
  static OpMetaInfoMap& Instance() {
    static OpMetaInfoMap g_custom_op_meta_info_map;
    return g_custom_op_meta_info_map;
  }

  std::vector<OpMetaInfo>& operator[](const std::string& name) {
    return map_[name];
  }

  const std::unordered_map<std::string, std::vector<OpMetaInfo>>& GetMap()
      const {
    return map_;
  }

 private:
  OpMetaInfoMap() = default;

  std::unordered_map<std::string, std::vector<OpMetaInfo>> map_;

  DISABLE_COPY_AND_ASSIGN(OpMetaInfoMap);
};

class OpMetaInfoBuilder {
 public:
  explicit OpMetaInfoBuilder(std::string&& name)
      : name_(std::forward<std::string>(name)) {
    auto& info_vector = OpMetaInfoMap::Instance()[name_];
    auto op_meta = OpMetaInfo(name_);
    info_vector.emplace_back(op_meta);
    info_ptr_ = &(info_vector.back());
  }
  OpMetaInfoBuilder& Inputs(std::vector<std::string>&& inputs) {
    info_ptr_->Inputs(std::forward<std::vector<std::string>>(inputs));
    return *this;
  }
  OpMetaInfoBuilder& Outputs(std::vector<std::string>&& outputs) {
    info_ptr_->Outputs(std::forward<std::vector<std::string>>(outputs));
    return *this;
  }
  OpMetaInfoBuilder& SetKernelFn(KernelFunc&& func) {
    info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
    return *this;
  }
  OpMetaInfoBuilder& SetInferShapeFn(InferShapeFunc&& func) {
    info_ptr_->SetInferShapeFn(std::forward<InferShapeFunc>(func));
    return *this;
  }
  OpMetaInfoBuilder& SetInferDtypeFn(InferDtypeFunc&& func) {
    info_ptr_->SetInferDtypeFn(std::forward<InferDtypeFunc>(func));
    return *this;
  }
  OpMetaInfoBuilder& SetBackwardOp(std::string&& bwd_op_name) {
    auto& info_vector = OpMetaInfoMap::Instance()[name_];
    auto op_meta = OpMetaInfo(std::forward<std::string>(bwd_op_name));
    info_vector.emplace_back(op_meta);
    info_ptr_ = &(info_vector.back());
    return *this;
  }

 private:
  std::string name_;
  // Point to the currently constructed op meta info
  OpMetaInfo* info_ptr_;
};

/////////////////////// Op register marco /////////////////////////

#define PD_BUILD_OPERATOR(op_name)                                      \
  static ::paddle::OpMetaInfoBuilder __op_meta_info_##__COUNTER__##__ = \
      ::paddle::OpMetaInfoBuilder(op_name)

}  // namespace paddle
