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

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/common/exception.h"
#include "paddle/common/macros.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

#ifdef PADDLE_WITH_TENSORRT
#include "NvInfer.h"
#endif

/**
 * Op Meta Info Related Define.
 *
 * Used to maintain operator core information.
 *
 */

namespace paddle {

class PADDLE_API OpMetaInfoHelper;
using Tensor = paddle::Tensor;

///////////////// Util Marco Define ////////////////

#define PD_DISABLE_COPY_AND_ASSIGN(classname)      \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

///////////////// Util Define and Function ////////////////

constexpr char kGradTensorSuffix[] = "@GRAD";
constexpr char kTensorVectorSuffix[] = "@VECTOR";
constexpr char kDoubleGradNewOutSuffix[] = "@NEW";
constexpr char kOptionalSuffix[] = "@OPTIONAL";

// Used for Construct Grad Tensor name
inline std::string Grad(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 5U);
  result += t_name;
  result += kGradTensorSuffix;
  return result;
}

// Used for Construct std::vector<Tensor> name
inline std::string Vec(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 7U);
  result += t_name;
  result += kTensorVectorSuffix;
  return result;
}

// Used for Construct double grad output name
inline std::string New(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 4U);
  result += t_name;
  result += kDoubleGradNewOutSuffix;
  return result;
}

// Used for Construct paddle::optional name
inline std::string Optional(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 9U);
  result += t_name;
  result += kOptionalSuffix;
  return result;
}

std::vector<std::string> ParseAttrStr(const std::string& attr);

PADDLE_API void AssignTensorImpl(const Tensor& src, Tensor* dst);

////////////////////// Kernel Context ////////////////////////

class PADDLE_API CustomOpKernelContext {
 public:
  CustomOpKernelContext() = default;

  void EmplaceBackInput(Tensor&& input);
  void EmplaceBackInput(const Tensor& input);
  void EmplaceBackInputs(const std::vector<Tensor>& inputs);
  void EmplaceBackOutput(Tensor&& output);
  void EmplaceBackOutputs(const std::vector<Tensor>& outputs);
  void EmplaceBackAttr(paddle::any attr);
  void EmplaceBackAttrs(const std::vector<paddle::any>& attrs);
  const std::pair<size_t, size_t>& InputRangeAt(size_t idx) const;
  const std::pair<size_t, size_t>& OutputRangeAt(size_t idx) const;

  const Tensor& InputAt(size_t idx) const;
  std::vector<Tensor> InputsBetween(size_t start, size_t end) const;
  Tensor& MutableInputAt(size_t idx);
  std::vector<Tensor>* AllMutableInput();
  paddle::optional<Tensor> OptionalInputAt(size_t idx) const;
  paddle::optional<std::vector<Tensor>> OptionalInputsBetween(size_t start,
                                                              size_t end) const;
  const std::vector<paddle::any>& Attrs() const;
  const std::vector<std::pair<size_t, size_t>>& InputRange() const;
  const std::vector<std::pair<size_t, size_t>>& OutputRange() const;
  Tensor* MutableOutputAt(size_t idx);
  std::vector<Tensor*> MutableOutputBetween(size_t start, size_t end);
  std::vector<Tensor> OutputsBetween(size_t start, size_t end) const;
  std::vector<Tensor>* AllMutableOutput();

  template <typename AttrType>
  AttrType AttrAt(size_t idx) const {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PD_THROW("Attribute cast error in Custom Op Kernel Context.");
    }
  }

  // handle inplace map
  void ConstructInplaceIndex(
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      const std::unordered_map<std::string, std::string>& inplace_map);
  void UpdatePlainOutputs(
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      const std::unordered_map<std::string, std::string>& inplace_map);
  void AssignInplaceOutputs();
  std::vector<Tensor*>* AllMutablePlainOutput();
  std::unordered_map<size_t, size_t> GetInplaceIndexMap() const;
  std::unordered_map<size_t, size_t> GetInplaceReverseIndexMap() const;

 private:
  // TODO(chenweihang): replaced be SmallVector
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<paddle::any> attrs_;
  // handle inplace map
  std::vector<Tensor*> plain_outputs_;
  // {input: output}
  std::unordered_map<size_t, size_t> inplace_idx_map_;
  // {output: input}
  std::unordered_map<size_t, size_t> inplace_reverse_idx_map_;

  std::vector<std::pair<size_t, size_t>> input_range_;
  std::vector<std::pair<size_t, size_t>> output_range_;
};

////////////////////// Kernel Function (PD_KERNEL) ////////////////////////

// Record Op kernel core function
using KernelFunc = void (*)(CustomOpKernelContext*);

#define PD_SPECIALIZE_ComputeCallHelper(attr_type)                             \
  template <typename... Tail>                                                  \
  struct ComputeCallHelper<attr_type, Tail...> {                               \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {  \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                        \
      ComputeCallHelper<                                                       \
          Tail...>::template Compute<in_idx, attr_idx + 1, out_idx>(ctx,       \
                                                                    pargs...,  \
                                                                    arg);      \
    }                                                                          \
  }

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct KernelFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelFuncImpl<Return (*)(Args...), impl_fn> {
  static void Compute(CustomOpKernelContext* ctx) {
    ComputeCallHelper<Args..., TypeTag<int>>::template Compute<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct ComputeCallHelper;

  // Handle args for general Tensor input case
  template <typename... Tail>
  struct ComputeCallHelper<const Tensor&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto& arg = ctx->MutableInputAt(range.first);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Handle args for optional Tensor input case
  template <typename... Tail>
  struct ComputeCallHelper<const paddle::optional<paddle::Tensor>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto& arg = ctx->InputAt(range.first);
      if (!arg.is_initialized()) {
        ComputeCallHelper<Tail...>::
            template Compute<in_idx + 1, attr_idx, out_idx>(
                ctx, pargs..., paddle::none);
      } else {
        ComputeCallHelper<
            Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                      pargs...,
                                                                      arg);
      }
    }
  };

  // Handle args for inplace Tensor case
  template <typename... Tail>
  struct ComputeCallHelper<Tensor&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto& arg = ctx->MutableInputAt(range.first);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Handle args for optional inplace Tensor input case
  template <typename... Tail>
  struct ComputeCallHelper<paddle::optional<paddle::Tensor>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto arg = ctx->OptionalInputAt(range.first);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Handle args for general vector<Tensor> input case
  template <typename... Tail>
  struct ComputeCallHelper<const std::vector<Tensor>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto arg = ctx->InputsBetween(range.first, range.second);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Handle args for optional vector<Tensor> input case
  template <typename... Tail>
  struct ComputeCallHelper<const paddle::optional<std::vector<paddle::Tensor>>&,
                           Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto arg = ctx->InputsBetween(range.first, range.second);
      if (arg.empty() || !arg[0].is_initialized()) {
        ComputeCallHelper<Tail...>::
            template Compute<in_idx + 1, attr_idx, out_idx>(
                ctx, pargs..., paddle::none);
      } else {
        ComputeCallHelper<
            Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                      pargs...,
                                                                      arg);
      }
    }
  };

  // Handle args for inplace vector<Tensor> case
  template <typename... Tail>
  struct ComputeCallHelper<std::vector<Tensor>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto arg = ctx->InputsBetween(range.first, range.second);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Handle args for optional inplace vector<Tensor> case
  template <typename... Tail>
  struct ComputeCallHelper<paddle::optional<std::vector<Tensor>>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->InputRangeAt(in_idx);
      auto arg = ctx->OptionalInputsBetween(range.first, range.second);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  PD_SPECIALIZE_ComputeCallHelper(bool);
  PD_SPECIALIZE_ComputeCallHelper(int);
  PD_SPECIALIZE_ComputeCallHelper(float);
  PD_SPECIALIZE_ComputeCallHelper(int64_t);
  PD_SPECIALIZE_ComputeCallHelper(const std::string&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<int>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<float>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<int64_t>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<std::string>&);
  // TODO(chenweihang): support other attribute type if needed.
  // Why not support other attribute type here?
  // - paddle::blank, std::vector<bool> and std::vector<double>
  //   are not used in op
  // - BlockDesc* and std::vector<BlockDesc*> are used in framework

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_ComputeCallHelper(const bool&);
  PD_SPECIALIZE_ComputeCallHelper(const int&);
  PD_SPECIALIZE_ComputeCallHelper(const float&);
  PD_SPECIALIZE_ComputeCallHelper(const int64_t&);

  // NOTE(chenweihang): Used to be compatible with the 2.1 released
  // interface, but not recommended
  PD_SPECIALIZE_ComputeCallHelper(std::string);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<int>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<float>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<int64_t>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<std::string>);

  // Used to be compatible with 2.3 released internal inplace interface, not
  // recommended
  // Handle args for compatible inplace case
  template <typename... Tail>
  struct ComputeCallHelper<Tensor*, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->OutputRangeAt(out_idx);
      auto* arg = ctx->MutableOutputAt(range.first);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx, attr_idx, out_idx + 1>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Used to be compatible with 2.3 released internal inplace interface, not
  // recommended
  // TODO(chenweihang): What is the appropriate output form?
  // std::vector<Tensor>*? or std::vector<Tensor*>? or std::vector<Tensor*>*
  // Handle args for compatible inplace case
  template <typename... Tail>
  struct ComputeCallHelper<std::vector<Tensor*>, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      auto& range = ctx->OutputRangeAt(out_idx);
      auto arg = ctx->MutableOutputBetween(range.first, range.second);
      ComputeCallHelper<
          Tail...>::template Compute<in_idx, attr_idx, out_idx + 1>(ctx,
                                                                    pargs...,
                                                                    arg);
    }
  };

  template <int out_idx, typename T>
  struct ComputeReturnHelper;

  // For compatibility with the original custom op form
  template <int out_idx>
  struct ComputeReturnHelper<out_idx, std::vector<Tensor>> {
    static void Compute(CustomOpKernelContext* ctx, Args&... args) {
      static_assert(out_idx == 0,
                    "If return std::vector<Tensor> in Custom OpKernel, "
                    "you cannot pass output by kernel function argument.");
      auto outs = impl_fn(args...);
      auto* orig_outs = ctx->AllMutablePlainOutput();
      PD_CHECK(orig_outs->size() == outs.size(),
               "The number of element in custom operator outputs is wrong, "
               "expected contains ",
               orig_outs->size(),
               " Tensors, but actually contains ",
               outs.size(),
               " Tensors.");
      for (size_t i = 0; i < outs.size(); ++i) {
        AssignTensorImpl(outs.at(i), orig_outs->at(i));
      }
    }
  };

  template <int out_idx>
  struct ComputeReturnHelper<out_idx, void> {
    static void Compute(CustomOpKernelContext* ctx, Args&... args) {
      impl_fn(args...);
    }
  };

  // end: base template
  template <typename T>
  struct ComputeCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Compute(CustomOpKernelContext* ctx, PreviousArgs&... pargs) {
      ComputeReturnHelper<out_idx, Return>::Compute(ctx, pargs...);
    }
  };
};

#define PD_KERNEL(...) \
  ::paddle::KernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

/////////////// InferShape Function (PD_INFER_SHAPE) ///////////////

// Record Op infershape core function
using InferShapeFunc = std::vector<std::vector<int64_t>> (*)(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
    const std::vector<paddle::any>& attrs);

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(input_type)     \
  template <typename... Tail>                                        \
  struct InferShapeCallHelper<input_type, Tail...> {                 \
    template <int in_idx,                                            \
              int vec_in_idx,                                        \
              int attr_idx,                                          \
              typename... PreviousArgs>                              \
    static Return InferShape(                                        \
        const std::vector<std::vector<int64_t>>& input_shapes,       \
        const std::vector<std::vector<std::vector<int64_t>>>&        \
            vec_input_shapes,                                        \
        const std::vector<paddle::any>& attrs,                       \
        const PreviousArgs&... pargs) {                              \
      input_type arg = input_shapes[in_idx];                         \
      return InferShapeCallHelper<Tail...>::                         \
          template InferShape<in_idx + 1, vec_in_idx, attr_idx>(     \
              input_shapes, vec_input_shapes, attrs, pargs..., arg); \
    }                                                                \
  }

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(input_type)    \
  template <typename... Tail>                                        \
  struct InferShapeCallHelper<input_type, Tail...> {                 \
    template <int in_idx,                                            \
              int vec_in_idx,                                        \
              int attr_idx,                                          \
              typename... PreviousArgs>                              \
    static Return InferShape(                                        \
        const std::vector<std::vector<int64_t>>& input_shapes,       \
        const std::vector<std::vector<std::vector<int64_t>>>&        \
            vec_input_shapes,                                        \
        const std::vector<paddle::any>& attrs,                       \
        const PreviousArgs&... pargs) {                              \
      input_type arg = vec_input_shapes[vec_in_idx];                 \
      return InferShapeCallHelper<Tail...>::                         \
          template InferShape<in_idx, vec_in_idx + 1, attr_idx>(     \
              input_shapes, vec_input_shapes, attrs, pargs..., arg); \
    }                                                                \
  }

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(attr_type)               \
  template <typename... Tail>                                                \
  struct InferShapeCallHelper<attr_type, Tail...> {                          \
    template <int in_idx,                                                    \
              int vec_in_idx,                                                \
              int attr_idx,                                                  \
              typename... PreviousArgs>                                      \
    static Return InferShape(                                                \
        const std::vector<std::vector<int64_t>>& input_shapes,               \
        const std::vector<std::vector<std::vector<int64_t>>>&                \
            vec_input_shapes,                                                \
        const std::vector<paddle::any>& attrs,                               \
        const PreviousArgs&... pargs) {                                      \
      try {                                                                  \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);        \
        return InferShapeCallHelper<Tail...>::                               \
            template InferShape<in_idx, vec_in_idx, attr_idx + 1>(           \
                input_shapes, vec_input_shapes, attrs, pargs..., arg);       \
      } catch (paddle::bad_any_cast&) {                                      \
        PD_THROW(                                                            \
            "Attribute cast error in custom operator InferShapeFn. "         \
            "Expected " #attr_type                                           \
            " value. InferShapeFn's attribute list must be exactly same as " \
            "Forward "                                                       \
            "KernelFn's attribute list except std::vector<int64_t> "         \
            "attribute.");                                                   \
      }                                                                      \
    }                                                                        \
  }

template <typename F, F f>
struct InferShapeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferShapeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferShape(
      const std::vector<std::vector<int64_t>>& input_shapes,
      const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
      const std::vector<paddle::any>& attrs) {
    return InferShapeCallHelper<Args..., TypeTag<int>>::
        template InferShape<0, 0, 0>(input_shapes, vec_input_shapes, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct InferShapeCallHelper;

  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(const std::vector<int64_t>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(
      const std::vector<std::vector<int64_t>>&);

  template <typename... Tail>
  struct InferShapeCallHelper<const paddle::optional<std::vector<int64_t>>&,
                              Tail...> {
    template <int in_idx,
              int vec_in_idx,
              int attr_idx,
              typename... PreviousArgs>
    static Return InferShape(
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
        const std::vector<paddle::any>& attrs,
        const PreviousArgs&... pargs) {
      const std::vector<int64_t>& arg = input_shapes[in_idx];
      if (arg.empty()) {
        return InferShapeCallHelper<Tail...>::
            template InferShape<in_idx + 1, vec_in_idx, attr_idx>(
                input_shapes, vec_input_shapes, attrs, pargs..., paddle::none);
      } else {
        return InferShapeCallHelper<Tail...>::
            template InferShape<in_idx + 1, vec_in_idx, attr_idx>(
                input_shapes, vec_input_shapes, attrs, pargs..., arg);
      }
    }
  };

  template <typename... Tail>
  struct InferShapeCallHelper<
      const paddle::optional<std::vector<std::vector<int64_t>>>&,
      Tail...> {
    template <int in_idx,
              int vec_in_idx,
              int attr_idx,
              typename... PreviousArgs>
    static Return InferShape(
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
        const std::vector<paddle::any>& attrs,
        const PreviousArgs&... pargs) {
      const std::vector<std::vector<int64_t>>& arg =
          vec_input_shapes[vec_in_idx];
      if (arg.empty()) {
        return InferShapeCallHelper<Tail...>::
            template InferShape<in_idx, vec_in_idx + 1, attr_idx>(
                input_shapes, vec_input_shapes, attrs, pargs..., paddle::none);
      } else {
        return InferShapeCallHelper<Tail...>::
            template InferShape<in_idx, vec_in_idx + 1, attr_idx>(
                input_shapes, vec_input_shapes, attrs, pargs..., arg);
      }
    }
  };

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(std::vector<int64_t>);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(
      std::vector<std::vector<int64_t>>);

  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(bool);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(int);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(float);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(int64_t);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::string&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<int>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<float>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<std::string>&);
  // NOTE(chenweihang): InferShape can't support std::vector<int64_t> attr type,
  // because the input type is std::vector<int64_t>, only can use one rule to
  // parse std::vector<int64_t> parameter

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const bool&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const int&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const float&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const int64_t&);

  // NOTE(chenweihang): Used to be compatible with the 2.1 released
  // interface, but not recommended
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(std::string);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(std::vector<int>);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(std::vector<float>);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(std::vector<std::string>);

  // end: base template
  template <typename T>
  struct InferShapeCallHelper<TypeTag<T>> {
    template <int in_idx, int vec_in_idx, int attr_idx>
    static Return InferShape(
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
        const std::vector<paddle::any>& attrs,
        const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_SHAPE(...) \
  ::paddle::InferShapeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferShape

/////////////// InferDataType Function (PD_INFER_DTYPE) ///////////////

// Record Op Infer dtype core function
using InferDtypeFunc = std::vector<DataType> (*)(
    const std::vector<DataType>& input_dtypes,
    const std::vector<std::vector<DataType>>& vec_input_dtypes,
    const std::vector<paddle::any>& attrs);

#define PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPE(input_type)     \
  template <typename... Tail>                                        \
  struct InferDtypeCallHelper<input_type, Tail...> {                 \
    template <int in_idx,                                            \
              int vec_in_idx,                                        \
              int attr_idx,                                          \
              typename... PreviousArgs>                              \
    static Return InferDtype(                                        \
        const std::vector<DataType>& input_dtypes,                   \
        const std::vector<std::vector<DataType>>& vec_input_dtypes,  \
        const std::vector<paddle::any>& attrs,                       \
        const PreviousArgs&... pargs) {                              \
      input_type arg = input_dtypes[in_idx];                         \
      return InferDtypeCallHelper<Tail...>::                         \
          template InferDtype<in_idx + 1, vec_in_idx, attr_idx>(     \
              input_dtypes, vec_input_dtypes, attrs, pargs..., arg); \
    }                                                                \
  }

#define PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(input_type)    \
  template <typename... Tail>                                        \
  struct InferDtypeCallHelper<input_type, Tail...> {                 \
    template <int in_idx,                                            \
              int vec_in_idx,                                        \
              int attr_idx,                                          \
              typename... PreviousArgs>                              \
    static Return InferDtype(                                        \
        const std::vector<DataType>& input_dtypes,                   \
        const std::vector<std::vector<DataType>>& vec_input_dtypes,  \
        const std::vector<paddle::any>& attrs,                       \
        const PreviousArgs&... pargs) {                              \
      input_type arg = vec_input_dtypes[vec_in_idx];                 \
      return InferDtypeCallHelper<Tail...>::                         \
          template InferDtype<in_idx, vec_in_idx + 1, attr_idx>(     \
              input_dtypes, vec_input_dtypes, attrs, pargs..., arg); \
    }                                                                \
  }

#define PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(attr_type)               \
  template <typename... Tail>                                                \
  struct InferDtypeCallHelper<attr_type, Tail...> {                          \
    template <int in_idx,                                                    \
              int vec_in_idx,                                                \
              int attr_idx,                                                  \
              typename... PreviousArgs>                                      \
    static Return InferDtype(                                                \
        const std::vector<DataType>& input_dtypes,                           \
        const std::vector<std::vector<DataType>>& vec_input_dtypes,          \
        const std::vector<paddle::any>& attrs,                               \
        const PreviousArgs&... pargs) {                                      \
      try {                                                                  \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);        \
        return InferDtypeCallHelper<Tail...>::                               \
            template InferDtype<in_idx, vec_in_idx, attr_idx + 1>(           \
                input_dtypes, vec_input_dtypes, attrs, pargs..., arg);       \
      } catch (paddle::bad_any_cast&) {                                      \
        PD_THROW(                                                            \
            "Attribute cast error in custom operator InferDtypeFn. "         \
            "Expected " #attr_type                                           \
            " value. InferDtypeFn's attribute list must be exactly same as " \
            "Forward KernelFn's attribute list");                            \
      }                                                                      \
    }                                                                        \
  }

template <typename F, F f>
struct InferDtypeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferDtypeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferDtype(
      const std::vector<DataType>& input_dtypes,
      const std::vector<std::vector<DataType>>& vec_input_dtypes,
      const std::vector<paddle::any>& attrs) {
    return InferDtypeCallHelper<Args..., TypeTag<int>>::
        template InferDtype<0, 0, 0>(input_dtypes, vec_input_dtypes, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct InferDtypeCallHelper;

  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPE(const DataType&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(const std::vector<DataType>&);

  template <typename... Tail>
  struct InferDtypeCallHelper<const paddle::optional<DataType>&, Tail...> {
    template <int in_idx,
              int vec_in_idx,
              int attr_idx,
              typename... PreviousArgs>
    static Return InferDtype(
        const std::vector<DataType>& input_dtypes,
        const std::vector<std::vector<DataType>>& vec_input_dtypes,
        const std::vector<paddle::any>& attrs,
        const PreviousArgs&... pargs) {
      const DataType& arg = input_dtypes[in_idx];
      if (arg == DataType::UNDEFINED) {
        return InferDtypeCallHelper<Tail...>::
            template InferDtype<in_idx + 1, vec_in_idx, attr_idx>(
                input_dtypes, vec_input_dtypes, attrs, pargs..., paddle::none);
      } else {
        return InferDtypeCallHelper<Tail...>::
            template InferDtype<in_idx + 1, vec_in_idx, attr_idx>(
                input_dtypes, vec_input_dtypes, attrs, pargs..., arg);
      }
    }
  };

  template <typename... Tail>
  struct InferDtypeCallHelper<const paddle::optional<std::vector<DataType>>&,
                              Tail...> {
    template <int in_idx,
              int vec_in_idx,
              int attr_idx,
              typename... PreviousArgs>
    static Return InferDtype(
        const std::vector<DataType>& input_dtypes,
        const std::vector<std::vector<DataType>>& vec_input_dtypes,
        const std::vector<paddle::any>& attrs,
        const PreviousArgs&... pargs) {
      const std::vector<DataType>& arg = vec_input_dtypes[vec_in_idx];
      if (arg.empty()) {
        return InferDtypeCallHelper<Tail...>::
            template InferDtype<in_idx, vec_in_idx + 1, attr_idx>(
                input_dtypes, vec_input_dtypes, attrs, pargs..., paddle::none);
      } else {
        return InferDtypeCallHelper<Tail...>::
            template InferDtype<in_idx, vec_in_idx + 1, attr_idx>(
                input_dtypes, vec_input_dtypes, attrs, pargs..., arg);
      }
    }
  };

  // NOTE(HongyuJia): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPE(DataType);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(std::vector<DataType>);

  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(bool);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(int);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(float);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(int64_t);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const std::string&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const std::vector<int>&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const std::vector<float>&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const std::vector<int64_t>&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const std::vector<std::string>&);

  // NOTE(HongyuJia): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const bool&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const int&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const float&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(const int64_t&);

  // NOTE(HongyuJia): Used to be compatible with the 2.1 released
  // interface, but not recommended
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(std::string);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(std::vector<int>);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(std::vector<float>);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_ATTR(std::vector<std::string>);

  // end: base template
  template <typename T>
  struct InferDtypeCallHelper<TypeTag<T>> {
    template <int in_idx, int vec_in_idx, int attr_idx>
    static Return InferDtype(
        const std::vector<DataType>& input_dtypes,
        const std::vector<std::vector<DataType>>& vec_input_dtypes,
        const std::vector<paddle::any>& attrs,
        const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_DTYPE(...) \
  ::paddle::InferDtypeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferDtype

#ifdef PADDLE_WITH_TENSORRT
////////////////////// GetOutputDimensions Function (PD_TRT_INFER_SHAPE)
///////////////////////////
using TrtGetOutputDimsFunc = nvinfer1::DimsExprs (*)(
    std::pair<int32_t, int32_t> output_index_and_nb_inputs,
    const nvinfer1::DimsExprs* inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const std::vector<paddle::any>& attrs);

#define PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(attr_type)         \
  template <typename... Tail>                                                \
  struct TrtGetOutputDimsCallHelper<attr_type, Tail...> {                    \
    template <int attr_idx, typename... PreviousArgs>                        \
    static Return GetOutputDims(                                             \
        std::pair<int32_t, int32_t> output_index_and_nb_inputs,              \
        const nvinfer1::DimsExprs* inputs,                                   \
        nvinfer1::IExprBuilder& expr_builder,                                \
        const std::vector<paddle::any>& attrs,                               \
        PreviousArgs&... pargs) {                                            \
      try {                                                                  \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);        \
        return TrtGetOutputDimsCallHelper<Tail...>::template GetOutputDims<  \
            attr_idx + 1>(output_index_and_nb_inputs,                        \
                          inputs,                                            \
                          expr_builder,                                      \
                          attrs,                                             \
                          pargs...,                                          \
                          arg);                                              \
      } catch (paddle::bad_any_cast&) {                                      \
        PD_THROW(                                                            \
            "Attribute cast error in custom operator TrtGetOutputDimsFunc "  \
            "function. "                                                     \
            "Expected " #attr_type                                           \
            " value. TrtGetOutputDimsFunc's attribute list must be exactly " \
            "same "                                                          \
            "as "                                                            \
            "Forward "                                                       \
            "KernelFn's attribute list except std::vector<int64_t> "         \
            "attribute.");                                                   \
      }                                                                      \
    }                                                                        \
  }

template <typename F, F f>
struct TrtGetOutputDimsFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct TrtGetOutputDimsFuncImpl<Return (*)(Args...), impl_fn> {
  static Return GetOutputDims(
      std::pair<int32_t, int32_t> output_index_and_nb_inputs,
      const nvinfer1::DimsExprs* inputs,
      nvinfer1::IExprBuilder& expr_builder,  // NOLINT
      const std::vector<paddle::any>& attrs) {
    return TrtGetOutputDimsCallHelper<Args..., TypeTag<int>>::
        template GetOutputDims<0>(
            output_index_and_nb_inputs, inputs, expr_builder, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct TrtGetOutputDimsCallHelper;

  template <typename... Tail>
  struct TrtGetOutputDimsCallHelper<std::pair<int32_t, int32_t>, Tail...> {
    template <int attr_idx, typename... PreviousArgs>
    static Return GetOutputDims(
        std::pair<int32_t, int32_t> output_index_and_nb_inputs,
        const nvinfer1::DimsExprs* inputs,
        nvinfer1::IExprBuilder& expr_builder,  // NOLINT
        const std::vector<paddle::any>& attrs,
        PreviousArgs&... pargs) {
      return TrtGetOutputDimsCallHelper<Tail...>::template GetOutputDims<
          attr_idx>(output_index_and_nb_inputs,
                    inputs,
                    expr_builder,
                    attrs,
                    pargs...,
                    output_index_and_nb_inputs);
    }
  };

  template <typename... Tail>
  struct TrtGetOutputDimsCallHelper<const nvinfer1::DimsExprs*, Tail...> {
    template <int attr_idx, typename... PreviousArgs>
    static Return GetOutputDims(
        std::pair<int32_t, int32_t> output_index_and_nb_inputs,
        const nvinfer1::DimsExprs* inputs,
        nvinfer1::IExprBuilder& expr_builder,  // NOLINT
        const std::vector<paddle::any>& attrs,
        PreviousArgs&... pargs) {
      return TrtGetOutputDimsCallHelper<Tail...>::template GetOutputDims<
          attr_idx>(output_index_and_nb_inputs,
                    inputs,
                    expr_builder,
                    attrs,
                    pargs...,
                    inputs);
    }
  };

  template <typename... Tail>
  struct TrtGetOutputDimsCallHelper<nvinfer1::IExprBuilder&, Tail...> {
    template <int attr_idx, typename... PreviousArgs>
    static Return GetOutputDims(
        std::pair<int32_t, int32_t> output_index_and_nb_inputs,
        const nvinfer1::DimsExprs* inputs,
        nvinfer1::IExprBuilder& expr_builder,  // NOLINT
        const std::vector<paddle::any>& attrs,
        PreviousArgs&... pargs) {
      return TrtGetOutputDimsCallHelper<Tail...>::template GetOutputDims<
          attr_idx>(output_index_and_nb_inputs,
                    inputs,
                    expr_builder,
                    attrs,
                    pargs...,
                    expr_builder);
    }
  };

  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(bool);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(int);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(float);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(int64_t);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const std::string&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const std::vector<int>&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const std::vector<float>&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(
      const std::vector<std::string>&);

  // NOTE(HongyuJia): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const bool&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const int&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const float&);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(const int64_t&);

  // NOTE(HongyuJia): Used to be compatible with the 2.1 released
  // interface, but not recommended
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(std::string);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(std::vector<int>);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(std::vector<float>);
  PD_SPECIALIZE_TrtGetOutputDimsCallHelper_FOR_ATTR(std::vector<std::string>);

  // end: base template
  template <typename T>
  struct TrtGetOutputDimsCallHelper<TypeTag<T>> {
    template <int attr_idx>
    static Return GetOutputDims(
        std::pair<int32_t, int32_t> output_index_and_nb_inputs,
        const nvinfer1::DimsExprs* inputs,
        nvinfer1::IExprBuilder& expr_builder,  // NOLINT
        const std::vector<paddle::any>& attrs,
        const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_TRT_INFER_SHAPE(...)                              \
  ::paddle::TrtGetOutputDimsFuncImpl<decltype(&__VA_ARGS__), \
                                     &__VA_ARGS__>::GetOutputDims
#endif

////////////////////// Op Meta Info //////////////////////

using CustomSpmdInferTensorArg =
    paddle::variant<phi::distributed::DistMetaTensor,
                    std::vector<phi::distributed::DistMetaTensor>>;
using CustomSpmdInferAttrArg = paddle::any;

using InferSpmdFunc = phi::distributed::SpmdInfo (*)(
    const std::vector<CustomSpmdInferTensorArg>& inputs,
    const std::vector<CustomSpmdInferAttrArg>& attrs);

class PADDLE_API OpMetaInfo {
 public:
  explicit OpMetaInfo(const std::string& op_name) : name_(op_name) {}

  // format: {"<name1>", "<name2>", ...}
  OpMetaInfo& Inputs(std::vector<std::string>&& inputs);

  // format: {"<name1>", "<name2>", ...}
  OpMetaInfo& Outputs(std::vector<std::string>&& outputs);

  // format: {"<name1>:<type1>", "<name2>:<type2>", ...}
  OpMetaInfo& Attrs(std::vector<std::string>&& attrs);

  // format: {"<input_name1>:<output_name1>",
  // "<input_name2>:<output_name2>",...}
  OpMetaInfo& SetInplaceMap(
      std::unordered_map<std::string, std::string>&& inplace_map);

  // format: PD_KERNEL(...)
  OpMetaInfo& SetKernelFn(KernelFunc&& func);

  // format: PD_INFER_SHAPE(...)
  OpMetaInfo& SetInferShapeFn(InferShapeFunc&& func);

  // format: PD_INFER_DTYPE(...)
  OpMetaInfo& SetInferDtypeFn(InferDtypeFunc&& func);

  // format: PD_INFER_SPMD_RULE(...)
  OpMetaInfo& SetInferSpmdFn(InferSpmdFunc&& func);

  bool IsGradOp() const;

  bool IsDoubleGradOp() const;

#ifdef PADDLE_WITH_TENSORRT
  // format: PD_TRT_INFER_SHAPE(...)
  OpMetaInfo& SetTrtInferShapeFn(TrtGetOutputDimsFunc&& func);
  // format: {"<dtype1>[:format1]+<dtype2>[:format2]+...",
  // "<dtype3>[:format3]+<dtype4>[:format4]+..."}
  OpMetaInfo& SetTrtSupportsFormatConfig(std::vector<std::string>&& config);
#endif

 private:
  friend class OpMetaInfoHelper;

  // 1. desc info
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
  std::unordered_map<std::string, std::string> inplace_map_;
  std::unordered_map<std::string, std::string> inplace_reverse_map_;
  // 2. func info
  KernelFunc kernel_fn_{nullptr};
  InferShapeFunc infer_shape_fn_{nullptr};
  InferDtypeFunc infer_dtype_fn_{nullptr};
  InferSpmdFunc infer_spmd_fn_{nullptr};
#ifdef PADDLE_WITH_TENSORRT
  TrtGetOutputDimsFunc trt_infer_shape_fn_{nullptr};
  std::vector<std::string> trt_supports_format_config_;
#endif
};

//////////////// Op Meta Info Helper /////////////////
class OpMetaInfoHelper {
 public:
  static const std::string& GetOpName(const paddle::OpMetaInfo& info);
  static const std::vector<std::string>& GetInputs(
      const paddle::OpMetaInfo& info);
  static const std::vector<std::string>& GetOutputs(
      const paddle::OpMetaInfo& info);
  static const std::vector<std::string>& GetAttrs(
      const paddle::OpMetaInfo& info);
  static const std::unordered_map<std::string, std::string>& GetInplaceMap(
      const paddle::OpMetaInfo& info);
  static const std::unordered_map<std::string, std::string>&
  GetInplaceReverseMap(const paddle::OpMetaInfo& info);
  static const KernelFunc& GetKernelFn(const paddle::OpMetaInfo& info);
  static const InferShapeFunc& GetInferShapeFn(const paddle::OpMetaInfo& info);
  static const InferDtypeFunc& GetInferDtypeFn(const paddle::OpMetaInfo& info);
  static const InferSpmdFunc& GetInferSpmdFn(const paddle::OpMetaInfo& info);

#ifdef PADDLE_WITH_TENSORRT
  static const TrtGetOutputDimsFunc& GetTrtInferShapeFn(
      const paddle::OpMetaInfo& info);
  static const std::vector<std::string>& GetTrtSupportsFormatConfig(
      const paddle::OpMetaInfo& info);
#endif
};

//////////////// Op Meta Info Map /////////////////

class PADDLE_API OpMetaInfoMap {
 public:
  static OpMetaInfoMap& Instance();

  std::vector<OpMetaInfo>& operator[](const std::string& name);

  const std::unordered_map<std::string, std::vector<OpMetaInfo>>& GetMap()
      const;

 private:
  OpMetaInfoMap() = default;
  std::unordered_map<std::string, std::vector<OpMetaInfo>> map_;

  PD_DISABLE_COPY_AND_ASSIGN(OpMetaInfoMap);
};

//////////////// Op Meta Info Builder /////////////////

class PADDLE_API OpMetaInfoBuilder {
 public:
  explicit OpMetaInfoBuilder(std::string&& name, size_t index);
  OpMetaInfoBuilder& Inputs(std::vector<std::string>&& inputs);
  OpMetaInfoBuilder& Outputs(std::vector<std::string>&& outputs);
  OpMetaInfoBuilder& Attrs(std::vector<std::string>&& attrs);
  OpMetaInfoBuilder& SetInplaceMap(
      std::unordered_map<std::string, std::string>&& inplace_map);
  OpMetaInfoBuilder& SetKernelFn(KernelFunc func);
  OpMetaInfoBuilder& SetInferShapeFn(InferShapeFunc func);
  OpMetaInfoBuilder& SetInferDtypeFn(InferDtypeFunc func);
  OpMetaInfoBuilder& SetInferSpmdFn(InferSpmdFunc func);

#ifdef PADDLE_WITH_TENSORRT
  OpMetaInfoBuilder& SetTrtInferShapeFn(TrtGetOutputDimsFunc func);
  OpMetaInfoBuilder& SetTrtSupportsFormatConfig(
      std::vector<std::string>&& config);
#endif

 private:
  // Forward Op name
  std::string name_;
  // ref current info ptr
  OpMetaInfo* info_ptr_;
  // The current op meta info index in vector
  // - 0: op, 1: grad_op, 2: grad_grad_op
  size_t index_;
};

/////////////////////// Op register Macro /////////////////////////

#define PD_BUILD_OP(op_name)                                                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      __reg_op__##op_name, "PD_BUILD_OP must be called in global namespace."); \
  static ::paddle::OpMetaInfoBuilder __op_meta_info_##op_name##__ =            \
      ::paddle::OpMetaInfoBuilder(#op_name, 0)

#define PD_BUILD_GRAD_OP(op_name)                                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      __reg_grad_op__##op_name,                                          \
      "PD_BUILD_GRAD_OP must be called in global namespace.");           \
  static ::paddle::OpMetaInfoBuilder __grad_op_meta_info_##op_name##__ = \
      ::paddle::OpMetaInfoBuilder(#op_name, 1)

#define PD_BUILD_DOUBLE_GRAD_OP(op_name)                                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                             \
      __reg_grad_grad_op__##op_name,                                          \
      "PD_BUILD_DOUBLE_GRAD_OP must be called in global namespace.");         \
  static ::paddle::OpMetaInfoBuilder __grad_grad_op_meta_info_##op_name##__ = \
      ::paddle::OpMetaInfoBuilder(#op_name, 2)

}  // namespace paddle

///////////////////// C API ///////////////////

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
// C-API to get global OpMetaInfoMap.
__declspec(dllexport) inline paddle::OpMetaInfoMap& PD_GetOpMetaInfoMap() {
  return paddle::OpMetaInfoMap::Instance();
}
#endif  // _WIN32

#ifdef __cplusplus
}
#endif
