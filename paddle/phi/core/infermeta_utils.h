/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {

class InferMetaContext {
 public:
  InferMetaContext() = default;
  explicit InferMetaContext(MetaConfig config) : config_(config) {}

  void SetMetaConfig(MetaConfig config);
  const MetaConfig& GetMetaConfig() const;

  void EmplaceBackInput(MetaTensor input);
  void EmplaceBackOutput(MetaTensor output);
  void EmplaceBackAttr(Attribute attr);

  void EmplaceBackInputs(
      paddle::small_vector<MetaTensor, phi::kInputSmallVectorSize> inputs);
  void EmplaceBackOutputs(
      paddle::small_vector<MetaTensor, phi::kOutputSmallVectorSize> outputs);

  virtual const MetaTensor& InputAt(size_t idx) const;

  virtual std::vector<const MetaTensor*> InputsBetween(size_t start,
                                                       size_t end) const;
  virtual paddle::optional<std::vector<const MetaTensor*>>
  OptionalInputsBetween(size_t start, size_t end) const;

  virtual MetaTensor* MutableOutputAt(size_t idx);
  virtual std::vector<MetaTensor*> MutableOutputBetween(size_t start,
                                                        size_t end);

  template <typename AttrType>
  const AttrType& AttrAt(size_t idx) const;

  const std::pair<int, int>& InputRangeAt(size_t idx) const;
  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  virtual ~InferMetaContext() = default;

 protected:
  MetaConfig config_;

  paddle::small_vector<Attribute, kAttrSmallVectorSize> attrs_;

  paddle::small_vector<std::pair<int, int>, phi::kInputSmallVectorSize>
      input_range_;
  paddle::small_vector<std::pair<int, int>, phi::kOutputSmallVectorSize>
      output_range_;

 private:
  paddle::small_vector<MetaTensor, phi::kInputSmallVectorSize> inputs_;
  paddle::small_vector<MetaTensor, phi::kOutputSmallVectorSize> outputs_;
};

#define PD_INFER_META(...) \
  ::phi::InferMetaFnImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Call

#define PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(attr_type)           \
  template <typename... Tail>                                                  \
  struct InferMetaFnCallHelper<attr_type, Tail...> {                           \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {          \
      static_assert(out_idx == 0,                                              \
                    "InferMeta's Attributes should appear before Outputs.");   \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                        \
      InferMetaFnCallHelper<                                                   \
          Tail...>::template Call<in_idx, attr_idx + 1, out_idx>(ctx,          \
                                                                 pargs...,     \
                                                                 arg);         \
    }                                                                          \
  }

#define PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(attr_type) \
  template <typename... Tail>                                                  \
  struct InferMetaFnCallHelper<const attr_type&, Tail...> {                    \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {          \
      static_assert(out_idx == 0,                                              \
                    "InferMeta's Attributes should appear before Outputs.");   \
      const attr_type& arg = ctx->AttrAt<attr_type>(attr_idx);                 \
      InferMetaFnCallHelper<                                                   \
          Tail...>::template Call<in_idx, attr_idx + 1, out_idx>(ctx,          \
                                                                 pargs...,     \
                                                                 arg);         \
    }                                                                          \
  }

template <typename T>
struct InferMetaTypeTag {};

template <typename Fn, Fn fn>
struct InferMetaFnImpl;

template <typename Return, typename... Args, Return (*infer_meta_fn)(Args...)>
struct InferMetaFnImpl<Return (*)(Args...), infer_meta_fn> {
  static void Call(InferMetaContext* ctx) {
    InferMetaFnCallHelper<Args...,
                          InferMetaTypeTag<int>>::template Call<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct InferMetaFnCallHelper;

  template <typename... Tail>
  struct InferMetaFnCallHelper<const MetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferMeta's Input should appear before Attributes.");
      static_assert(out_idx == 0,
                    "InferMeta's Input should appear before Outputs.");
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
      const MetaTensor& arg = ctx->InputAt(range.first);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                 pargs...,
                                                                 arg);
    }
  };

  template <typename... Tail>
  struct InferMetaFnCallHelper<const std::vector<const MetaTensor*>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferMeta's Input should appear before Attributes.");
      static_assert(out_idx == 0,
                    "InferMeta's Input should appear before Outputs.");
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
      std::vector<const MetaTensor*> arg =
          ctx->InputsBetween(range.first, range.second);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                 pargs...,
                                                                 arg);
    }
  };

  template <typename... Tail>
  struct InferMetaFnCallHelper<
      const paddle::optional<std::vector<const MetaTensor*>>&,
      Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferMeta's Input should appear before Attributes.");
      static_assert(out_idx == 0,
                    "InferMeta's Input should appear before Outputs.");
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
      paddle::optional<std::vector<const MetaTensor*>> arg =
          ctx->OptionalInputsBetween(range.first, range.second);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                 pargs...,
                                                                 arg);
    }
  };

  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(int);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(DataType);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(Backend);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(DataLayout);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(std::string);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(Scalar);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(IntArray);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<bool>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(std::vector<int>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<int64_t>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<float>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<double>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<std::string>);
  PD_SPECIALIZE_InferMetaFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<Scalar>);

  template <typename... Tail>
  struct InferMetaFnCallHelper<MetaTensor*, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
      MetaTensor* arg = ctx->MutableOutputAt(range.first);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx, attr_idx, out_idx + 1>(ctx,
                                                                 pargs...,
                                                                 arg);
    }
  };

  template <typename... Tail>
  struct InferMetaFnCallHelper<std::vector<MetaTensor*>, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
      std::vector<MetaTensor*> arg =
          ctx->MutableOutputBetween(range.first, range.second);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx, attr_idx, out_idx + 1>(ctx,
                                                                 pargs...,
                                                                 arg);
    }
  };

  template <typename... Tail>
  struct InferMetaFnCallHelper<MetaConfig, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      MetaConfig arg = ctx->GetMetaConfig();
      InferMetaFnCallHelper<Tail...>::template Call<in_idx, attr_idx, out_idx>(
          ctx, pargs..., arg);
    }
  };

  /* End case */
  template <typename T>
  struct InferMetaFnCallHelper<InferMetaTypeTag<T>> {
    template <int in_idx, int attr_idx, int out_idx>
    static void Call(InferMetaContext* ctx, Args&... args) {
      return infer_meta_fn(args...);
    }
  };
};

class MetaFnFactory {
 public:
  static MetaFnFactory& Instance();

  bool Contains(const std::string& kernel_name_prefix) const {
    return meta_fn_map_.count(kernel_name_prefix) > 0;
  }

  void Insert(std::string kernel_name_prefix, InferMetaFn infer_meta_fn) {
    PADDLE_ENFORCE_NE(
        Contains(kernel_name_prefix),
        true,
        phi::errors::AlreadyExists(
            "`%s`'s Series Kernel's InferMetaFn has been registered.",
            kernel_name_prefix));
    meta_fn_map_.insert(
        {std::move(kernel_name_prefix), std::move(infer_meta_fn)});
  }

  const InferMetaFn& Get(const std::string& kernel_name_prefix) const {
    auto it = meta_fn_map_.find(kernel_name_prefix);
    PADDLE_ENFORCE_NE(
        it,
        meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s`'s Series Kernel's InferMetaFn is not registered.",
            kernel_name_prefix));
    return it->second;
  }

 private:
  MetaFnFactory() = default;

  /**
   * [ Why use kernel name prefix? ]
   *
   * one op -> a matrix of kernels
   *
   * such as, scale op, it may correspond to the following kernels:
   *
   * - scale, scale_sr, scale_dnnl
   * - scale_raw, scale_raw_sr, scale_raw_dnnl
   *
   * All the kernels in each row correspond to the same infershape function,
   * the number of kernel arguments in the same row is the same, and only
   * the tensor types in the arguments are different.
   */
  paddle::flat_hash_map<std::string, InferMetaFn> meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(MetaFnFactory);
};

struct InferMetaFnRegistrar {
  InferMetaFnRegistrar(const char* kernel_name_prefix,
                       InferMetaFn infer_meta_fn) {
    MetaFnFactory::Instance().Insert(kernel_name_prefix,
                                     std::move(infer_meta_fn));
  }
};

#define PD_REGISTER_INFER_META_FN(kernel_name_prefix, variadic_infer_meta_fn) \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      PD_REGISTER_infer_meta_fn_ns_check_##kernel_name_prefix,                \
      "PD_REGISTER_INFER_META_FN must be called in global namespace.");       \
  static const ::phi::InferMetaFnRegistrar                                    \
      __registrar_arg_map_fn_for_##kernel_name_prefix(                        \
          #kernel_name_prefix, PD_INFER_META(variadic_infer_meta_fn))

}  // namespace phi
