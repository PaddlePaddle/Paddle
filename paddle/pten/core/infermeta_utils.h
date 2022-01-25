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
#include <utility>

#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/kernel_def.h"
#include "paddle/pten/core/macros.h"
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace pten {

// TODO(chenweihang): add other flags if needed
struct MetaConfig {
  bool is_runtime{true};

  MetaConfig() = default;

  // supporting implicit construction is easier to use
  MetaConfig(bool is_runtime) : is_runtime(is_runtime) {}  // NOLINT
};

class InferMetaContext {
 public:
  InferMetaContext() = default;
  explicit InferMetaContext(MetaConfig config) : config_(config) {}

  void SetMetaConfig(MetaConfig config);
  void EmplaceBackInput(std::shared_ptr<pten::MetaTensor> input);
  void EmplaceBackOutput(std::shared_ptr<pten::MetaTensor> output);
  void EmplaceBackAttr(paddle::any attr);

  void EmplaceBackInputs(
      paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> inputs);
  void EmplaceBackOutputs(
      paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> outputs);

  const std::pair<int, int>& InputRangeAt(size_t idx) const;
  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  const MetaConfig& GetMetaConfig() const;
  const MetaTensor& InputAt(size_t idx) const;
  MetaTensor* MutableOutputAt(size_t idx);

  template <typename AttrType>
  AttrType AttrAt(size_t idx) {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(pten::errors::InvalidArgument(
          "Attribute cast error in InferMeta Context."));
    }
  }

 private:
  MetaConfig config_;

  // NOTE(chenweihang): Because the MetaTensor is a base class, and MetaTensor
  // objects are all created in each round, so we have to use smart pointer
  // here, maybe we can implemented a new InferMetaContext and a series utils
  // specifically for fluid to avoid using shared_ptr
  paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> inputs_;
  paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> outputs_;
  paddle::SmallVector<paddle::any> attrs_;

  paddle::SmallVector<std::pair<int, int>> input_range_;
  paddle::SmallVector<std::pair<int, int>> output_range_;
};

#define PT_INFER_META(...) \
  ::pten::InferMetaFnImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Call

#define PT_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(attr_type)           \
  template <typename... Tail>                                                  \
  struct InferMetaFnCallHelper<attr_type, Tail...> {                           \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {          \
      static_assert(out_idx == 0,                                              \
                    "InferMeta's Attributes should appear before Outputs.");   \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                        \
      InferMetaFnCallHelper<                                                   \
          Tail...>::template Call<in_idx, attr_idx + 1, out_idx>(pargs...,     \
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

  // TODO(chenweihang): support vector<MetaTensor> input later

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

  // TODO(chenweihang): support vector<MetaTensor> output later

  template <typename... Tail>
  struct InferMetaFnCallHelper<MetaConfig, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      const MetaConfig& arg = ctx->GetMetaConfig();
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

class MetaFunctionMap {
 public:
  static MetaFunctionMap& Instance();

  bool Contains(const std::string& kernel_name_prefix) const {
    return meta_fn_map_.count(kernel_name_prefix) > 0;
  }

  void Insert(std::string kernel_name_prefix, InferMetaFn infer_meta_fn) {
    PADDLE_ENFORCE_NE(
        Contains(kernel_name_prefix),
        true,
        platform::errors::AlreadyExists(
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
        pten::errors::NotFound(
            "`%s`'s Series Kernel's InferMetaFn is not registered.",
            kernel_name_prefix));
    return it->second;
  }

 private:
  MetaFunctionMap() = default;

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

  DISABLE_COPY_AND_ASSIGN(MetaFunctionMap);
};

struct InferMetaFnRegistrar {
  InferMetaFnRegistrar(const char* kernel_name_prefix,
                       InferMetaFn infer_meta_fn) {
    MetaFunctionMap::Instance().Insert(kernel_name_prefix,
                                       std::move(infer_meta_fn));
  }
};

#define PT_REGISTER_INFER_META_FN(kernel_name_prefix, variadic_infer_meta_fn) \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      pt_register_infer_meta_fn_ns_check_##kernel_name_prefix,                \
      "PT_REGISTER_INFER_META_FN must be called in global namespace.");       \
  static const ::pten::ArgumentMappingFnRegistrar                             \
      __registrar_arg_map_fn_for_##kernel_name_prefix(                        \
          #kernel_name_prefix, PT_INFER_META(variadic_infer_meta_fn));        \
  int TouchArgumentMappingFnSymbol_##op_type() { return 0; }

}  // namespace pten
