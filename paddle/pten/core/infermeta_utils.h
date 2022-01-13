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

#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/utils/small_vector.h"

namespace pten {

struct InferMetaConfigs {
  bool is_runtime{true};

  InferMetaConfigs() = default;
  explicit InferMetaConfigs(bool is_runtime) : is_runtime(is_runtime) {}
};

class InferMetaContext {
 public:
  InferMetaContext() = default;

  void EmplaceBackInput(pten::MetaTensor input);
  void EmplaceBackOutput(pten::MetaTensor output);
  void EmplaceBackAttr(paddle::any attr);

  void EmplaceBackInputs(paddle::SmallVector<pten::MetaTensor> inputs);
  void EmplaceBackOutputs(paddle::SmallVector<pten::MetaTensor> outputs);

  const std::pair<int, int>& InputRangeAt(size_t idx) const;
  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  const MetaTensor& InputAt(size_t idx) const;
  MetaTensor& MutableOutputAt(size_t idx);

  template <typename AttrType>
  AttrType AttrAt(size_t idx) {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Attribute cast error in InferMeta Context."));
    }
  }

 private:
  paddle::SmallVector<pten::MetaTensor> inputs_;
  paddle::SmallVector<pten::MetaTensor> outputs_;
  paddle::SmallVector<paddle::any> attrs_;

  paddle::SmallVector<std::pair<int, int>> input_range_;
  paddle::SmallVector<std::pair<int, int>> output_range_;

  InferMetaConfigs configs;
};

#define PT_INFER_META(...) ::pten::InferMetaFnImpl<decltype(&__VA_ARGS__)>

#define PT_SPECIALIZE_InferMetaFnCallHelper_FOR_ATTRIBUTE(attr_type)           \
  template <typename... Tail>                                                  \
  struct InferMetaFnCallHelper<attr_type, Tail...> {                           \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static void Compute(InferMetaContext* ctx, PreviousArgs&... pargs) {       \
      static_assert(out_idx == 0,                                              \
                    "InferMeta's Attributes should appear before Outputs.");   \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                        \
      InferMetaFnCallHelper<                                                   \
          Tail...>::template Call<in_idx, attr_idx + 1, out_idx>(pargs...,     \
                                                                 arg);         \
    }                                                                          \
  }

template <typename T>
struct TypeTag {};

template <typename Fn, Fn fn>
struct InferMetaFnImpl;

template <typename Return, typename... Args, Return (*infer_meta_fn)(Args...)>
struct InferMetaFnImpl<Return (*)(Args...), infer_meta_fn> {
  static void Call(InferMetaContext* ctx) {
    InferMetaFnCallHelper<Args..., TypeTag<int>>::template Call<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs...>
  struct InferMetaFnCallHelper;

  template <typename... Tail>
  struct InferMetaFnCallHelper<const pten::MetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferMeta's Input should appear before Attributes.");
      static_assert(out_idx == 0,
                    "InferMeta's Input should appear before Outputs.");
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
      const pten::MetaTensor& arg = ctx->InputAt(range.first);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx + 1, attr_idx, out_idx>(pargs..., arg);
    }
  };

  // TODO(chenweihang): support vector<Meta> input later

  template <typename... Tail>
  struct InferMetaFnCallHelper<const pten::MetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static void Call(InferMetaContext* ctx, PreviousArgs&... pargs) {
      const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
      const pten::MetaTensor& arg = ctx->OutputAt(range.first);
      InferMetaFnCallHelper<
          Tail...>::template Call<in_idx, attr_idx, out_idx + 1>(pargs..., arg);
    }
  };

  // TODO(chenweihang): support vector<Meta> output later
};

}  // namespace pten
