/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/macros.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {
namespace distributed {

class InferSpmdContext {
 public:
  InferSpmdContext() = default;
  InferSpmdContext(
      paddle::small_vector<DistMetaTensor, phi::kInputSmallVectorSize> inputs,
      paddle::small_vector<Attribute, phi::kAttrSmallVectorSize> attrs)
      : inputs_(std::move(inputs)), attrs_(std::move(attrs)) {}

  void EmplaceBackInput(DistMetaTensor input);
  void EmplaceBackAttr(Attribute attr);
  void EmplaceBackInputs(
      paddle::small_vector<DistMetaTensor, phi::kInputSmallVectorSize> inputs);

  const DistMetaTensor& InputAt(size_t idx) const;

  const std::pair<int, int>& InputRangeAt(size_t idx) const;
  const std::vector<const DistMetaTensor*> InputsBetween(size_t start,
                                                         size_t end) const;

  template <typename AttrType>
  AttrType AttrAt(size_t idx) const;

  const Attribute& AttrAt(size_t idx) const;

 private:
  // Now we only need `inputs`, for backward, the `output` is passed as input
  paddle::small_vector<DistMetaTensor, phi::kInputSmallVectorSize> inputs_;
  // Because the attribute arguments of dygraph do not have `attr name`,
  // so we use vector instead of map
  paddle::small_vector<Attribute, phi::kAttrSmallVectorSize> attrs_;
  // for vector arguments
  paddle::small_vector<std::pair<int, int>, phi::kInputSmallVectorSize>
      input_range_;
};

using InferSpmdFn = SpmdInfo (*)(const InferSpmdContext&);

#define PD_INFER_SPMD(...)                                    \
  ::phi::distributed::InferSpmdFnImpl<decltype(&__VA_ARGS__), \
                                      &__VA_ARGS__>::Call

template <typename T>
struct InferSpmdTypeTag {};

template <typename Fn, Fn fn>
struct InferSpmdFnImpl;

template <typename Return, typename... Args, Return (*infer_spmd_fn)(Args...)>
struct InferSpmdFnImpl<Return (*)(Args...), infer_spmd_fn> {
  static SpmdInfo Call(const InferSpmdContext& ctx) {
    return InferSpmdFnCallHelper<Args..., InferSpmdTypeTag<int>>::
        template Call<0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct InferSpmdFnCallHelper;

  // TODO(chenweihang): support other input type later as needed
  template <typename... Tail>
  struct InferSpmdFnCallHelper<const DistMetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static SpmdInfo Call(const InferSpmdContext& ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferSpmd's Input should appear before Attributes.");
      const DistMetaTensor& arg = ctx.InputAt(in_idx);
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx + 1,
                                                           attr_idx>(
          ctx, pargs..., arg);
    }
  };

  // for vector slot
  template <typename... Tail>
  struct InferSpmdFnCallHelper<const std::vector<const DistMetaTensor*>&,
                               Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static SpmdInfo Call(const InferSpmdContext& ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferSpmd's Input should appear before Attributes.");

      const std::pair<int, int> range = ctx.InputRangeAt(in_idx);
      std::vector<const DistMetaTensor*> arg =
          ctx.InputsBetween(range.first, range.second);
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx + 1,
                                                           attr_idx>(
          ctx, pargs..., arg);
    }
  };

  // direct vector
  template <typename... Tail>
  struct InferSpmdFnCallHelper<const std::vector<DistMetaTensor>&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static SpmdInfo Call(const InferSpmdContext& ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferSpmd's Input should appear before Attributes.");
      // TODO(liuzhenhai): parse input list as vector directly
      const std::pair<int, int> range = ctx.InputRangeAt(in_idx);
      std::vector<const DistMetaTensor*> tmp_arg =
          ctx.InputsBetween(range.first, range.second);
      std::vector<DistMetaTensor> arg;
      std::transform(tmp_arg.begin(),
                     tmp_arg.end(),
                     std::back_inserter(arg),
                     [](const DistMetaTensor* arg_ptr) { return *arg_ptr; });
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx + 1,
                                                           attr_idx>(
          ctx, pargs..., arg);
    }
  };

#define PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(attr_type)      \
  template <typename... Tail>                                             \
  struct InferSpmdFnCallHelper<attr_type, Tail...> {                      \
    template <int in_idx, int attr_idx, typename... PreviousArgs>         \
    static SpmdInfo Call(const InferSpmdContext& ctx,                     \
                         PreviousArgs&... pargs) {                        \
      attr_type arg = ctx.AttrAt<attr_type>(attr_idx);                    \
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx,        \
                                                           attr_idx + 1>( \
          ctx, pargs..., arg);                                            \
    }                                                                     \
  }

#define PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_CONST_ATTRIBUTE_REF(attr_type) \
  template <typename... Tail>                                                  \
  struct InferSpmdFnCallHelper<const attr_type&, Tail...> {                    \
    template <int in_idx, int attr_idx, typename... PreviousArgs>              \
    static SpmdInfo Call(const InferSpmdContext& ctx,                          \
                         PreviousArgs&... pargs) {                             \
      attr_type arg = ctx.AttrAt<attr_type>(attr_idx);                         \
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx,             \
                                                           attr_idx + 1>(      \
          ctx, pargs..., arg);                                                 \
    }                                                                          \
  }

  // TODO(chenweihang): support other attr type later as needed
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(int);
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_CONST_ATTRIBUTE_REF(std::vector<int>);
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<int64_t>);

  /* End case */
  template <typename T>
  struct InferSpmdFnCallHelper<InferSpmdTypeTag<T>> {
    template <int in_idx, int attr_idx>
    static SpmdInfo Call(const InferSpmdContext& ctx UNUSED, Args&... args) {
      return infer_spmd_fn(args...);
    }
  };
};

class SpmdRule {
 public:
  explicit SpmdRule(InferSpmdFn forward_fn)
      : forward_fn_(forward_fn), backward_fn_(nullptr) {}

  SpmdRule(InferSpmdFn forward_fn, InferSpmdFn backward_fn)
      : forward_fn_(forward_fn), backward_fn_(backward_fn) {}

  SpmdInfo InferForward(const InferSpmdContext& ctx) const {
    PADDLE_ENFORCE_NE(
        forward_fn_,
        nullptr,
        phi::errors::NotFound("Current SpmdRule's forward function is not "
                              "found, Please make sure "
                              "that you have registered the rule correctly."));
    return forward_fn_(ctx);
  }

  SpmdInfo InferBackward(const InferSpmdContext& ctx) const {
    PADDLE_ENFORCE_NE(
        backward_fn_,
        nullptr,
        phi::errors::NotFound("Current SpmdRule's backward function is not "
                              "found, Please make sure "
                              "that you have registered the rule correctly."));
    return backward_fn_(ctx);
  }

 private:
  InferSpmdFn forward_fn_;
  InferSpmdFn backward_fn_;
};

// SpmdRuleFactory manage the spmd rules and cache the propagate results
// TODO(chenweihang): Add spmd caching impl later
class SpmdRuleFactory {
 public:
  static SpmdRuleFactory& Instance();

  bool ContainsSpmdRule(const std::string& kernel_name) const;

  int InsertSpmdRule(std::string kernel_name, SpmdRule rule);

  const SpmdRule& GetSpmdRule(const std::string& kernel_name) const;

 private:
  SpmdRuleFactory() = default;

  paddle::flat_hash_map<std::string, SpmdRule> spmd_rule_map_;

  DISABLE_COPY_AND_ASSIGN(SpmdRuleFactory);
};

#define PD_REGISTER_SPMD_RULE(kernel_name, ...)                       \
  UNUSED static int ___registrar_spmd_rule_for_##kernel_name =        \
      ::phi::distributed::SpmdRuleFactory::Instance().InsertSpmdRule( \
          #kernel_name, ::phi::distributed::SpmdRule(__VA_ARGS__));

}  // namespace distributed
}  // namespace phi
