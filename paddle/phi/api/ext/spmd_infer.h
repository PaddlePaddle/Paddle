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
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace paddle {

using CustomSpmdInferTensorArg =
    paddle::variant<phi::distributed::DistMetaTensor,
                    std::vector<phi::distributed::DistMetaTensor>>;

using CustomSpmdInferAttrArg = paddle::any;
template <typename T>
struct SpmdInferHelperTypeEnd {};

#define PD_INFER_SPMD_CHECK_INPUTS_SIZE_GT(inputs, size_bound) \
  PD_CHECK(inputs.size() > size_bound,                         \
           #inputs,                                            \
           " size must be great than ",                        \
           size_bound)

#define PD_INFER_SPMD_CHECK_INPUTS_SIZE_EQ(inputs, size_bound) \
  PD_CHECK(                                                    \
      inputs.size() == size_bound, #inputs, " size must be eq ", size_bound)

#define PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(attr_type)              \
  template <typename... Tail>                                              \
  struct SpmdInferHelper<attr_type, Tail...> {                             \
    template <int in_idx, int attr_idx, typename... PreviousArgs>          \
    static phi::distributed::SpmdInfo InferSpmd(                           \
        const std::vector<CustomSpmdInferTensorArg>& inputs,               \
        const std::vector<CustomSpmdInferAttrArg>& attrs,                  \
        const PreviousArgs&... pargs) {                                    \
      try {                                                                \
        PD_INFER_SPMD_CHECK_INPUTS_SIZE_GT(attrs, attr_idx);               \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);      \
        return SpmdInferHelper<Tail...>::template InferSpmd<in_idx,        \
                                                            attr_idx + 1>( \
            inputs, attrs, pargs..., arg);                                 \
      } catch (paddle::bad_any_cast&) {                                    \
        PD_THROW(                                                          \
            "Attribute cast error in custom operator SpmdInferFunc "       \
            "function. "                                                   \
            "Expected " #attr_type                                         \
            " value. SpmdInferFunc's attribute list must be exactly "      \
            "same "                                                        \
            "as "                                                          \
            "Forward "                                                     \
            "KernelFn's attribute list except std::vector<int64_t> "       \
            "attribute.");                                                 \
      }                                                                    \
    }                                                                      \
  }

template <typename F, F f>
struct SpmdInferImpl;

template <typename... Args, phi::distributed::SpmdInfo (*impl_fn)(Args...)>
struct SpmdInferImpl<phi::distributed::SpmdInfo (*)(Args...), impl_fn> {
  static phi::distributed::SpmdInfo InferSpmd(
      const std::vector<CustomSpmdInferTensorArg>& inputs,
      const std::vector<CustomSpmdInferAttrArg>& attrs) {
    return SpmdInferHelper<Args..., SpmdInferHelperTypeEnd<int>>::
        template InferSpmd<0, 0>(inputs, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct SpmdInferHelper;

  // Handle args for general tensor input case
  template <typename... Tail>
  struct SpmdInferHelper<const phi::distributed::DistMetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static phi::distributed::SpmdInfo InferSpmd(
        const std::vector<CustomSpmdInferTensorArg>& inputs,
        const std::vector<CustomSpmdInferAttrArg>& attrs,
        PreviousArgs&... pargs) {
      static_assert(attr_idx == 0, "attributes must come after tensor inputs");
      PD_INFER_SPMD_CHECK_INPUTS_SIZE_GT(inputs, in_idx);
      auto& arg =
          PADDLE_GET_CONST(phi::distributed::DistMetaTensor, inputs[in_idx]);
      return SpmdInferHelper<Tail...>::template InferSpmd<in_idx + 1, attr_idx>(
          inputs, attrs, pargs..., arg);
    }
  };

  // Handle args for vector of Tensor input case
  template <typename... Tail>
  struct SpmdInferHelper<const std::vector<phi::distributed::DistMetaTensor>&,
                         Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static phi::distributed::SpmdInfo InferSpmd(
        const std::vector<CustomSpmdInferTensorArg>& inputs,
        const std::vector<CustomSpmdInferAttrArg>& attrs,
        PreviousArgs&... pargs) {
      static_assert(attr_idx == 0, "attributes must come after tensor inputs");
      PD_INFER_SPMD_CHECK_INPUTS_SIZE_GT(inputs, in_idx);
      auto& arg = PADDLE_GET_CONST(
          std::vector<phi::distributed::DistMetaTensor>, inputs[in_idx]);
      return SpmdInferHelper<Tail...>::template InferSpmd<in_idx + 1, attr_idx>(
          inputs, attrs, pargs..., arg);
    }
  };

  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(bool);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(int);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(float);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(int64_t);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::string&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::vector<int>&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::vector<float>&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::vector<std::string>&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::vector<int64_t>&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const bool&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const int&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const float&);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const int64_t&);

  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(std::string);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(std::vector<int>);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(std::vector<float>);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(std::vector<std::string>);
  PD_SPECIALIZE_SpmdInferHelper_FOR_AttrType(const std::vector<int64_t>);

  // end: base template
  template <typename T>
  struct SpmdInferHelper<SpmdInferHelperTypeEnd<T>> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static phi::distributed::SpmdInfo InferSpmd(
        const std::vector<CustomSpmdInferTensorArg>& inputs,
        const std::vector<CustomSpmdInferAttrArg>& attrs,
        PreviousArgs&... pargs) {
      PD_INFER_SPMD_CHECK_INPUTS_SIZE_EQ(inputs, in_idx);
      PD_INFER_SPMD_CHECK_INPUTS_SIZE_EQ(attrs, attr_idx);
      return impl_fn(pargs...);
    }
  };
};

#define PD_INFER_SPMD_RULE(...) \
  ::paddle::SpmdInferImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferSpmd

}  // namespace paddle
