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

#include "paddle/fluid/framework/shape_inference.h"

namespace pten {
class InferMetaContext;
}  // namespace pten

namespace paddle {
namespace framework {

pten::InferMetaContext BuildInferMetaContext(InferShapeContext* ctx,
                                             const std::string& op_type);

template <typename Fn, typename T>
struct InferMetaFunctor;

#define DELCARE_INFER_SHAPE_FUNCTOR(op_type, functor_name, fn) \
  struct functor_name : InferMetaFunctor<fn, functor_name> {   \
    functor_name() = default;                                  \
    functor_name(const functor_name&) = delete;                \
    functor_name& operator=(const functor_name&) = delete;     \
    void operator(InferShapeContext* ctx) {                    \
      fn(&(BuildInferMetaContext(ctx, #op_type)));             \
    }                                                          \
  }

}  // namespace framework
}  // namespace paddle
