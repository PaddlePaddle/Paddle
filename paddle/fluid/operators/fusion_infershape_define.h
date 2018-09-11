/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_OPERATORS_FUSION_INFERSHAPE_DEFINE_H_
#define PADDLE_FLUID_OPERATORS_FUSION_INFERSHAPE_DEFINE_H_

#include <string>
#include "paddle/fluid/framework/shape_runtime_infer.h"

namespace paddle {
namespace operators {

#define FUSION_INFERSHAPE_INIT                                                 \
  auto* runtime_ctx = dynamic_cast<framework::RuntimeInferShapeContext*>(ctx); \
  if (runtime_ctx == nullptr) {                                                \
    LOG(FATAL) << "Should have runtime infer context";                         \
  }                                                                            \
  const auto& ins = runtime_ctx->OpBase().Inputs();                            \
  const auto& outs = runtime_ctx->OpBase().Outputs();                          \
  const auto& scope = runtime_ctx->InferScope();                               \
  const auto ins_end = ins.end();                                              \
  const auto outs_end = outs.end();                                            \
  auto fair_input = [&](const std::string& name) -> bool {                     \
    auto it = ins.find(name);                                                  \
    if (it == ins_end) {                                                       \
      return false;                                                            \
    }                                                                          \
    const auto& in = it->second;                                               \
    if (in.size() != 1 || in[0] == framework::kEmptyVarName) {                 \
      return false;                                                            \
    }                                                                          \
    return scope.FindVar(in[0]) != nullptr;                                    \
  };                                                                           \
  auto fair_output = [&](const std::string& name) -> bool {                    \
    auto it = outs.find(name);                                                 \
    if (it == outs_end) {                                                      \
      return false;                                                            \
    }                                                                          \
    const auto& out = it->second;                                              \
    if (out.size() != 1 || out[0] == framework::kEmptyVarName) {               \
      return false;                                                            \
    }                                                                          \
    return scope.FindVar(out[0]) != nullptr;                                   \
  }

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_FLUID_OPERATORS_FUSION_INFERSHAPE_DEFINE_H_
