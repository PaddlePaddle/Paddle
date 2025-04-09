// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"

#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/imperative/saved_variable_wrapper_list.h"

namespace paddle::framework {

const Attribute &InferNoNeedBufferVarsContext::GetAttr(
    const std::string &name) const {
  auto iter = attrs_.find(name);
  PADDLE_ENFORCE_NE(
      iter,
      attrs_.end(),
      common::errors::NotFound("Cannot find attribute (%s).", name));
  return iter->second;
}

StaticGraphInferNoNeedBufferVarsContext::
    StaticGraphInferNoNeedBufferVarsContext(const VariableNameMap &inputs,
                                            const VariableNameMap &outputs,
                                            const AttributeMap &attrs)
    : InferNoNeedBufferVarsContext(attrs), inputs_(inputs), outputs_(outputs) {}

bool StaticGraphInferNoNeedBufferVarsContext::HasOutput(
    const std::string &slot) const {
  auto iter = outputs_.find(slot);
  if (iter != outputs_.end()) {
    for (auto &var : iter->second) {
      if (var != kEmptyVarName) return true;
    }
  }
  return false;
}

DyGraphInferNoNeedBufferVarsContext::DyGraphInferNoNeedBufferVarsContext(
    const imperative::NameVarMap<imperative::VariableWrapper> &inputs,
    const imperative::NameVarMap<imperative::VariableWrapper> &outputs,
    const AttributeMap &attrs)
    : InferNoNeedBufferVarsContext(attrs), inputs_(inputs), outputs_(outputs) {}

bool DyGraphInferNoNeedBufferVarsContext::HasOutput(
    const std::string &slot) const {
  auto iter = outputs_.find(slot);
  if (iter != outputs_.end()) {
    for (auto &var : iter->second) {
      if (var) return true;
    }
  }
  return false;
}

}  // namespace paddle::framework
