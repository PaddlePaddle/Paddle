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

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace framework {

class NoNeedBufferVarsInference {
 public:
  virtual ~NoNeedBufferVarsInference() = default;

  virtual std::unordered_set<std::string> operator()(
      const VariableNameMap &inputs, const VariableNameMap &outputs,
      const AttributeMap &attrs) const = 0;

  virtual std::unordered_set<std::string> operator()(
      const imperative::NameVarBaseMap &inputs,
      const imperative::NameVarBaseMap &outputs,
      const AttributeMap &attrs) const = 0;
};

#define DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(class_type, ...)        \
  class class_type final                                              \
      : public ::paddle::framework::NoNeedBufferVarsInference {       \
   public:                                                            \
    using ::paddle::framework::NoNeedBufferVarsInference::            \
        NoNeedBufferVarsInference;                                    \
                                                                      \
    std::unordered_set<std::string> operator()(                       \
        const ::paddle::framework::VariableNameMap &inputs,           \
        const ::paddle::framework::VariableNameMap &outputs,          \
        const ::paddle::framework::AttributeMap &attrs) const final { \
      return {__VA_ARGS__};                                           \
    }                                                                 \
                                                                      \
    std::unordered_set<std::string> operator()(                       \
        const ::paddle::imperative::NameVarBaseMap &inputs,           \
        const ::paddle::imperative::NameVarBaseMap &outputs,          \
        const ::paddle::framework::AttributeMap &attrs) const final { \
      return {__VA_ARGS__};                                           \
    }                                                                 \
  }

class InferNoNeedBufferVarsFN {
 public:
  inline std::unordered_set<std::string> operator()(
      const VariableNameMap &inputs, const VariableNameMap &outputs,
      const AttributeMap &attrs) const {
    PADDLE_ENFORCE_NOT_NULL(inferer_);
    return (*inferer_)(inputs, outputs, attrs);
  }

  inline std::unordered_set<std::string> operator()(
      const imperative::NameVarBaseMap &inputs,
      const imperative::NameVarBaseMap &outputs,
      const AttributeMap &attrs) const {
    PADDLE_ENFORCE_NOT_NULL(inferer_);
    return (*inferer_)(inputs, outputs, attrs);
  }

  inline operator bool() const { return inferer_ != nullptr; }

  inline bool operator!() const { return inferer_ == nullptr; }

  inline void Set(const std::shared_ptr<NoNeedBufferVarsInference> &inferer) {
    PADDLE_ENFORCE_NOT_NULL(inferer);
    PADDLE_ENFORCE_EQ(inferer_, nullptr);
    inferer_ = inferer;
  }

 private:
  std::shared_ptr<NoNeedBufferVarsInference> inferer_;
};

}  // namespace framework
}  // namespace paddle
