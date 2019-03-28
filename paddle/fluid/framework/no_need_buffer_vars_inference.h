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

#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"

namespace paddle {
namespace framework {

class NoNeedBufferVarsInference {
 public:
  NoNeedBufferVarsInference(const VariableNameMap &inputs,
                            const VariableNameMap &outputs,
                            const AttributeMap &attrs)
      : inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  virtual ~NoNeedBufferVarsInference() = default;

  const VariableNameMap &Inputs() const { return inputs_; }

  const VariableNameMap &Outputs() const { return outputs_; }

  const AttributeMap &Attrs() const { return attrs_; }

  virtual std::unordered_set<std::string> operator()() const = 0;

 private:
  const VariableNameMap &inputs_;
  const VariableNameMap &outputs_;
  const AttributeMap &attrs_;
};

#define DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(class_type, ...)               \
  class class_type : public ::paddle::framework::NoNeedBufferVarsInference { \
   public:                                                                   \
    using ::paddle::framework::NoNeedBufferVarsInference::                   \
        NoNeedBufferVarsInference;                                           \
                                                                             \
    std::unordered_set<std::string> operator()() const override {            \
      return {__VA_ARGS__};                                                  \
    }                                                                        \
  }

}  // namespace framework
}  // namespace paddle
