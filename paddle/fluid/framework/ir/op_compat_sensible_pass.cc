/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>

#include "paddle/fluid/framework/ir/op_compat_sensible_pass.h"

namespace paddle {
namespace framework {
namespace ir {

AttrCompat& AttrCompat::IsStringIn(const std::set<std::string>& candidates) {
  conditions_.emplace_back([candidates](const Attribute& attr) -> bool {
    std::string value = BOOST_GET_CONST(std::string, attr);
    for (auto& str : candidates) {
      if (str == value) {
        return true;
      }
    }
    return false;
  });
  return *this;
}

AttrCompat& AttrCompat::IsStringMatch(
    const std::function<bool(const std::string&)>& func) {
  conditions_.emplace_back([func](const Attribute& attr) -> bool {
    std::string value = BOOST_GET_CONST(std::string, attr);
    return func(value);
  });
  return *this;
}

AttrCompat& AttrCompat::IsIntIn(const std::set<int>& candidates) {
  conditions_.emplace_back([candidates](const Attribute& attr) -> bool {
    int value = BOOST_GET_CONST(int, attr);
    return candidates.find(value) != candidates.end();
  });
  return *this;
}

//! Todo: append the definition.
AttrCompat& AttrCompat::IsLeftDefault() { return *this; }

bool AttrCompat::operator()(const OpDesc& op_desc) {
  if (!op_desc.HasAttr(attr_name_)) {
    return false;
  }
  const Attribute attr = op_desc.GetAttr(attr_name_);
  for (auto& func : conditions_) {
    if (!func(attr)) {
      return false;
    }
  }
  return true;
}

AttrCompat& AttrCompat::IsBoolEQ(bool v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    bool value = BOOST_GET_CONST(bool, attr);
    return value == v;
  });
  return *this;
}

InputOrOutputCompat& InputOrOutputCompat::IsTensor() {
  conditions_.emplace_back([](const std::vector<std::string>& input) -> bool {
    return input.size() == 1u;
  });
  return *this;
}

InputOrOutputCompat& InputOrOutputCompat::IsOptional() {
  optional_ = true;
  return *this;
}

bool InputOrOutputCompat::operator()(
    const std::vector<std::string>& input) const {
  if (input.empty()) return false;
  for (auto& func : conditions_) {
    if (!func(input)) {
      return false;
    }
  }
  return true;
}

AttrCompat& OpCompat::AddAttr(const std::string& attr_name) {
  attr_compats_.emplace_back(attr_name, this);
  return attr_compats_.back();
}

InputOrOutputCompat& OpCompat::AddInput(const std::string& name) {
  PADDLE_ENFORCE_EQ(input_compats_.find(name), input_compats_.end(),
                    platform::errors::InvalidArgument(
                        "The input with the same name has been added"));
  input_compats_.emplace(name, InputOrOutputCompat(name, this));
  return input_compats_.at(name);
}

InputOrOutputCompat& OpCompat::AddOutput(const std::string& name) {
  PADDLE_ENFORCE_EQ(output_compats_.find(name), output_compats_.end(),
                    platform::errors::InvalidArgument(
                        "The output with the same name has been added"));
  output_compats_.emplace(name, InputOrOutputCompat(name, this));
  return output_compats_.at(name);
}

bool OpCompat::Judge(const OpDesc& op_desc) {
  for (auto& attr_compat : attr_compats_) {
    if (!attr_compat(op_desc)) {
      return false;
    }
  }

  const VariableNameMap& inputs_map = op_desc.Inputs();
  for (auto& input_desc : inputs_map) {
    if (input_compats_.find(input_desc.first) == input_compats_.end()) {
      if (!input_desc.second.empty()) {
        return false;
      }
    }
  }
  for (auto& input_val : input_compats_) {
    if (inputs_map.find(input_val.first) == inputs_map.end()) {
      if (!input_val.second.Optional()) {
        return false;
      }
    } else {
      if (!input_val.second(inputs_map.at(input_val.first))) {
        return false;
      }
    }
  }

  const VariableNameMap& outputs_map = op_desc.Outputs();
  for (auto& output_desc : outputs_map) {
    if (output_compats_.find(output_desc.first) == output_compats_.end()) {
      if (!output_desc.second.empty()) {
        return false;
      }
    }
  }
  for (auto& output_val : output_compats_) {
    if (outputs_map.find(output_val.first) == outputs_map.end()) {
      if (!output_val.second.Optional()) {
        return false;
      }
    } else {
      if (!output_val.second(outputs_map.at(output_val.first))) {
        return false;
      }
    }
  }
  return true;
}

OpCompat& OpCompatSensiblePass::AddOpCompat(OpCompat&& op_compat) {
  std::string name = op_compat.Name();
  op_compat_judgers_[name].reset(new OpCompat(std::move(op_compat)));
  return *(op_compat_judgers_[name]);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
