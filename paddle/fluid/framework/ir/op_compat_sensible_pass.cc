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
#include "paddle/fluid/framework/op_info.h"
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
AttrCompat& AttrCompat::IsLeftDefault() {
  const std::string& op_name = op_compat_->Name();
  if (!OpInfoMap::Instance().Has(op_name)) {
    VLOG(3) << "Op (" << op_name << ") is not registered!";
    conditions_.emplace_back([](const Attribute& attr) { return false; });
    return *this;
  }
  const OpInfo& op_info = OpInfoMap::Instance().Get(op_name);
  const AttributeMap attrs = op_info.Checker()->GetAttrsDefaultValuesMap();
  if (attrs.find(attr_name_) == attrs.end()) {
    VLOG(3) << "Op (" << op_name << ") has no default attr:" << attr_name_;
    conditions_.emplace_back([](const Attribute& attr) { return false; });
  } else {
    Attribute default_attr = attrs.at(attr_name_);
    conditions_.emplace_back([default_attr](const Attribute& attr) -> bool {
      return attr == default_attr;
    });
  }
  return *this;
}

bool AttrCompat::operator()(const OpDesc& op_desc) {
  if (conditions_.empty()) {
    return true;
  }
  if (!op_desc.HasAttr(attr_name_)) {
    return optional_;
  }
  const Attribute attr = op_desc.GetAttr(attr_name_);
  for (auto& func : conditions_) {
    if (!func(attr)) {
      return false;
    }
  }
  return true;
}
AttrCompat& AttrCompat::IsOptional() {
  optional_ = true;
  return *this;
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
  PADDLE_ENFORCE_EQ(
      attr_compats_.find(attr_name), attr_compats_.end(),
      platform::errors::InvalidArgument(
          "The attrubute compat with the same name has been added"));
  attr_compats_.emplace(attr_name, AttrCompat(attr_name, this));
  return attr_compats_.at(attr_name);
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
  for (auto& attr_map : op_desc.GetAttrMap()) {
    if (attr_compats_.find(attr_map.first) == attr_compats_.end()) {
      if (!AttrCompat(attr_map.first, this).IsLeftDefault()(op_desc)) {
        VLOG(3) << "The Attr(" << attr_map.first << ") of Op (" << op_name_
                << ") not reigistered in OpCompat, not equal to default value!";
        return false;
      }
    }
  }
  for (auto& attr_compat : attr_compats_) {
    if (!attr_compat.second(op_desc)) {
      VLOG(3) << " Check the Attr(" << attr_compat.first << ") of Op("
              << op_name_ << ") failed!";
      return false;
    }
  }

  const VariableNameMap& inputs_map = op_desc.Inputs();
  for (auto& input_desc : inputs_map) {
    if (input_compats_.find(input_desc.first) == input_compats_.end()) {
      if (!input_desc.second.empty()) {
        VLOG(3) << "The Input (" << input_desc.first << ") of Operator ("
                << op_name_ << ") not reigistered in OpCompat!";
        return false;
      }
    }
  }
  for (auto& input_val : input_compats_) {
    if (inputs_map.find(input_val.first) == inputs_map.end()) {
      if (!input_val.second.Optional()) {
        VLOG(3) << "The No optional Input (" << input_val.first
                << ") of Operator (" << op_name_ << ") not find in op_desc!";
        return false;
      }
    } else {
      if (!input_val.second(inputs_map.at(input_val.first))) {
        VLOG(3) << "The Input (" << input_val.first << ") of Operator ("
                << op_name_ << ") compat check failed!";
        return false;
      }
    }
  }

  const VariableNameMap& outputs_map = op_desc.Outputs();
  for (auto& output_desc : outputs_map) {
    if (output_compats_.find(output_desc.first) == output_compats_.end()) {
      if (!output_desc.second.empty()) {
        VLOG(3) << "The Output (" << output_desc.first << ") of Operator ("
                << op_name_ << ") not reigistered in OpCompat!";
        return false;
      }
    }
  }
  for (auto& output_val : output_compats_) {
    if (outputs_map.find(output_val.first) == outputs_map.end()) {
      if (!output_val.second.Optional()) {
        VLOG(3) << "The No optional Output (" << output_val.first
                << ") of Operator (" << op_name_ << ") not find in op_desc!";
        return false;
      }
    } else {
      if (!output_val.second(outputs_map.at(output_val.first))) {
        VLOG(3) << "The Output (" << output_val.first << ") of Operator ("
                << op_name_ << ") compat check failed!";
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
