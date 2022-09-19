// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {

bool PluginArgumentMappingContext::HasInput(const std::string& name) const {
  auto inputs = op_desc_ptr_->Inputs();
  for (auto& i : inputs) {
    if (i.first == name && !i.second.empty()) return true;
  }
  return false;
}

bool PluginArgumentMappingContext::HasOutput(const std::string& name) const {
  auto outputs = op_desc_ptr_->Outputs();
  for (auto& i : outputs) {
    if (i.first == name && !i.second.empty()) return true;
  }
  return false;
}

bool PluginArgumentMappingContext::HasAttr(const std::string& name) const {
  return op_desc_ptr_->HasAttr(name);
}

paddle::any PluginArgumentMappingContext::Attr(
    const std::string& attr_name) const {
  auto attr_type = op_desc_ptr_->GetAttrType(attr_name);
  switch (attr_type) {
    case framework::proto::AttrType::INT: {
      return PADDLE_GET_CONST(int, op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::FLOAT: {
      return PADDLE_GET_CONST(float, op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::STRING: {
      return PADDLE_GET_CONST(std::string, op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::INTS: {
      return PADDLE_GET_CONST(std::vector<int>,
                              op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::FLOATS: {
      return PADDLE_GET_CONST(std::vector<float>,
                              op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::STRINGS: {
      return PADDLE_GET_CONST(std::vector<std::string>,
                              op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::BOOLEAN: {
      return PADDLE_GET_CONST(bool, op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    case framework::proto::AttrType::BOOLEANS: {
      return PADDLE_GET_CONST(std::vector<bool>,
                              op_desc_ptr_->GetAttr(attr_name));
      break;
    };
    default: {
      LOG(ERROR) << "Can't conver op's attribute [" << attr_name
                 << "] to paddle any.";
    }
  }
  return paddle::any();
}

size_t PluginArgumentMappingContext::InputSize(const std::string& name) const {
  return op_desc_ptr_->Inputs().at(name).size();
}
size_t PluginArgumentMappingContext::OutputSize(const std::string& name) const {
  return op_desc_ptr_->Outputs().at(name).size();
}
bool PluginArgumentMappingContext::IsDenseTensorInput(
    const std::string& name) const {
  return false;
}
bool PluginArgumentMappingContext::IsDenseTensorInputs(
    const std::string& name) const {
  return false;
}
bool PluginArgumentMappingContext::IsSelectedRowsInput(
    const std::string& name) const {
  return false;
}
bool PluginArgumentMappingContext::IsSelectedRowsInputs(
    const std::string& name) const {
  return false;
}
bool PluginArgumentMappingContext::IsDenseTensorVectorInput(
    const std::string& name) const {
  return false;
}

bool PluginArgumentMappingContext::IsDenseTensorOutput(
    const std::string& name) const {
  return false;
}
bool PluginArgumentMappingContext::IsSelectedRowsOutput(
    const std::string& name) const {
  return false;
}
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
