// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"

namespace paddle {
namespace dialect {

OpYamlInfoParser::OpYamlInfoParser(const OpInfoTuple& op_info_tuple)
    : op_info_tuple_(op_info_tuple) {
  parse();
}

bool OpYamlInfoParser::IsTensorAttribute(size_t index) const {
  PADDLE_ENFORCE_LT(
      index,
      InputInfo().size(),
      phi::errors::OutOfRange("Input index [%d] large than op input size [d]",
                              index,
                              InputInfo().size()));

  return InputInfo()[index].is_mutable_attribute;
}

size_t OpYamlInfoParser::InputTensorNumber() const {
  return input_tensor_number_;
}

const std::string& OpYamlInfoParser::AttrTypeName(
    const std::string& name) const {
  auto it = attr_info_.find(name);

  PADDLE_ENFORCE_NE(
      it,
      attr_info_.end(),
      phi::errors::NotFound("Not found [%s] in attribute map", name));
  return it->second.type_name;
}

const std::string& OpYamlInfoParser::TensorAttrTypeName(
    const std::string& name) const {
  auto it = input_info_.find(name);

  PADDLE_ENFORCE_NE(it,
                    input_info_.end(),
                    phi::errors::NotFound("Not found [%s] in input map", name));

  PADDLE_ENFORCE_EQ(
      it->second.is_mutable_attribute,
      true,
      phi::errors::PreconditionNotMet("[%s] MUST be a tensor attribute", name));
  return it->second.type_name;
}

const std::vector<std::string>& OpYamlInfoParser::TensorParams(
    bool is_kernel) const {
  if (is_kernel) {
    return kernel_fn_tensor_params_;
  } else {
    return infer_meta_tensor_params_;
  }
}
const std::vector<std::string>& OpYamlInfoParser::AttrParams(
    bool is_kernel) const {
  if (is_kernel) {
    return kernel_fn_attr_params_;
  } else {
    return infer_meta_attr_params_;
  }
}

const OpRunTimeInfo& OpYamlInfoParser::OpRuntimeInfo() const {
  return std::get<3>(op_info_tuple_);
}

const std::map<std::string, int>& OpYamlInfoParser::InputName2Id() const {
  return input_name2id_;
}

const std::map<std::string, int>& OpYamlInfoParser::OutputName2Id() const {
  return input_name2id_;
}

bool OpYamlInfoParser::HasInplace(const std::string& out_name) const {
  auto& inplace_info = std::get<3>(op_info_tuple_).inplace;
  for (size_t i = 0; i < inplace_info.size(); i++) {
    if (out_name == inplace_info[i].first) {
      return true;
    }
  }
  return false;
}

const std::string& OpYamlInfoParser::InplaceName(
    const std::string& out_name) const {
  auto& inplace_info = std::get<3>(op_info_tuple_).inplace;
  for (size_t i = 0; i < inplace_info.size(); i++) {
    if (out_name == inplace_info[i].first) {
      return inplace_info[i].second;
    }
  }
  PADDLE_THROW(phi::errors::PreconditionNotMet(
      "Can not find inplace input of [%s].", out_name));
}

void OpYamlInfoParser::parse() {
  auto input_info = std::get<0>(op_info_tuple_);

  int input_start_index = 0;
  for (size_t i = 0; i < input_info.size(); ++i) {
    input_name2id_[input_info[i].name] = input_start_index++;
    input_name_list_.push_back(input_info[i].name);
    input_info_[input_info[i].name] = input_info[i];
    if (!input_info[i].is_mutable_attribute) {
      input_tensor_number_++;
    }
  }

  auto attribute_info = std::get<1>(op_info_tuple_);
  for (size_t i = 0; i < attribute_info.size(); ++i) {
    attribute_name_list_.push_back(attribute_info[i].name);
    attr_info_[attribute_info[i].name] = attribute_info[i];
  }

  int output_start_index = 0;
  auto output_info = std::get<2>(op_info_tuple_);
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_name2id_[output_info[i].name] = output_start_index++;
    output_name_list_.push_back(output_info[i].name);
    output_info_[output_info[i].name] = output_info[i];
  }

  auto runtime_info = std::get<3>(op_info_tuple_);

  for (auto& name : runtime_info.infer_meta_param) {
    if (input_name2id_.count(name) && !input_info_[name].is_mutable_attribute) {
      infer_meta_tensor_params_.push_back(name);
    } else {
      infer_meta_attr_params_.push_back(name);
    }
  }

  for (auto& name : runtime_info.kernel_param) {
    if (input_name2id_.count(name) && !input_info_[name].is_mutable_attribute) {
      kernel_fn_tensor_params_.push_back(name);
    } else {
      kernel_fn_attr_params_.push_back(name);
    }
  }
}

}  // namespace dialect
}  // namespace paddle
