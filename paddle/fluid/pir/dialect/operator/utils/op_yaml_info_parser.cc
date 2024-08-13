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

#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"

#include <utility>
#include "paddle/phi/core/enforce.h"

namespace paddle::dialect {

OpYamlInfoParser::OpYamlInfoParser(OpInfoTuple op_info_tuple, bool is_legacy_op)
    : op_info_tuple_(std::move(op_info_tuple)), is_legacy_op_(is_legacy_op) {
  parse();
}

bool OpYamlInfoParser::IsTensorAttribute(size_t index) const {
  PADDLE_ENFORCE_LT(index,
                    InputInfo().size(),
                    common::errors::OutOfRange(
                        "Input index [%d] large than op input size [%d]",
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
      common::errors::NotFound("Not found [%s] in attribute map", name));
  return it->second.type_name;
}

const std::string& OpYamlInfoParser::TensorAttrTypeName(
    const std::string& name) const {
  auto it = input_info_.find(name);

  PADDLE_ENFORCE_NE(
      it,
      input_info_.end(),
      common::errors::NotFound("Not found [%s] in input map", name));

  PADDLE_ENFORCE_EQ(it->second.is_mutable_attribute,
                    true,
                    common::errors::PreconditionNotMet(
                        "[%s] MUST be a tensor attribute", name));
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

const std::map<std::string, uint32_t>& OpYamlInfoParser::InputName2Id() const {
  return input_name2id_;
}

const std::map<std::string, uint32_t>& OpYamlInfoParser::OutputName2Id() const {
  return output_name2id_;
}

const std::string& OpYamlInfoParser::GetInputType(uint32_t input_id) const {
  PADDLE_ENFORCE_EQ(input_id < input_name_list_.size(),
                    true,
                    common::errors::NotFound("Exceeding maximum input id %d",
                                             input_name_list_.size()));
  std::string input_name = input_name_list_[input_id];
  auto it = input_info_.find(input_name);
  return it->second.type_name;
}

const std::string& OpYamlInfoParser::GetOutputType(uint32_t output_id) const {
  PADDLE_ENFORCE_EQ(output_id < output_name_list_.size(),
                    true,
                    common::errors::NotFound("Exceeding maximum output id %d",
                                             output_name_list_.size()));
  std::string output_name = output_name_list_[output_id];
  auto it = output_info_.find(output_name);
  return it->second.type_name;
}

const std::vector<uint32_t>& OpYamlInfoParser::NoNeedBufferIds() const {
  return no_need_buffer_ids_;
}

bool OpYamlInfoParser::HasInplace(const std::string& out_name) const {
  auto& inplace_info = std::get<3>(op_info_tuple_).inplace;
  for (const auto& info : inplace_info) {
    if (out_name == info.first) {
      return true;
    }
  }
  return false;
}

const std::string& OpYamlInfoParser::InplaceName(
    const std::string& out_name) const {
  auto& inplace_info = std::get<3>(op_info_tuple_).inplace;
  for (const auto& info : inplace_info) {
    if (out_name == info.first) {
      return info.second;
    }
  }
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "Can not find inplace input of [%s].", out_name));
}

std::unordered_map<uint32_t, uint32_t> OpYamlInfoParser::GetInplaceIdMap()
    const {
  std::unordered_map<uint32_t, uint32_t> inplace_id_map;
  auto& inplace_info = std::get<3>(op_info_tuple_).inplace;
  for (const auto& info : inplace_info) {
    inplace_id_map[OutputName2Id().at(info.first)] =
        InputName2Id().at(info.second);
  }
  return inplace_id_map;
}

bool OpYamlInfoParser::HasView(const std::string& out_name) const {
  auto& view_info = std::get<3>(op_info_tuple_).view;
  for (const auto& i : view_info) {
    if (out_name == i.first) {
      return true;
    }
  }
  return false;
}

const std::string& OpYamlInfoParser::ViewName(
    const std::string& out_name) const {
  auto& view_info = std::get<3>(op_info_tuple_).view;
  for (const auto& i : view_info) {
    if (out_name == i.first) {
      return i.second;
    }
  }
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "Can not find inplace input of [%s].", out_name));
}

void OpYamlInfoParser::parse() {
  auto input_info = std::get<0>(op_info_tuple_);

  for (size_t i = 0; i < input_info.size(); ++i) {
    input_name2id_[input_info[i].name] = i;
    input_name_list_.push_back(input_info[i].name);
    input_info_[input_info[i].name] = input_info[i];
    if (!input_info[i].is_mutable_attribute) {
      input_tensor_number_++;
    }
    if (input_info[i].no_need_buffer) {
      no_need_buffer_ids_.push_back(i);
    }
  }

  auto attribute_info = std::get<1>(op_info_tuple_);
  for (auto& info : attribute_info) {
    attribute_name_list_.push_back(info.name);
    attr_info_[info.name] = info;
  }

  auto output_info = std::get<2>(op_info_tuple_);
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_name2id_[output_info[i].name] = i;
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
    if ((input_name2id_.count(name) &&
         (!input_info_[name].is_mutable_attribute)) ||
        (is_legacy_op_ && input_info_[name].is_mutable_attribute)) {
      kernel_fn_tensor_params_.push_back(name);
    } else {
      kernel_fn_attr_params_.push_back(name);
    }
  }
}

const std::string& OpYamlInfoParser::GetOriginOpName() const {
  return std::get<4>(op_info_tuple_);
}

int OpYamlInfoParser::GetTensorParamIndexByArgsName(
    const std::string& args_name) const {
  const auto& iter = std::find(kernel_fn_tensor_params_.begin(),
                               kernel_fn_tensor_params_.end(),
                               args_name);
  if (iter != kernel_fn_tensor_params_.end()) {
    return std::distance(kernel_fn_tensor_params_.begin(), iter);  // NOLINT
  } else {
    return -1;
  }
}

}  // namespace paddle::dialect
