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
  auto it = map_attr_info_.find(name);

  PADDLE_ENFORCE_NE(
      it,
      map_attr_info_.end(),
      phi::errors::NotFound("Not found [%s] in attribute map", name));
  return it->second.type_name;
}

const std::vector<std::string>& OpYamlInfoParser::InferMetaTensorParams()
    const {
  return vec_infer_meta_tensor_params_;
}
const std::vector<std::string>& OpYamlInfoParser::InferMetaAttrParams() const {
  return vec_infer_meta_attr_params_;
}
const std::vector<std::string>& OpYamlInfoParser::KernelFnTensorParams() const {
  return vec_kernel_fn_tensor_params_;
}
const std::vector<std::string>& OpYamlInfoParser::KernelFnAttrParams() const {
  return vec_kernel_fn_attr_params_;
}

const OpRunTimeInfo& OpYamlInfoParser::OpRuntimeInfo() const {
  return std::get<3>(op_info_tuple_);
}

const std::map<std::string, int>& OpYamlInfoParser::Name2Id() const {
  return map_name2id_;
}

void OpYamlInfoParser::parse() {
  auto input_info = std::get<0>(op_info_tuple_);

  int start_index = 0;

  for (size_t i = 0; i < input_info.size(); ++i) {
    map_name2id_[input_info[i].name] = start_index++;

    if (!input_info[i].is_mutable_attribute) {
      input_tensor_number_++;
    }

    map_input_info_[input_info[i].name] = input_info[i];
  }

  auto attribute_info = std::get<1>(op_info_tuple_);
  for (size_t i = 0; i < attribute_info.size(); ++i) {
    map_attr_info_[attribute_info[i].name] = attribute_info[i];
  }

  auto output_info = std::get<2>(op_info_tuple_);

  for (size_t i = 0; i < output_info.size(); ++i) {
    map_output_info_[output_info[i].name] = output_info[i];
  }

  auto runtime_info = std::get<3>(op_info_tuple_);

  for (auto& name : runtime_info.infer_meta_param) {
    if (map_name2id_.count(name)) {
      vec_infer_meta_tensor_params_.push_back(name);
    } else {
      vec_infer_meta_attr_params_.push_back(name);
    }
  }

  for (auto& name : runtime_info.kernel_param) {
    if (map_name2id_.count(name)) {
      vec_kernel_fn_tensor_params_.push_back(name);
    } else {
      vec_kernel_fn_attr_params_.push_back(name);
    }
  }
}

}  // namespace dialect
}  // namespace paddle
