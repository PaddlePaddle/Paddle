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

#pragma once

#include "paddle/fluid/ir/interface/op_yaml_info.h"

namespace paddle {
namespace dialect {

class OpYamlInfoParser {
 public:
  OpYamlInfoParser() = delete;

  explicit OpYamlInfoParser(const OpInfoTuple& op_info_tuple);

  bool IsTensorAttribute(size_t index) const;
  size_t InputTensorNumber() const;

  const std::string& AttrTypeName(const std::string& name) const;
  const std::string& TensorAttrTypeName(const std::string& name) const;

  const std::vector<std::string>& TensorParams(bool is_kernel = false) const;
  const std::vector<std::string>& AttrParams(bool is_kernel = false) const;
  const OpRunTimeInfo& OpRuntimeInfo() const;
  const std::map<std::string, int>& Name2Id() const;

 private:
  void parse();
  inline const std::vector<OpInputInfo>& InputInfo() const {
    return std::get<0>(op_info_tuple_);
  }

  OpInfoTuple op_info_tuple_;

  std::map<std::string, int> name2id_;

  std::map<std::string, OpInputInfo> input_info_;
  std::map<std::string, OpAttributeInfo> attr_info_;
  std::map<std::string, OpOutputInfo> output_info_;

  std::vector<std::string> infer_meta_tensor_params_;
  std::vector<std::string> infer_meta_attr_params_;
  std::vector<std::string> kernel_fn_tensor_params_;
  std::vector<std::string> kernel_fn_attr_params_;

  int input_tensor_number_{0};
};

}  // namespace dialect
}  // namespace paddle
