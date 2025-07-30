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

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/pir/include/core/dll_decl.h"

namespace paddle {
namespace dialect {

class OpYamlInfoParser {
 public:
  OpYamlInfoParser() = delete;

  TEST_API explicit OpYamlInfoParser(OpInfoTuple op_info_tuple,
                                     bool is_legacy_op = false);

  TEST_API bool IsTensorAttribute(size_t index) const;
  TEST_API size_t InputTensorNumber() const;

  TEST_API const std::string& AttrTypeName(const std::string& name) const;
  const std::string& TensorAttrTypeName(const std::string& name) const;

  const std::vector<std::string>& TensorParams(bool is_kernel = false) const;
  const std::vector<std::string>& AttrParams(bool is_kernel = false) const;
  const OpRunTimeInfo& OpRuntimeInfo() const;
  TEST_API const std::map<std::string, uint32_t>& InputName2Id() const;
  TEST_API const std::map<std::string, uint32_t>& OutputName2Id() const;

  const std::vector<uint32_t>& NoNeedBufferIds() const;

  const std::vector<std::string>& InputNames() const {
    return input_name_list_;
  }
  const std::vector<std::string>& AttributeNames() const {
    return attribute_name_list_;
  }
  const std::vector<std::string>& OutputNames() const {
    return output_name_list_;
  }

  const std::string& GetInputType(uint32_t input_id) const;

  const std::string& GetOutputType(uint32_t output_id) const;

  bool HasInplace(const std::string& out_name) const;

  const std::string& InplaceName(const std::string& out_name) const;

  std::unordered_map<uint32_t, uint32_t> GetInplaceIdMap() const;

  bool HasView(const std::string& out_name) const;

  const std::string& ViewName(const std::string& out_name) const;

  const std::string& GetOriginOpName() const;

  int GetTensorParamIndexByArgsName(const std::string& args_name) const;

 private:
  void parse();
  inline const std::vector<OpInputInfo>& InputInfo() const {
    return std::get<0>(op_info_tuple_);
  }

  OpInfoTuple op_info_tuple_;
  bool is_legacy_op_;

  // input info
  std::map<std::string, uint32_t> input_name2id_;
  std::vector<std::string> input_name_list_;
  std::map<std::string, OpInputInfo> input_info_;
  uint32_t input_tensor_number_{0};

  // no_need_buffer_ids
  std::vector<uint32_t> no_need_buffer_ids_;

  // attribute info
  std::vector<std::string> attribute_name_list_;
  std::map<std::string, OpAttributeInfo> attr_info_;

  // output info
  std::map<std::string, uint32_t> output_name2id_;
  std::vector<std::string> output_name_list_;
  std::map<std::string, OpOutputInfo> output_info_;

  // runtime info
  std::vector<std::string> infer_meta_tensor_params_;
  std::vector<std::string> infer_meta_attr_params_;
  std::vector<std::string> kernel_fn_tensor_params_;
  std::vector<std::string> kernel_fn_attr_params_;
};

}  // namespace dialect
}  // namespace paddle
