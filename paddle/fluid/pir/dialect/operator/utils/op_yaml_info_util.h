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

#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace paddle {
namespace dialect {

struct OpInputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool no_need_buffer = false;
  bool is_mutable_attribute = false;
  /***
   * "with_grad_semantic" represents whether the input of the OP has gradient
   * semantics. For example, gather op contains three inputs (x, index, axis),
   * but the backward op gather_grad calculates only the gradient with respect
   * to x. Therefore, for gather op, only x has gradient semantics.
   * The "with_grad_semantic" fields in OpInputInfo for x is true,
   * and "with_grad_semantic" fields in OpInputInfo for index and axis are both
   * false.
   */
  bool with_grad_semantic = true;

  OpInputInfo() = default;
  OpInputInfo(const OpInputInfo& input_info) = default;

  OpInputInfo(const std::string& name,
              const std::string& type_name,
              bool optional,
              bool no_need_buffer,
              bool is_mutable_attribute,
              bool with_grad_semantic)
      : name(name),
        type_name(type_name),
        optional(optional),
        no_need_buffer(no_need_buffer),
        is_mutable_attribute(is_mutable_attribute),
        with_grad_semantic(with_grad_semantic) {}
};

struct OpOutputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool intermediate = false;
  OpOutputInfo() = default;
  OpOutputInfo(const OpOutputInfo& output_info) = default;
  OpOutputInfo(const std::string& name,
               const std::string& type_name,
               bool optional,
               bool intermediate)
      : name(name),
        type_name(type_name),
        optional(optional),
        intermediate(intermediate) {}
};

struct OpAttributeInfo {
  std::string name;
  std::string type_name;
  std::string data_type;
  OpAttributeInfo() = default;
  OpAttributeInfo(const OpAttributeInfo& attr_info) = default;
  OpAttributeInfo(const std::string& name,
                  const std::string& type_name,
                  const std::string& data_type)
      : name(name), type_name(type_name), data_type(data_type) {}
};

struct OpRunTimeInfo {
  std::string infer_meta_func;
  std::vector<std::string> infer_meta_param;
  std::string kernel_func;
  std::vector<std::string> kernel_param;
  std::vector<std::string> kernel_key_dtype;
  std::vector<std::string> kernel_key_backend;
  std::vector<std::pair<std::string, std::string>> inplace;
  std::vector<std::pair<std::string, std::string>> view;
  std::vector<std::string> extra_args;
  std::vector<std::string> skip_transform_inputs;
  pir::AttributeMap extra_args_default_value;
  std::vector<std::string> data_format_tensors;
  bool is_onednn_only = false;
  bool dynamic_fallback = false;
  OpRunTimeInfo() = default;

  OpRunTimeInfo(const std::string& infer_meta_func,
                const std::vector<std::string>& infer_meta_param,
                const std::string& kernel_func,
                const std::vector<std::string>& kernel_param,
                const std::vector<std::string>& dtype,
                const std::vector<std::string>& backend,
                const std::vector<std::pair<std::string, std::string>>& inplace,
                const std::vector<std::pair<std::string, std::string>>& view,
                const std::vector<std::string>& extra_args = {},
                const std::vector<std::string>& skip_transform_inputs = {},
                const pir::AttributeMap& extra_args_default_value = {{}},
                const std::vector<std::string>& data_format_tensors = {},
                bool is_onednn_only = false,
                bool dynamic_fallback = false)
      : infer_meta_func(infer_meta_func),
        infer_meta_param(infer_meta_param),
        kernel_func(kernel_func),
        kernel_param(kernel_param),
        kernel_key_dtype(dtype),
        kernel_key_backend(backend),
        inplace(inplace),
        view(view),
        extra_args(extra_args),
        skip_transform_inputs(skip_transform_inputs),
        extra_args_default_value(extra_args_default_value),
        data_format_tensors(data_format_tensors),
        is_onednn_only(is_onednn_only),
        dynamic_fallback(dynamic_fallback) {}
};

}  // namespace dialect
}  // namespace paddle
