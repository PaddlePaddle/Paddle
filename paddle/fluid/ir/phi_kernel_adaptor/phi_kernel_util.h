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

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"

#include "glog/logging.h"

namespace ir {

void build_scope(ir::Block* block,
                 paddle::framework::Scope* scope,
                 std::unordered_map<ir::Value, std::string>* name_map);

template <typename T>
void build_context(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    const OpInfoTuple& op_yaml_info,
    T* ctx,
    bool is_infer_meta = true,
    std::map<std::string, std::vector<int>>* input_map = nullptr,
    std::map<std::string, std::vector<int>>* output_map = nullptr) {
  // inputs include input and mutable attributes
  auto input_info = std::get<0>(op_yaml_info);
  std::map<std::string, size_t> input_index_map;
  std::map<std::string, std::string> mutable_attr_type_map;
  int input_index = 0;
  for (auto& t : input_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    input_index_map[t.name] = input_index++;
    if (t.is_mutable_attribute) {
      mutable_attr_type_map[t.name] = t.type_name;
    }
  }

  auto attr_info = std::get<1>(op_yaml_info);
  std::map<std::string, std::string> attr_type_map;
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }

  auto attr_map = op->attributes();
  auto runtime_info = std::get<3>(op_yaml_info);

  // int input_index = 0;
  std::vector<std::string> vec_param_list;
  if (is_infer_meta) {
    vec_param_list = runtime_info.infer_meta_param;
  } else {
    vec_param_list = runtime_info.kernel_param;
  }
  for (auto& t : vec_param_list) {
    if (input_index_map.count(t)) {
      // get information from input
      ir::Value ptr = op->GetOperandByIndex(input_index_map[t]).source();
      auto in_var_name = name_map.at(ptr);

      if (!is_infer_meta && input_map) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        size_t tmp_id = std::atol(in_var_name.substr(4, 100).c_str());
        (*input_map)[std::to_string(input_index_map.at(t))].push_back(tmp_id);
      }

      if (mutable_attr_type_map.count(t)) {
        VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t"
                << in_var_name;
        if (mutable_attr_type_map[t] == "paddle::dialect::IntArrayAttribute") {
          ctx->EmplaceBackAttr(phi::IntArray(
              *(scope->Var(in_var_name)->GetMutable<phi::DenseTensor>())));
        } else if (mutable_attr_type_map[t] ==
                   "paddle::dialect::ScalarAttribute") {
          ctx->EmplaceBackAttr(phi::Scalar(
              *(scope->Var(in_var_name)->GetMutable<phi::DenseTensor>())));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  mutable_attr_type_map[t]));
        }

      } else {
        VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;
        ctx->EmplaceBackInput(
            scope->Var(in_var_name)->GetMutable<phi::DenseTensor>());
      }
    }

    if (attr_type_map.count(t)) {
      auto type_name = attr_type_map[t];
      if (type_name == "paddle::dialect::IntArrayAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
      } else if (type_name == "paddle::dialect::DataTypeAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
      } else if (type_name == "ir::Int32_tAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<ir::Int32_tAttribute>().data());
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else if (type_name == "paddle::dialect::ScalarAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
      VLOG(6) << "ctx->EmplaceBackAttr: " << t;
    }
  }

  ir::Value out_ptr = op->GetResultByIndex(0);
  auto name = name_map.at(out_ptr);

  ctx->EmplaceBackOutput(scope->Var(name)->GetMutable<phi::DenseTensor>());

  if (!is_infer_meta && output_map) {
    // only deal with single input for now, [todo] need support multi input like
    // concat
    size_t tmp_id = std::atol(name.substr(4, 100).c_str());
    (*output_map)["out"].push_back(tmp_id);
  }
}

}  // namespace ir
