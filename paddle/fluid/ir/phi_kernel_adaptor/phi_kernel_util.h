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

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
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
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/phi/core/infermeta_utils.h"

#include "glog/logging.h"

namespace ir {

void BuildScope(ir::Block* block,
                paddle::framework::Scope* scope,
                std::unordered_map<ir::Value, std::string>* name_map);

template <typename Context,
          typename InType,
          typename OutType,
          typename ListType,
          bool is_kernel>
void BuildPhiContext(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    Context* ctx,
    std::map<std::string, std::vector<int>>* input_map = nullptr,
    std::map<std::string, std::vector<int>>* output_map = nullptr) {
  // inputs include input and mutable attributes

  auto attr_map = op->attributes();

  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(is_kernel);

  auto& name2id = op_yaml_info.Name2Id();
  for (auto& t : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", t));
    auto index = op_yaml_info.Name2Id().at(t);
    ir::Value ptr = op->operand(index);
    auto in_var_name = name_map.at(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(scope->FindLocalVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));

    auto var = scope->Var(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
      ctx->EmplaceBackInput(InType(tensor_in));
    } else if (var->IsType<paddle::framework::TensorRefArray>()) {
      ListType inputs;
      auto& tensor_array = var->Get<paddle::framework::TensorRefArray>();
      for (size_t i = 0; i < tensor_array.size(); ++i) {
        inputs.emplace_back(InType(tensor_array[i]));
      }
      ctx->EmplaceBackInputs(inputs);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                              var->Type()));
    }
  }

  auto& vec_kernel_fn_attr_params = op_yaml_info.AttrParams(is_kernel);
  for (auto& t : vec_kernel_fn_attr_params) {
    if (name2id.count(t)) {
      // tensor attribute, get information from input
      ir::Value ptr = op->operand(name2id.at(t));

      auto in_var_name = name_map.at(ptr);
      if (input_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10

        size_t tmp_id = std::atol(in_var_name.substr(4, 100).c_str());
        (*input_map)[std::to_string(name2id.at(t))].push_back(tmp_id);
      }

      auto& tensor_attr_type = op_yaml_info.TensorAttrTypeName(t);
      VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t" << in_var_name;
      if (tensor_attr_type == "paddle::dialect::IntArrayAttribute") {
        phi::Attribute r1 =
            phi::TensorRef(&(scope->Var(in_var_name)->Get<phi::DenseTensor>()));
        ctx->EmplaceBackAttr(r1);
      } else if (tensor_attr_type == "paddle::dialect::ScalarAttribute") {
        phi::Attribute r1 =
            phi::TensorRef(&(scope->Var(in_var_name)->Get<phi::DenseTensor>()));

        ctx->EmplaceBackAttr(r1);
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                tensor_attr_type));
      }

      continue;
    }

    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "paddle::dialect::IntArrayAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::DataTypeAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
    } else if (attr_type_name == "ir::Int32Attribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::Int32Attribute>().data());
    } else if (attr_type_name == "ir::FloatAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::FloatAttribute>().data());
    } else if (attr_type_name == "ir::BoolAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::BoolAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::PlaceAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::ScalarAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                              attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // TODO(phlrain): use var type instead of op name
  if (op->attributes().count("op_name") &&
      (op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data() ==
       "pd.fetch")) {
    // process fetch op
    auto fetch_var = scope->Var("fetch");
    auto* fetch_list = fetch_var->GetMutable<paddle::framework::FetchList>();
    auto* out_tensor = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(0)));
    ctx->EmplaceBackOutput(out_tensor);
  } else {
    for (size_t i = 0; i < op->num_results(); ++i) {
      ir::Value out_ptr = op->result(i);
      auto name = name_map.at(out_ptr);
      ctx->EmplaceBackOutput(OutType(const_cast<phi::DenseTensor*>(
          &(scope->Var(name)->Get<phi::DenseTensor>()))));

      if (output_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10

        size_t tmp_id = std::atol(name.substr(4, 100).c_str());
        (*output_map)["out"].push_back(tmp_id);
      }
    }
  }
}

}  // namespace ir
