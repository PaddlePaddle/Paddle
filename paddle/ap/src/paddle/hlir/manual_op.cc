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

#include <glog/logging.h>

#include "paddle/ap/include/paddle/hlir/manual_op.h"
#include "paddle/ap/include/paddle/pir/infer_symbolic_shape_util.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"

namespace ap::dialect {

const char* FacadeOp::attributes_name[FacadeOp::attributes_num] = {
    "custom_op_name", "infer_meta", "infer_symbolic"};

void FacadeOp::Build(pir::Builder& builder,             // NOLINT
                     pir::OperationArgument& argument,  // NOLINT
                     const std::vector<pir::Value>& inputs,
                     const pir::AttributeMap& attributes,
                     const std::vector<pir::Type>& output_types) {
  argument.inputs = inputs;
  argument.attributes = attributes;
  argument.output_types = output_types;
}

bool FacadeOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  return ApOpFacadeOpInferSymbolicShape(*this, infer_context);
}

void AddOp::Build(pir::Builder& builder,             // NOLINT
                  pir::OperationArgument& argument,  // NOLINT
                  pir::Value lhs,
                  pir::Value rhs) {
  argument.AddInput(lhs);
  argument.AddInput(rhs);
}

bool AddOp::InferSymbolicShape(pir::InferSymbolicShapeContext* infer_context) {
  const auto& lhs_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source(0));
  const auto& rhs_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source(1));
  PADDLE_ENFORCE_GT(lhs_shape_or_data.shape().size(),
                    rhs_shape_or_data.shape().size(),
                    phi::errors::InvalidArgument(
                        "lhs and rhs of ap_op.add should have same rank"));
  for (int i = 0; i < lhs_shape_or_data.shape().size(); ++i) {
    const auto& lhs_dim_expr = lhs_shape_or_data.shape().at(i);
    const auto& rhs_dim_expr = rhs_shape_or_data.shape().at(i);
    if (lhs_dim_expr != rhs_dim_expr) {
      infer_context->AddEqualCstr(lhs_dim_expr, rhs_dim_expr);
    }
  }
  infer_context->SetShapeOrDataForValue(result(0), rhs_shape_or_data);
  return true;
}

void UpSpiderOp::Build(pir::Builder& builder,             // NOLINT
                       pir::OperationArgument& argument,  // NOLINT
                       pir::Value lhs,
                       pir::Value rhs) {
  argument.AddInput(lhs);
  argument.AddInput(rhs);
}

void DownSpiderOp::Build(pir::Builder& builder,
                         pir::OperationArgument& argument,
                         pir::Value x) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
}

bool DownSpiderOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char*
    LoadFromRegisterOp::attributes_name[LoadFromRegisterOp::attributes_num] = {
        "type", "symbolic_shape_or_data", "name", "register_var_name"};

void LoadFromRegisterOp::Build(pir::Builder& builder,
                               pir::OperationArgument& argument,
                               pir::Type output_type,
                               const symbol::ShapeOrDataDimExprs& shape_or_data,
                               const std::string& name,
                               const std::string& register_var_name) {
  argument.inputs = {};
  argument.output_types = {output_type};
  argument.AddAttribute(
      "type", pir::TypeAttribute::get(pir::IrContext::Instance(), output_type));
  argument.AddAttribute("symbolic_shape_or_data",
                        pir::shape::SymbolAttribute::get(
                            pir::IrContext::Instance(), shape_or_data));
  argument.AddAttribute(
      "name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
  argument.AddAttribute(
      "register_var_name",
      pir::StrAttribute::get(pir::IrContext::Instance(), register_var_name));
}

bool LoadFromRegisterOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  const auto& symbolic_shape_or_data =
      this->attributes()
          .at("symbolic_shape_or_data")
          .dyn_cast<pir::shape::SymbolAttribute>()
          .data();
  infer_context->SetShapeOrDataForValue(result(0), symbolic_shape_or_data);
  return true;
}

const char*
    StoreToRegisterOp::attributes_name[StoreToRegisterOp::attributes_num] = {
        "name", "register_var_name"};

void StoreToRegisterOp::Build(pir::Builder& builder,
                              pir::OperationArgument& argument,
                              pir::Value x,
                              const std::string& name,
                              const std::string& register_var_name) {
  argument.inputs = {x};
  argument.AddAttribute(
      "name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
  argument.AddAttribute(
      "register_var_name",
      pir::StrAttribute::get(pir::IrContext::Instance(), register_var_name));
}

bool StoreToRegisterOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  return true;
}

const char*
    LoadFromGlobalOp::attributes_name[LoadFromGlobalOp::attributes_num] = {
        "index_func_unique_id"};

void LoadFromGlobalOp::Build(pir::Builder& builder,
                             pir::OperationArgument& argument,
                             pir::Value input,
                             const std::string& index_func_unique_id) {
  argument.inputs = {input};
  argument.output_types = {input.type()};
  argument.AddAttribute(
      "index_func_unique_id",
      pir::StrAttribute::get(pir::IrContext::Instance(), index_func_unique_id));
}

bool LoadFromGlobalOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char* StoreToGlobalOp::attributes_name[StoreToGlobalOp::attributes_num] =
    {"index_func_unique_id"};

void StoreToGlobalOp::Build(pir::Builder& builder,
                            pir::OperationArgument& argument,
                            pir::Value var,
                            pir::Value val,
                            const std::string& index_func_unique_id) {
  argument.inputs = {var, val};
  argument.output_types = {};
  argument.AddAttribute(
      "index_func_unique_id",
      pir::StrAttribute::get(pir::IrContext::Instance(), index_func_unique_id));
}

bool StoreToGlobalOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  return true;
}

}  // namespace ap::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::FacadeOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::AddOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::UpSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::DownSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromRegisterOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StoreToRegisterOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromGlobalOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StoreToGlobalOp);
