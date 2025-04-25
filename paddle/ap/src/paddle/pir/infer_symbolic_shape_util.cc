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

#include "paddle/ap/include/paddle/pir/infer_symbolic_shape_util.h"
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/memory/guard.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/paddle/pir/infer_symbolic_shape_context_method_class.h"
#include "paddle/ap/include/paddle/pir/shape_or_data_method_class.h"

namespace ap::dialect {

namespace {

using Lambda = axpr::Lambda<axpr::CoreExpr>;

adt::Result<axpr::Value> GetInferCtxVal(
    pir::InferSymbolicShapeContext* infer_context) {
  return ap::paddle::GetPirInferSymbolicShapeContextClass().New(infer_context);
}

adt::Result<axpr::Value> GetApOpFacadeOpInputsVal(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  adt::List<axpr::Value> lst;
  lst->reserve(op->num_operands());
  for (int i = 0; i < op->num_operands(); ++i) {
    const auto& shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(i));
    lst->emplace_back(ap::paddle::GetPirShapeOrDataClass().New(shape_or_data));
  }
  return axpr::Value{lst};
}

adt::Result<axpr::Value> GetPdOpApFacadeOpInputsVal(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  ADT_CHECK(op->num_operands() == 1);
  const auto& shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  ADT_CHECK(
      shape_or_data.template isa<symbol::TensorListShapeOrDataDimExprs>());
  const auto& tensor_list_shape_or_data =
      shape_or_data.template dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  adt::List<axpr::Value> lst;
  lst->reserve(tensor_list_shape_or_data.size());
  for (const auto& elt_shape_or_data : tensor_list_shape_or_data) {
    lst->emplace_back(
        ap::paddle::GetPirShapeOrDataClass().New(elt_shape_or_data));
  }
  return axpr::Value{lst};
}

adt::Result<Lambda> CastStrToLambda(const std::string& serialized_attributes) {
  ADT_LET_CONST_REF(anf_expr,
                    axpr::MakeAnfExprFromJsonString(serialized_attributes));
  const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
  std::vector<ap::axpr::tVar<std::string>> args{};
  return Lambda{args, core_expr};
}

adt::Result<axpr::Value> Unserialize(const std::string& serialized_attributes) {
  ADT_LET_CONST_REF(lambda, CastStrToLambda(serialized_attributes));
  ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(ret_val, interpreter.Interpret(lambda, {}));
  ADT_CHECK(ret_val.template CastableTo<axpr::AttrMap<axpr::Value>>());
  return ret_val;
}

adt::Result<std::string> GetApOpApFacadeOpSerializedAttributes(
    pir::Operation* op) {
  const auto& iter =
      op->attributes().find("__original_serialized_attributes__");
  ADT_CHECK(iter != op->attributes().end());
  ADT_CHECK(iter->second.template isa<pir::StrAttribute>());
  const auto serialized_attributes =
      iter->second.template dyn_cast<pir::StrAttribute>().AsString();
  return serialized_attributes;
}

adt::Result<axpr::Value> GetApOpFacadeOpAttrsVal(pir::Operation* op) {
  ADT_LET_CONST_REF(serialized_attributes,
                    GetApOpApFacadeOpSerializedAttributes(op));
  return Unserialize(serialized_attributes);
}

adt::Result<std::string> GetPdOpApFacadeOpSerializedAttributes(
    pir::Operation* op) {
  const auto& iter = op->attributes().find("serialized_attributes");
  ADT_CHECK(iter != op->attributes().end());
  ADT_CHECK(iter->second.template isa<pir::StrAttribute>());
  const auto serialized_attributes =
      iter->second.template dyn_cast<pir::StrAttribute>().AsString();
  return serialized_attributes;
}

adt::Result<axpr::Value> GetPdOpApFacadeOpAttrsVal(pir::Operation* op) {
  ADT_LET_CONST_REF(serialized_attributes,
                    GetPdOpApFacadeOpSerializedAttributes(op));
  return Unserialize(serialized_attributes);
}

adt::Result<Lambda> GetInferSymbolicLambda() {
  static ap::axpr::Lambda<ap::axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd;
    const ap::axpr::AnfExpr anf_expr =
        lmd.Lambda({"infer_ctx", "inputs", "attrs"}, [](auto& ctx) {
          auto& infer_hooks =
              ctx.Var("import").Call(ctx.String("__ap_infer_hooks__"));
          auto& method = infer_hooks.Attr("infer_symbolic");
          auto& infer_ctx = ctx.Var("infer_ctx");
          auto& inputs = ctx.Var("inputs");
          auto& attrs = ctx.Var("attrs");
          auto& ret = method.Call(infer_ctx, inputs, attrs);
          return ret;
        });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  return lambda;
}

adt::Result<std::vector<symbol::TensorShapeOrDataDimExprs>>
InferOutputsShapeOrValue(const axpr::Value& infer_ctx_val,
                         const axpr::Value& inputs_val,
                         const axpr::Value& attrs_val) {
  ADT_LET_CONST_REF(lambda, GetInferSymbolicLambda());
  ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      ret_val,
      interpreter.Interpret(lambda, {infer_ctx_val, inputs_val, attrs_val}));
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(ret_val));
  std::vector<symbol::TensorShapeOrDataDimExprs> ret{};
  ADT_LET_CONST_REF(lst_size, lst.size());
  ret.reserve(lst_size);
  for (int i = 0; i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt_val, lst.at(i));
    ADT_LET_CONST_REF(shape_or_data,
                      elt_val.template CastTo<symbol::ShapeOrDataDimExprs>());
    ADT_CHECK(shape_or_data.template isa<symbol::TensorShapeOrDataDimExprs>());
    ret.emplace_back(
        shape_or_data.template dyn_cast<symbol::TensorShapeOrDataDimExprs>());
  }
  return ret;
}

adt::Result<adt::Ok> TryApOpFacadeOpInferSymbolicShape(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  ADT_LET_CONST_REF(infer_ctx_val, GetInferCtxVal(infer_context));
  ADT_LET_CONST_REF(inputs_val, GetApOpFacadeOpInputsVal(op, infer_context));
  ADT_LET_CONST_REF(attrs_val, GetApOpFacadeOpAttrsVal(op));
  ADT_LET_CONST_REF(
      outputs_shape_or_value,
      InferOutputsShapeOrValue(infer_ctx_val, inputs_val, attrs_val));
  ADT_CHECK(op->num_results() == outputs_shape_or_value.size());
  for (int i = 0; i < op->num_results(); ++i) {
    infer_context->SetShapeOrDataForValue(op->result(i),
                                          outputs_shape_or_value.at(i));
  }
  return adt::Ok{};
}

adt::Result<adt::Ok> TryPdOpApFacadeOpInferSymbolicShape(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  ADT_LET_CONST_REF(infer_ctx_val, GetInferCtxVal(infer_context));
  ADT_LET_CONST_REF(inputs_val, GetPdOpApFacadeOpInputsVal(op, infer_context));
  ADT_LET_CONST_REF(attrs_val, GetPdOpApFacadeOpAttrsVal(op));
  ADT_LET_CONST_REF(
      outputs_shape_or_value,
      InferOutputsShapeOrValue(infer_ctx_val, inputs_val, attrs_val));
  std::size_t num_outputs = 0;
  {
    const auto iter = op->attributes().find("num_outputs");
    ADT_CHECK(iter != op->attributes().end());
    const auto& num_outputs_attr = iter->second;
    ADT_CHECK(num_outputs_attr.isa<pir::Int64Attribute>());
    num_outputs = num_outputs_attr.dyn_cast<pir::Int64Attribute>().data();
  }
  ADT_CHECK(num_outputs == outputs_shape_or_value.size());
  ADT_CHECK(op->num_results(), 1);
  symbol::ShapeOrDataDimExprs shape_or_value{outputs_shape_or_value};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_or_value);
  return adt::Ok{};
}

}  // namespace

bool ApOpFacadeOpInferSymbolicShape(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  const auto& ret = TryApOpFacadeOpInferSymbolicShape(op, infer_context);
  bool success = !ret.HasError();
  PADDLE_ENFORCE(
      success,
      "ApOpFacadeOpInferSymbolicShape failed. \nTraceback (most recent call "
      "last):\n%s\n%s: %s. ",
      ret.GetError().CallStackToString(),
      ret.GetError().class_name(),
      ret.GetError().msg());
  return success;
}

bool PdOpApFacadeOpInferSymbolicShape(
    pir::Operation* op, pir::InferSymbolicShapeContext* infer_context) {
  const auto& ret = TryPdOpApFacadeOpInferSymbolicShape(op, infer_context);
  bool success = !ret.HasError();
  PADDLE_ENFORCE(
      success,
      "PdOpApFacadeOpInferSymbolicShape failed. \nTraceback (most recent call "
      "last):\n%s\n%s: %s. ",
      ret.GetError().CallStackToString(),
      ret.GetError().class_name(),
      ret.GetError().msg());
  return success;
}

}  // namespace ap::dialect
