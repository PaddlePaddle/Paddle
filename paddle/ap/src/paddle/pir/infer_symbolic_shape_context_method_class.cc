// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pir/infer_symbolic_shape_context_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/dim_expr.h"
#include "paddle/ap/include/paddle/pir/type_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::paddle {

namespace {

adt::Result<axpr::Value> Max(const axpr::Value& self_val,
                             const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  symbol::DimExpr ret{symbol::Max<symbol::DimExpr>{
      symbol::List<symbol::DimExpr>{lhs, rhs},
  }};
  return axpr::GetDimExprClass<axpr::Value>().New(symbol::SimplifyDimExpr(ret));
}

adt::Result<axpr::Value> Min(const axpr::Value& self_val,
                             const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  symbol::DimExpr ret{symbol::Min<symbol::DimExpr>{
      symbol::List<symbol::DimExpr>{lhs, rhs},
  }};
  return axpr::GetDimExprClass<axpr::Value>().New(symbol::SimplifyDimExpr(ret));
}

adt::Result<axpr::Value> Broadcast(const axpr::Value& self_val,
                                   const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  symbol::DimExpr ret{symbol::Broadcast<symbol::DimExpr>{
      symbol::List<symbol::DimExpr>{lhs, rhs},
  }};
  return axpr::GetDimExprClass<axpr::Value>().New(symbol::SimplifyDimExpr(ret));
}

adt::Result<axpr::Value> NewSymbolicName(const axpr::Value& self_val,
                                         const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 0);
  return self->GetNextSymName();
}

adt::Result<axpr::Value> AddEqualCstr(const axpr::Value& self_val,
                                      const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  self->AddEqualCstr(lhs, rhs);
  return adt::Nothing{};
}

adt::Result<axpr::Value> IsEqual(const axpr::Value& self_val,
                                 const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  return self->IsEqual(lhs, rhs);
}

adt::Result<axpr::Value> AddGreatThanOneCstr(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(operand, args.at(0).template CastTo<symbol::DimExpr>());
  self->AddGreatThanOneCstr(operand);
  return adt::Nothing{};
}

adt::Result<axpr::Value> IsGreatThanOne(const axpr::Value& self_val,
                                        const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(operand, args.at(0).template CastTo<symbol::DimExpr>());
  return self->IsGreatThanOne(operand);
}

adt::Result<axpr::Value> AddBroadcastableCstr(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  self->AddBroadcastableCstr(lhs, rhs);
  return adt::Nothing{};
}

adt::Result<axpr::Value> IsBroadcastable(const axpr::Value& self_val,
                                         const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(
      self, self_val.template CastTo<pir::InferSymbolicShapeContext*>());
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(lhs, args.at(0).template CastTo<symbol::DimExpr>());
  ADT_LET_CONST_REF(rhs, args.at(1).template CastTo<symbol::DimExpr>());
  return self->IsBroadcastable(lhs, rhs);
}

}  // namespace

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirInferSymbolicShapeContextClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirInferSymbolicShapeContext", [&](const auto& Yield) {
        Yield("max", &Max);
        Yield("min", &Min);
        Yield("broadcast", &Broadcast);
        Yield("new_symbolic_name", &NewSymbolicName);
        Yield("add_equal_cstr", &AddEqualCstr);
        Yield("is_equal", &IsEqual);
        Yield("add_greater_than_one_cstr", &AddGreatThanOneCstr);
        Yield("is_greater_than_one", &IsGreatThanOne);
        Yield("add_broadcastable_cstr", &AddBroadcastableCstr);
        Yield("is_broadcastable", &IsBroadcastable);
      }));
  return axpr::MakeGlobalNaiveClassOps<pir::InferSymbolicShapeContext*>(cls);
}

}  // namespace ap::paddle
