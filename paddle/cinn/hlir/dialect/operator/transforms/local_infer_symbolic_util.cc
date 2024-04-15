// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <optional>
#include <functional>
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/local_infer_symbolic_util.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {
  
void InitLocalShapeAnalysis(
    const pir::Operation& op,
    pir::ShapeConstraintIRAnalysis* shape_analysis,
    const OptDimExprs4ValueT& GraphDimExprs4Value) {
  auto VisitEachInputAndDimExprs = [&](const auto& Visit) {
    for (int i = 0; i < op.num_operands(); ++i) {
      pir::Value input = op.operand_source(i);
      std::optional<const symbol::ShapeOrDataDimExprs*> value_dim_exprs =
          GraphDimExprs4Value(input, op.GetParent());
      PADDLE_ENFORCE(value_dim_exprs.has_value());
      Visit(input, *value_dim_exprs.value());
    }
  };
  auto NewSymbolReplacedDimExprs = [&](const auto& dim_exprs) {
    auto NewSymbolReplaced = [shape_analysis](const auto& dim_expr) {
      if (dim_expr.template isa<int64_t>()) return dim_expr;
      return symbol::DimExpr(shape_analysis->GetNextSymName());
    };
    std::vector<symbol::DimExpr> ret;
    ret.reserve(dim_exprs.size());
    for (const auto& dim_expr : dim_exprs) {
      ret.push_back(NewSymbolReplaced(dim_expr));
    }
    return ret;
  };
  auto NewSymbolReplacedTensor =
    [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
      auto shape = NewSymbolReplacedDimExprs(tensor_shape_or_data.shape());
      const auto& data = tensor_shape_or_data.data();
      if (!data.has_value()) {
        return symbol::ShapeOrDataDimExprs(
            symbol::TensorShapeOrDataDimExprs(shape));
      } else {
        auto replaecd_data = NewSymbolReplacedDimExprs(data.value());
        return symbol::ShapeOrDataDimExprs(
            symbol::TensorShapeOrDataDimExprs(shape, replaecd_data));
      }
    };
  auto NewSymbolReplacedTensorList =
    [&](const symbol::TensorListShapeOrDataDimExprs& shape_or_data_list) {
      symbol::TensorListShapeOrDataDimExprs ret;
      ret.reserve(shape_or_data_list.size());
      for (auto& shape_or_data : shape_or_data_list) {
        const auto& replaced_shape_or_data =
            NewSymbolReplacedTensor(shape_or_data);
        using ImplT = symbol::TensorShapeOrDataDimExprs;
        PADDLE_ENFORCE(replaced_shape_or_data.isa<ImplT>());
        ret.push_back(replaced_shape_or_data.dyn_cast<ImplT>());
      }
      return symbol::ShapeOrDataDimExprs(ret);
    };
  auto GetNewSymbolReplaced = [&](const auto& value_dim_exprs) {
    auto patterns = symbol::Overloaded{
        NewSymbolReplacedTensor,
        NewSymbolReplacedTensorList
    };
    return std::visit(patterns, value_dim_exprs.variant());
  };
  VisitEachInputAndDimExprs([&](auto value, const auto& value_dim_exprs){
    const auto& new_symbol_replaced = GetNewSymbolReplaced(value_dim_exprs);
    shape_analysis->SetShapeOrDataForValue(value, new_symbol_replaced);
  });
}

}

std::shared_ptr<pir::ShapeConstraintIRAnalysis> MakeOpShapeAnalysis(
    const pir::Operation* op, const OptDimExprs4ValueT& GraphDimExprs4Value) {
  auto shape_analysis = std::make_shared<pir::ShapeConstraintIRAnalysis>();
  InitLocalShapeAnalysis(*op, shape_analysis.get(), GraphDimExprs4Value);

  pir::Operation* mut_op = const_cast<pir::Operation*>(op);
  auto interface =
      mut_op->dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
  if (!interface) {
    PADDLE_THROW(phi::errors::Unimplemented(
        op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  } else {
    bool infer_result = interface.InferSymbolicShape(shape_analysis.get());
    PADDLE_ENFORCE(
        infer_result,
        "InferSymbolicShape for %s failed.",
        op->name());
  }
  return shape_analysis;
}

OptDimExprs4ValueT MakeOpDimExprs4Value(
    const pir::Operation* op, const OptDimExprs4ValueT& GraphDimExprs4Value) {
  auto shape_analysis = MakeOpShapeAnalysis(op, GraphDimExprs4Value);
  const auto* my_block = op.GetParent();
  return [shape_analysis, my_block](pir::Value value, const pir::Block* block)
      -> std::optional<const symbol::ShapeOrDataDimExprs*> {
    if (block != my_block) return std::nullopt;
    if (!shape_analysis->HasShapeOrDataForValue(value)) return std::nullopt;
    return &shape_analysis->GetShapeOrDataForValue(value);
  };
}

}


}  // namespace ir
}  // namespace dialect
}  // namespace cinn
