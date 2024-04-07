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

#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/common/ddim.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class BlockDimExprsAsserter {
 public:
  BlockDimExprsAsserter(
      const OptDimExprs4ValueT& func,
      pir::IrContext* ir_ctx,
      pir::Block* block)
    : GraphDimExprs4Value(func),
      block_(block),
      ir_ctx_(ir_ctx),
      builder_(ir_ctx, block) {}

  void AssertDimExprs() {
    auto ops = block_->GetOperators();
    for (const auto* op : ops) {
      if (op->num_regions() == 0) {
        AssertDimExprForOutput(*op);
      } else {
        AssertOpRegions(op);
      }
    }
  }


 private:
  void AssertOpRegions(const pir::Operation* op) {
    for (std::size_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        BlockDimExprsAsserter asserter(GraphDimExprs4Value, ir_ctx_, &block);
        asserter.AssertDimExprs();
      }
    }
  }

  void InitLocalShapeAnalysis(
      const pir::Operation& op,
      pir::ShapeConstraintIRAnalysis* shape_analysis) {
    auto VisitEachInputAndDimExprs = [&](const auto& Visit) {
      for (int i = 0; i < op.num_operands(); ++i) {
        pir::Value input = op.operand_source(i);
        std::optional<const symbol::ShapeOrDataDimExprs*> value_dim_exprs =
            GraphDimExprs4Value(value);
        PADDLE_ENFORCE(value_dim_exprs.has_value());
        Visit(input, *value_dim_exprs.value());
      }
    };
    auto NewSymbolReplacedDimExprs = [&](const auto& dim_exprs) {
      auto NewSymbolReplaced = [](const auto& dim_expr) {
        if (dim_expr.isa<int64_t>()) return dim_expr;
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
          auto data = NewSymbolReplacedDimExprs(data.value());
          return symbol::ShapeOrDataDimExprs(
              symbol::TensorShapeOrDataDimExprs(shape, data));
        }
      };
    auto NewSymbolReplacedTensorList =
      [&](const NewSymbolReplacedTensorList& shape_or_data_list) {
        symbol::TensorListShapeOrDataDimExprs ret;
        ret.reserve(shape_or_data_list.size());
        for (auto& shape_or_data : shape_or_data_list) {
          ret.push_back(NewSymbolReplacedTensor(shape_or_data));
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

  OptDimExprs4ValueT MakeOpDimExprs4Value(const pir::Operation& op) {
    auto shape_analysis = std::make_shared<pir::ShapeConstraintIRAnalysis>();
    InitLocalShapeAnalysis(op, shape_analysis.get());
    auto infer_symbolic_shape_interface =
        op.dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
    if (!infer_symbolic_shape_interface) {
      PADDLE_THROW(phi::errors::Unimplemented(
          op.name() + " DOES NOT have InferSymbolicShapeInterface!"));
    } else {
      bool infer_result =
          infer_symbolic_shape_interface.InferSymbolicShape(shape_analysis);
      PADDLE_ENFORCE(
          infer_result,
          "InferSymbolicShape for %s failed.",
          op.name());
    }
    return [shape_analysis](pir::Value value)
        -> std::optional<const symbol::ShapeOrDataDimExprs*> {
      if (!shape_analysis->HasShapeOrDataForValue(value)) return std::nullopt;
      return &shape_analysis->GetShapeOrDataForValue(value);
    };
  }

  void AssertDimExprForOutput(
      const pir::Operation& op) {  // NOLINT
    auto OpDimExprs4Value = MakeOpDimExprs4Value(op);
    const auto& inputs = [&]{
      std::vector<pir::Value> inputs;
      inputs.reserve(op.num_operands())；
      for (int i = 0; i < op.num_operands(); ++i) {
        inputs.push_back(op.operand_source(i));
      }
      return inputs;
    }();
    for (std::size_t i = 0; i < op.num_results(); ++i) {
      pir::Value output = op.result(i);
      const auto& shape_or_data_dim_expr = GraphDimExprs4Value(output, block_);
      if (!shape_or_data_dim_expr.has_value()) continue;
      const auto* dim_exprs = shape_or_data_dim_expr.value();
      if (!dim_exprs->isa<symbol::TensorShapeOrDataDimExprs>()) continue;
      TryAssertDimExprsForOutputShape(inputs, output, OpDimExprs4Value);
      TryAssertDimExprsForOutputData(inputs, output, OpDimExprs4Value);
    }
  }

  void TryAssertDimExprsForOutputShape(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const OptDimExprs4ValueT& OpDimExprs4Value) {
    auto opt_shape_tensor_from_dim_exprs =
      BuildShapeTensorFromShapeDimExprs(inputs, output, OpDimExprs4Value);
    if (!opt_shape_tensor_from_dim_exprs.has_value()) return;
    shape_tensor_from_dim_exprs = opt_shape_tensor_from_dim_exprs.value();
    auto shape_tensor_from_infer_meta =
      BuildShapeTensorFromInferMeta(output);
    AddAssertEqual(shape_tensor_from_dim_exprs, shape_tensor_from_infer_meta);
  }

  std::optional<pir::Value> BuildShapeTensorFromShapeDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const OptDimExprs4ValueT& OpDimExprs4Value) {
    const auto& opt_shape_or_data = GraphDimExprs4Value(output, block_);
    PADDLE_ENFORCE(opt_shape_or_data.has_value());
    const auto& dim_exprs = opt_shape_or_data.value()->shape();
    return BuildShapeTensorFromDimExprs(inputs, dim_exprs, OpDimExprs4Value);
  }

  std::optional<pir::Value> BuildShapeTensorFromDataDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const OptDimExprs4ValueT& OpDimExprs4Value) {
    const auto& opt_shape_or_data = GraphDimExprs4Value(output, block_);
    PADDLE_ENFORCE(opt_shape_or_data.has_value());
    const auto& dim_exprs = opt_shape_or_data.value()->data();
    if (!dim_exprs.has_value()) return std::nullopt;
    return BuildShapeTensorFromDimExprs(inputs, dim_exprs.value(), OpDimExprs4Value);
  }

  std::optional<pir::Value> BuildShapeTensorFromDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const OptDimExprs4ValueT& OpDimExprs4Value) {
  const auto& LocalDimExprs4Value =
      [&](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    const auto& opt_dim_exprs = OpDimExprs4Value(value, block_);
    PADDLE_ENFORCE(opt_dim_exprs.has_value());
    return *opt_dim_exprs.value();
  };
  std::vector<pir::Value> input_tensors{};
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  GenerateShapeOp::SymbolBindings symbol_bindings{};
  bool success = MakeGenerateShapeOpAttribute(ir_ctx_,
                                              LocalDimExprs4Value,
                                              dim_exprs,
                                              /*origin inputs*/ inputs,
                                              /*minimal inputs*/ &input_tensors,
                                              &output_dim_expr_attrs,
                                              &symbol_bindings);
  if (!success) return std::nullopt;
  return builder_
      .Build<cinn::dialect::GenerateShapeOp>(
          input_tensors, output_dim_expr_attrs, symbol_bindings)
      .out();
  }

  pir::Value BuildShapeTensorFromInferMeta(pir::Value output) {
    return builder_.Build<paddle::dialect::ShapeOp>(output).out();
  }

  void TryAssertDimExprsForOutputData(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const OptDimExprs4ValueT& OpDimExprs4Value) {
    auto opt_shape_tensor_from_dim_exprs =
      BuildShapeTensorFromDataDimExprs(inputs, output, OpDimExprs4Value);
    if (!opt_shape_tensor_from_dim_exprs.has_value()) return;
    AddAssertEqual(opt_shape_tensor_from_dim_exprs.value(), output);
  }

  size_t GetNumel(pir::Value value) {
    const auto& dims = value.type().dyn_cast<pir::DenseTensorType>().dims();
    int64_t numel = ::common::product(dims);
    PADDLE_ENFORCE_GE(numel, 0);
    return numel;
  }

  void AddAssertEqual(pir::Value lhs, pir::Value rhs) {
    PADDLE_ENFORCE_EQ(GetNumel(lhs), GetNumel(rhs));
    pir::Value lhs_eq_rhs =
      builder_.Build<paddle::dialect::EqualOp>(lhs, rhs).out();
    pir::Value all_eq =
      builder_.Build<paddle::dialect::AllOp>(lhs_eq_rhs).out();
    builder_
      .Build<paddle::dialect::AssertOp>(all_eq, lhs_eq_rhs, GetNumel(lhs));
  }

  static std::vector<pir::Value> GetInputs(const pir::Block& block) {
    const auto& block_ops = [&]{
      std::set<const pir::Operation*> ops;
      for (const auto& op : block) {
        ops.insert(&op);
      }
      return ops;
    }();
    const auto& IsInput = [&](pir::Value input) {
      const auto* block_ops = input.defining_op();
      return ops.count(op) == 0;
    };
    std::vector<pir::Value> inputs;
    const auto& Collected = [&](pir::Value input) {
      return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
    };
    for (const auto& op : block) {
      for (int i = 0; i < op.num_operands(); ++i) {
        pir::Value input = op.operand_source(i);
        if (!IsInput(input)) continue;
        if (Collected(input)) continue;
        inputs.emplace_back(&op);
      }
    }
    return inputs;
  }

  OptDimExprs4ValueT GraphDimExprs4Value;
  pir::IrContext* ir_ctx_;
  pir::Block* block_;
  pir::Builder builder_;
};

class CheckInferSymbolicPass : public pir::Pass {
 public:
  CheckInferSymbolicPass(const OptDimExprs4ValueT& func)
    : pir::Pass("check_infer_symbolic", 1),
      GraphDimExprs4Value(func) {}


  void Run(pir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        auto* ir_ctx = IrContext::Instance();
        BlockDimExprsAsserter asserter(GraphDimExprs4Value, ir_ctx, &block);
        asserter.AssertDimExprs();
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:

  OptDimExprs4ValueT GraphDimExprs4Value;
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateCheckInferSymbolicPass(
    const OptDimExprs4ValueT& GraphDimExprs4Value) {
  return std::make_unique<CheckInferSymbolicPass>(GraphDimExprs4Value);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
