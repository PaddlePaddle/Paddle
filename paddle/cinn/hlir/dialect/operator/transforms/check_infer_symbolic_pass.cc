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

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class BlockDimExprsAsserter {
 public:
  BlockDimExprsAsserter(const DimExprs4ValueT& func,
                        pir::IrContext* ir_ctx,
                        pir::Block* block)
      : GraphDimExprs4Value(func),
        block_(block),
        ir_ctx_(ir_ctx),
        builder_(ir_ctx, block) {}

  void AssertDimExprs() {
    const auto ops = [&] {
      std::vector<pir::Operation*> ops;
      ops.reserve(block_->size());
      for (auto& op : *block_) {
        ops.push_back(&op);
      }
      return ops;
    }();
    for (auto* op : ops) {
      if (op->num_regions() == 0) {
        AssertDimExprForOutput(op);
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

  void InitLocalShapeAnalysis(const pir::Operation& op,
                              pir::ShapeConstraintIRAnalysis* shape_analysis) {
    auto VisitEachInputAndDimExprs = [&](const auto& Visit) {
      for (int i = 0; i < op.num_operands(); ++i) {
        pir::Value input = op.operand_source(i);
        const auto& value_dim_exprs = GraphDimExprs4Value(input);
        Visit(input, value_dim_exprs);
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
            ret.push_back(replaced_shape_or_data
                              .dyn_cast<symbol::TensorShapeOrDataDimExprs>());
          }
          return symbol::ShapeOrDataDimExprs(ret);
        };
    auto GetNewSymbolReplaced = [&](const auto& value_dim_exprs) {
      auto patterns = symbol::Overloaded{NewSymbolReplacedTensor,
                                         NewSymbolReplacedTensorList};
      return std::visit(patterns, value_dim_exprs.variant());
    };
    VisitEachInputAndDimExprs([&](auto value, const auto& value_dim_exprs) {
      const auto& new_symbol_replaced = GetNewSymbolReplaced(value_dim_exprs);
      shape_analysis->SetShapeOrDataForValue(value, new_symbol_replaced);
    });
  }

  DimExprs4ValueT MakeOpDimExprs4Value(const pir::Operation* op) {
    auto shape_analysis = std::make_shared<pir::ShapeConstraintIRAnalysis>();
    InitLocalShapeAnalysis(*op, shape_analysis.get());

    pir::Operation* mut_op = const_cast<pir::Operation*>(op);
    auto interface =
        mut_op->dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
    if (!interface) {
      PADDLE_THROW(phi::errors::Unimplemented(
          op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
    } else {
      bool infer_result = interface.InferSymbolicShape(shape_analysis.get());
      PADDLE_ENFORCE_EQ(infer_result,
                        true,
                        ::common::errors::PreconditionNotMet(
                            "InferSymbolicShape for %s failed.", op->name()));
    }
    return [shape_analysis](
               pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
      return shape_analysis->GetShapeOrDataForValue(value);
    };
  }

  void AssertDimExprForOutput(pir::Operation* op) {  // NOLINT
    VLOG(5) << "Add assert for result of [ " << op->name() << " ]";
    if (!op->HasInterface<paddle::dialect::InferSymbolicShapeInterface>()) {
      LOG(INFO) << "skip the checking for [ " << op->name() << " ]";
      return;
    }
    auto OpDimExprs4Value = MakeOpDimExprs4Value(op);
    const auto& inputs = [&] {
      std::vector<pir::Value> inputs;
      inputs.reserve(op->num_operands());
      for (int i = 0; i < op->num_operands(); ++i) {
        const auto& input = op->operand_source(i);
        if (input.type().isa<pir::VectorType>()) {
          return std::vector<pir::Value>{};
        }
        inputs.push_back(input);
      }
      return inputs;
    }();
    if (inputs.empty()) return;
    builder_.SetInsertionPointAfter(op);
    for (std::size_t i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      const auto& shape_or_data_dim_expr = GraphDimExprs4Value(output);
      if (!shape_or_data_dim_expr.isa<symbol::TensorShapeOrDataDimExprs>())
        continue;
      if (shape_or_data_dim_expr.data().has_value()) {
        TryAssertDimExprsForOutputData(inputs, output, OpDimExprs4Value);
      } else {
        TryAssertDimExprsForOutputShape(inputs, output, OpDimExprs4Value);
      }
    }
  }

  void TryAssertDimExprsForOutputShape(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const DimExprs4ValueT& OpDimExprs4Value) {
    if (!::common::contain_unknown_dim(
            output.type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dims())) {
      return;
    }
    auto opt_shape_tensor_from_dim_exprs =
        BuildShapeTensorFromShapeDimExprs(inputs, output, OpDimExprs4Value);
    if (!opt_shape_tensor_from_dim_exprs.has_value()) return;
    const auto& shape_tensor_from_dim_exprs =
        opt_shape_tensor_from_dim_exprs.value();
    auto shape_tensor_from_infer_meta = BuildShapeTensorFromInferMeta(output);
    AddAssertEqual(shape_tensor_from_dim_exprs, shape_tensor_from_infer_meta);
  }

  std::optional<pir::Value> BuildShapeTensorFromShapeDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const DimExprs4ValueT& OpDimExprs4Value) {
    const auto& shape_or_data = GraphDimExprs4Value(output);
    const auto& dim_exprs = shape_or_data.shape();
    return BuildShapeTensorFromDimExprs(inputs, dim_exprs, OpDimExprs4Value);
  }

  std::optional<pir::Value> BuildShapeTensorFromDataDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const DimExprs4ValueT& OpDimExprs4Value) {
    const auto& shape_or_data = GraphDimExprs4Value(output);
    const auto& dim_exprs = shape_or_data.data();
    if (!dim_exprs.has_value()) return std::nullopt;
    return BuildShapeTensorFromDimExprs(
        inputs, dim_exprs.value(), OpDimExprs4Value);
  }

  std::optional<pir::Value> BuildShapeTensorFromDimExprs(
      const std::vector<pir::Value>& inputs,
      const std::vector<symbol::DimExpr>& dim_exprs,
      const DimExprs4ValueT& OpDimExprs4Value) {
    const auto& LocalDimExprs4Value =
        [&](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
      return OpDimExprs4Value(value);
    };
    std::vector<pir::Value> input_tensors{};
    std::vector<pir::Attribute> output_dim_expr_attrs{};
    GenerateShapeOp::SymbolBindings symbol_bindings{};
    bool success =
        MakeGenerateShapeOpAttribute(ir_ctx_,
                                     LocalDimExprs4Value,
                                     dim_exprs,
                                     /*origin inputs*/ inputs,
                                     /*minimal inputs*/ &input_tensors,
                                     &output_dim_expr_attrs,
                                     &symbol_bindings);
    if (!success) return std::nullopt;
    auto out_shape_value =
        builder_
            .Build<cinn::dialect::GenerateShapeOp>(
                input_tensors, output_dim_expr_attrs, symbol_bindings)
            .out();
    return builder_
        .Build<paddle::dialect::CastOp>(out_shape_value, phi::DataType::INT32)
        .out();
  }

  pir::Value BuildShapeTensorFromInferMeta(pir::Value output) {
    return builder_.Build<paddle::dialect::ShapeOp>(output).out();
  }

  void TryAssertDimExprsForOutputData(const std::vector<pir::Value>& inputs,
                                      pir::Value output,
                                      const DimExprs4ValueT& OpDimExprs4Value) {
    auto opt_shape_tensor_from_dim_exprs =
        BuildShapeTensorFromDataDimExprs(inputs, output, OpDimExprs4Value);
    if (!opt_shape_tensor_from_dim_exprs.has_value()) return;
    AddAssertEqual(opt_shape_tensor_from_dim_exprs.value(), output);
  }

  size_t GetNumel(pir::Value value) {
    const auto& dims = value.type().dyn_cast<pir::DenseTensorType>().dims();
    int64_t numel = ::common::product(dims);
    PADDLE_ENFORCE_GE(
        numel,
        0,
        ::common::errors::InvalidArgument(
            "The numel of value must be >= 0, but received numel is %d.",
            numel));
    return numel;
  }

  void AddAssertEqual(pir::Value lhs, pir::Value rhs) {
    size_t lhs_numel = GetNumel(lhs);
    size_t rhs_numel = GetNumel(rhs);
    PADDLE_ENFORCE_EQ(lhs_numel,
                      rhs_numel,
                      ::common::errors::InvalidArgument(
                          "The numel of lhs and rhs must be equal, but "
                          "received lhs's numel is [%d], rhs's numel is [%d]",
                          lhs_numel,
                          rhs_numel));
    pir::Value lhs_eq_rhs =
        builder_.Build<paddle::dialect::EqualOp>(lhs, rhs).out();
    pir::Value all_eq =
        builder_.Build<paddle::dialect::AllOp>(lhs_eq_rhs).out();
    builder_.Build<paddle::dialect::AssertOp>(all_eq, lhs_eq_rhs, lhs_numel);
  }

  DimExprs4ValueT GraphDimExprs4Value;
  pir::IrContext* ir_ctx_;
  pir::Block* block_;
  pir::Builder builder_;
};

class CheckInferSymbolicPass : public pir::Pass {
 public:
  explicit CheckInferSymbolicPass(const DimExprs4ValueT& func)
      : pir::Pass("check_infer_symbolic", 1), GraphDimExprs4Value(func) {}

  void Run(pir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        auto* ir_ctx = pir::IrContext::Instance();
        BlockDimExprsAsserter asserter(GraphDimExprs4Value, ir_ctx, &block);
        asserter.AssertDimExprs();
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  DimExprs4ValueT GraphDimExprs4Value;
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateCheckInferSymbolicPass(
    const DimExprs4ValueT& GraphDimExprs4Value) {
  return std::make_unique<CheckInferSymbolicPass>(GraphDimExprs4Value);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
