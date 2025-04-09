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
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::unordered_set<std::string> SKIP_CHECK_OPS = {
    "pd_op.add",           "pd_op.subtract",     "pd_op.multiply",
    "pd_op.divide",        "pd_op.floor_divide", "pd_op.minimum",
    "pd_op.maximum",       "pd_op.remainder",    "pd_op.elementwise_pow",
    "pd_op.bitwise_and",   "pd_op.bitwise_or",   "pd_op.bitwise_xor",
    "pd_op.fmax",          "pd_op.fmin",         "pd_op.heaviside",
    "pd_op.less_than",     "pd_op.less_equal",   "pd_op.greater_than",
    "pd_op.greater_equal", "pd_op.equal",        "pd_op.not_equal",
    "pd_op.logical_and",   "pd_op.logical_or",   "pd_op.logical_xor",
    "pd_op.shape",         "pd_op.shape64"};

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
    // Inserting too many check op in while block will cause model too slow, so
    // skip the while block checking.
    if (op->isa<paddle::dialect::WhileOp>()) return;
    for (std::size_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        BlockDimExprsAsserter asserter(GraphDimExprs4Value, ir_ctx_, &block);
        asserter.AssertDimExprs();
      }
    }
  }

  DimExprs4ValueT GetOpDimExprs4Value(const pir::Operation* op) {
    return MakeOpDimExprs4Value(op, GraphDimExprs4Value);
  }

  void AssertDimExprForOutput(pir::Operation* op) {  // NOLINT
    VLOG(5) << "Add assert for result of [ " << op->name() << " ]";
    if (op->num_results() == 0) return;
    if (!op->HasInterface<paddle::dialect::InferSymbolicShapeInterface>() ||
        op->HasTrait<pir::SideEffectTrait>()) {
      LOG(INFO) << "skip the checking for [ " << op->name() << " ]";
      return;
    }
    const bool is_same_operand_result_op = [&] {
      if (op->num_operands() != 1 || op->num_results() != 1) return false;
      if (!op->operand_source(0).type().isa<paddle::dialect::DenseTensorType>())
        return false;
      if (op->result(0).type().isa<paddle::dialect::DenseTensorType>())
        return false;
      return GraphDimExprs4Value(op->operand_source(0)) ==
             GraphDimExprs4Value(op->result(0));
    }();
    // skip the ops which operand and result have same shape
    if (is_same_operand_result_op) return;
    if (SKIP_CHECK_OPS.count(op->name())) return;

    auto OpDimExprs4Value = GetOpDimExprs4Value(op);
    const auto& inputs = [&] {
      std::vector<pir::Value> inputs;
      inputs.reserve(op->num_operands());
      for (int i = 0; i < op->num_operands(); ++i) {
        const auto& input = op->operand_source(i);
        if (!input || !input.type()) continue;
        if (input.type().isa<pir::VectorType>() ||
            input.type().isa<paddle::dialect::DenseTensorArrayType>()) {
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
      if (!output || !output.type() || output.use_empty()) continue;
      const auto& shape_or_data_dim_expr = GraphDimExprs4Value(output);
      if (!shape_or_data_dim_expr.isa<symbol::TensorShapeOrDataDimExprs>())
        continue;
      if (shape_or_data_dim_expr.data().has_value()) {
        TryAssertDimExprsForOutputData(op, inputs, output, OpDimExprs4Value);
      } else {
        TryAssertDimExprsForOutputShape(op, inputs, output, OpDimExprs4Value);
      }
    }
  }

  void TryAssertDimExprsForOutputShape(
      const pir::Operation* op,
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
    AddAssertEqual(
        op, shape_tensor_from_dim_exprs, shape_tensor_from_infer_meta);
  }

  std::optional<pir::Value> BuildShapeTensorFromShapeDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const DimExprs4ValueT& OpDimExprs4Value) {
    const auto& shape_or_data = OpDimExprs4Value(output);
    const auto& dim_exprs = shape_or_data.shape();
    return BuildShapeTensorFromDimExprs(inputs, dim_exprs, OpDimExprs4Value);
  }

  std::optional<pir::Value> BuildShapeTensorFromDataDimExprs(
      const std::vector<pir::Value>& inputs,
      pir::Value output,
      const DimExprs4ValueT& OpDimExprs4Value) {
    const auto& shape_or_data = OpDimExprs4Value(output);
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
    SymbolBindings symbol_bindings{};
    bool success =
        MakeGenerateShapeOpAttribute(ir_ctx_,
                                     LocalDimExprs4Value,
                                     dim_exprs,
                                     /*origin inputs*/ inputs,
                                     /*minimal inputs*/ &input_tensors,
                                     &output_dim_expr_attrs,
                                     &symbol_bindings);
    if (!success) return std::nullopt;
    auto out_type = paddle::dialect::DenseTensorType::get(
        builder_.ir_context(),
        pir::Int64Type::get(builder_.ir_context()),
        ::common::make_ddim({dim_exprs.size()}));
    return builder_
        .Build<cinn::dialect::GenerateShapeOp>(
            input_tensors, output_dim_expr_attrs, symbol_bindings, out_type)
        .out();
  }

  pir::Value BuildShapeTensorFromInferMeta(pir::Value output) {
    return builder_.Build<paddle::dialect::Shape64Op>(output).out();
  }

  void TryAssertDimExprsForOutputData(const pir::Operation* op,
                                      const std::vector<pir::Value>& inputs,
                                      pir::Value output,
                                      const DimExprs4ValueT& OpDimExprs4Value) {
    auto opt_shape_tensor_from_dim_exprs =
        BuildShapeTensorFromDataDimExprs(inputs, output, OpDimExprs4Value);
    if (!opt_shape_tensor_from_dim_exprs.has_value()) return;
    const auto& output_dims =
        output.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    if (::common::contain_unknown_dim(output_dims)) return;
    pir::Value flatten_output = [&] {
      if (output_dims.size() > 1) {
        return builder_
            .Build<paddle::dialect::FlattenOp>(
                output, 0, output_dims.size() - 1)
            .out();
      }
      return output;
    }();
    AddAssertEqual(op, opt_shape_tensor_from_dim_exprs.value(), flatten_output);
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

  void AddAssertEqual(const pir::Operation* op,
                      pir::Value lhs,
                      pir::Value rhs) {
    size_t lhs_numel = GetNumel(lhs);
    size_t rhs_numel = GetNumel(rhs);
    PADDLE_ENFORCE_EQ(lhs_numel,
                      rhs_numel,
                      ::common::errors::InvalidArgument(
                          "Check [%s id:%d] infer symbolic shape failed."
                          "The numel of lhs and rhs must be equal, but "
                          "received lhs's numel is [%d], rhs's numel is [%d]",
                          op->name(),
                          op->id(),
                          lhs_numel,
                          rhs_numel));

    pir::Value rhs_value = [&] {
      const auto& lhs_dtype =
          lhs.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
      const auto& rhs_dtype =
          rhs.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
      if (lhs_dtype != rhs_dtype) {
        return builder_
            .Build<paddle::dialect::CastOp>(
                rhs, paddle::dialect::TransToPhiDataType(lhs_dtype))
            .out();
      }
      return rhs;
    }();

    pir::Value lhs_eq_rhs =
        builder_.Build<paddle::dialect::EqualOp>(lhs, rhs_value).out();
    pir::Value all_eq =
        builder_.Build<paddle::dialect::AllOp>(lhs_eq_rhs).out();
    pir::Value assert_data =
        builder_.Build<pir::CombineOp>(std::vector<pir::Value>{lhs, rhs_value})
            .out();
    auto assert_op = builder_.Build<paddle::dialect::AssertOp>(
        all_eq, assert_data, lhs_numel);
    const std::string error_msg = "Check [" + op->name() +
                                  " id:" + std::to_string(op->id()) +
                                  "] infer symbolic shape failed.";
    assert_op->set_attribute(
        paddle::dialect::AssertOp::ERROR_INFO_ATTR_NAME,
        pir::StrAttribute::get(pir::IrContext::Instance(), error_msg));
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
