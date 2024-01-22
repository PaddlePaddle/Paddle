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

#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include <glog/logging.h>
#include <algorithm>
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

using ShapeOrDataDimExprs4ValueT =
    std::function<symbol::ShapeOrDataDimExprs(pir::Value)>;

std::vector<pir::Value> FindSourceDenseTensorOfDimTensor(
    pir::Value shape,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> ret{};
  const auto& Emplace = [&](pir::Value value) {
    if (std::find(ret.begin(), ret.end(), value) != ret.end()) return;
    ret.emplace_back(value);
  };
  const auto& ForEachInputValue =
      [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
        // find input dimension tensor;
        pir::Operation* owner = value.defining_op();
        if (owner == nullptr) return;
        for (int i = 0; i < owner->num_operands(); ++i) {
          Visit(owner->operand_source(i));
        }
      };
  const auto& IsDimTensor = [&](pir::Value value) -> bool {
    return ShapeOrDataDimExprs4Value(value).data().has_value();
  };
  const auto& ForEachInputDimTensor =
      [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
        // find input dimension tensor;
        ForEachInputValue(value, [&](pir::Value input) {
          if (IsDimTensor(input)) {
            Visit(input);
          }
        });
      };
  common::BfsWalker<pir::Value> walker(ForEachInputDimTensor);
  walker(shape, [&](pir::Value value) {
    size_t input_cnt = 0;
    ForEachInputValue(value, [&](pir::Value input) {
      ++input_cnt;
      if (IsDimTensor(input)) return;
      Emplace(input);
    });
    if (input_cnt == 0) {
      // `value` is a result of a source op.
      Emplace(value);
    }
  });
  return ret;
}

bool MakeGenerateShapeOpAttribute(
    pir::IrContext* ir_context,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    pir::Value output_shape,
    const std::vector<pir::Value>& origin_inputs,
    std::vector<pir::Value>* minimal_inputs,
    std::vector<pir::Attribute>* output_dim_expr_attrs,
    GenerateShapeOp::SymbolBindings* symbol_bindings) {
  const auto& shape_or_data_dim_exprs = ShapeOrDataDimExprs4Value(output_shape);
  CHECK(shape_or_data_dim_exprs.data().has_value());
  const auto& out_dim_exprs = shape_or_data_dim_exprs.data().value();
  return MakeGenerateShapeOpAttribute(ir_context,
                                      ShapeOrDataDimExprs4Value,
                                      out_dim_exprs,
                                      origin_inputs,
                                      minimal_inputs,
                                      output_dim_expr_attrs,
                                      symbol_bindings);
}

std::optional<pir::Value> GetOutOfRewritedGenerateShapeOp(
    pir::Value shape,
    pir::PatternRewriter* rewriter,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> input_tensors =
      FindSourceDenseTensorOfDimTensor(shape, ShapeOrDataDimExprs4Value);
  if (input_tensors.empty()) return std::nullopt;
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  GenerateShapeOp::SymbolBindings symbol_bindings{};
  bool success = MakeGenerateShapeOpAttribute(rewriter->ir_context(),
                                              ShapeOrDataDimExprs4Value,
                                              shape,
                                              /*origin inputs*/ input_tensors,
                                              /*minimal inputs*/ &input_tensors,
                                              &output_dim_expr_attrs,
                                              &symbol_bindings);
  if (!success) return std::nullopt;
  return rewriter
      ->Build<cinn::dialect::GenerateShapeOp>(
          input_tensors, output_dim_expr_attrs, symbol_bindings)
      .out();
}

bool ReplaceShapeOpsToGenerateShape(
    pir::Value shape_operand,
    pir::PatternRewriter* rewriter,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  if (shape_operand.defining_op()->isa<cinn::dialect::GenerateShapeOp>()) {
    return false;
  }
  auto ShapeOrDataDimExprs4Value =
      [&shape_analysis](
          pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    CHECK(shape_analysis->HasShapeOrDataForValue(value));
    return shape_analysis->GetShapeOrDataForValue(value);
  };
  std::optional<pir::Value> opt_generated_shape =
      GetOutOfRewritedGenerateShapeOp(
          shape_operand, rewriter, ShapeOrDataDimExprs4Value);
  if (!opt_generated_shape.has_value()) return false;
  shape_analysis->SetShapeOrDataForValue(
      opt_generated_shape.value(), ShapeOrDataDimExprs4Value(shape_operand));
  rewriter->ReplaceAllUsesWith(shape_operand, opt_generated_shape.value());
  return true;
}

template <typename OP_TYPE>
bool ProcessOp(OP_TYPE op,
               pir::PatternRewriter* rewriter,
               pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return ReplaceShapeOpsToGenerateShape(
      op->operand_source(1), rewriter, shape_analysis);
}

}  // namespace

template <typename OPTYPE>
class FuseShapeOpsIntoGenerateShapeOpPattern
    : public pir::OpRewritePattern<OPTYPE> {
 public:
  explicit FuseShapeOpsIntoGenerateShapeOpPattern(pir::IrContext* context)
      : pir::OpRewritePattern<OPTYPE>(context) {}

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
    return ProcessOp(op, &rewriter, &shape_analysis);
  }
};

FuseShapeOpsIntoGenerateShapeOpPass::FuseShapeOpsIntoGenerateShapeOpPass()
    : pir::PatternRewritePass("fuse_shape_ops_into_generate_shape_op_pass", 1) {
}

pir::RewritePatternSet FuseShapeOpsIntoGenerateShapeOpPass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::ExpandOp>>(
      context);
  ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::ReshapeOp>>(
      context);

  return ps;
}

bool FuseShapeOpsIntoGenerateShapeOpPass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
