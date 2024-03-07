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

#include "paddle/cinn/hlir/dialect/operator/transforms/merge_full_with_broadcast_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

pir::Value GetOutputDimTensor(pir::PatternRewriter* rewriter,
                              pir::Value x,
                              pir::Value y) {
  pir::Value x_shape = rewriter->Build<paddle::dialect::ShapeOp>(x).out();
  pir::Value y_shape = rewriter->Build<paddle::dialect::ShapeOp>(y).out();
  return rewriter->Build<paddle::dialect::ShapeBroadcastOp>(x_shape, y_shape)
      .out();
}

pir::Value GetFullValueTensor(const paddle::dialect::FullOp& origin_full_op,
                              pir::PatternRewriter* rewriter) {
  float value = origin_full_op->attribute<pir::FloatAttribute>("value").data();
  auto dtype =
      origin_full_op->attribute<paddle::dialect::DataTypeAttribute>("dtype")
          .data();
  pir::Value new_value = rewriter
                             ->Build<paddle::dialect::FullOp>(
                                 std::vector<int64_t>{1}, value, dtype)
                             .out();
  return new_value;
}

bool ProcessOp(pir::Operation* op, pir::PatternRewriter* rewriter) {
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
  pir::Value x = op->operand_source(0);
  pir::Value y = op->operand_source(1);

  const auto& MergeFullBroadcast = [&](pir::Value input) {
    const auto& input_shape = shape_analysis.GetShapeOrDataForValue(input);
    const auto& out_shape =
        shape_analysis.GetShapeOrDataForValue(op->result(0));
    if (input_shape == out_shape) {
      return false;
    }
    pir::Value output_dim_tensor = GetOutputDimTensor(rewriter, x, y);
    pir::Value new_value = GetFullValueTensor(
        input.defining_op()->dyn_cast<paddle::dialect::FullOp>(), rewriter);
    pir::Value new_out = rewriter
                             ->Build<paddle::dialect::FullWithTensorOp>(
                                 output_dim_tensor, new_value)
                             .out();
    op->operand(0).set_source(new_out);

    return true;
  };

  bool has_merged = false;

  if (x.defining_op()->isa<paddle::dialect::FullOp>() &&
      shape_analysis.HasShapeOrDataForValue(x)) {
    has_merged = has_merged || MergeFullBroadcast(x);
  }

  if (y.defining_op()->isa<paddle::dialect::FullOp>() &&
      shape_analysis.HasShapeOrDataForValue(y)) {
    has_merged = has_merged || MergeFullBroadcast(y);
  }

  return has_merged;
}

}  // namespace

template <typename OPTYPE>
class MergeFullBroadcastPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

class MergeFullBroadcastPass : public pir::PatternRewritePass {
 public:
  MergeFullBroadcastPass()
      : pir::PatternRewritePass("merge_full_broadcast_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    // elementwise ops
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::AddOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::SubtractOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::MultiplyOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::DivideOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::ElementwisePowOp>>(
        context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::RemainderOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::FloorDivideOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::MaximumOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::MinimumOp>>(context);

    // compare ops
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::LessThanOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::LessEqualOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::EqualOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::NotEqualOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::GreaterThanOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::GreaterEqualOp>>(context);

    // bitwise ops
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::BitwiseOrOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::BitwiseXorOp>>(context);
    ps.Add<MergeFullBroadcastPattern<paddle::dialect::BitwiseNotOp>>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateMergeFullBroadcastPass() {
  return std::make_unique<MergeFullBroadcastPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
