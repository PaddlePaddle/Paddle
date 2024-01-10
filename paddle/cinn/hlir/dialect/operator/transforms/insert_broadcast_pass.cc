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

#include "paddle/cinn/hlir/dialect/operator/transforms/insert_broadcast_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

pir::Value GetOutputDimTensor(pir::PatternRewriter* rewriter,
                              pir::Value x,
                              pir::Value y) {
  pir::Value x_shape = rewriter->Build<paddle::dialect::ShapeOp>(x).out();
  pir::Value y_shape = rewriter->Build<paddle::dialect::ShapeOp>(y).out();
  return rewriter->Build<paddle::dialect::ShapeBroadcastOp>(x_shape, y_shape)
      .out();
}

bool ProcessOp(pir::Operation* op, pir::PatternRewriter* rewriter) {
  pir::Value x = op->operand_source(0);
  pir::Value y = op->operand_source(1);
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
  const auto& x_shape = shape_analysis.GetShapeOrDataForValue(&x);
  const auto& y_shape = shape_analysis.GetShapeOrDataForValue(&y);
  if (x_shape.shape() == y_shape.shape() && x_shape.data() == y_shape.data()) {
    return false;
  }

  pir::Value output_dim_tensor = GetOutputDimTensor(rewriter, x, y);
  {
    pir::Value broadcasted_x =
        rewriter->Build<paddle::dialect::ExpandOp>(x, output_dim_tensor).out();
    op->operand(0).set_source(broadcasted_x);
  }
  {
    pir::Value broadcasted_y =
        rewriter->Build<paddle::dialect::ExpandOp>(y, output_dim_tensor).out();
    op->operand(1).set_source(broadcasted_y);
  }
  return true;
}

template <typename OPTYPE>
class InsertBroadcastPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

class InsertBroadcastPass : public pir::PatternRewritePass {
 public:
  InsertBroadcastPass() : pir::PatternRewritePass("insert_broadcast_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    // elementwise ops
    ps.Add<InsertBroadcastPattern<paddle::dialect::AddOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::SubtractOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::MultiplyOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::DivideOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::ElementwisePowOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::RemainderOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::FloorDivideOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::MaximumOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::MinimumOp>>(context);

    // compare ops
    ps.Add<InsertBroadcastPattern<paddle::dialect::LessThanOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::LessEqualOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::EqualOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::NotEqualOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::GreaterThanOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::GreaterEqualOp>>(context);

    // bitwise ops
    ps.Add<InsertBroadcastPattern<paddle::dialect::BitwiseOrOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::BitwiseXorOp>>(context);
    ps.Add<InsertBroadcastPattern<paddle::dialect::BitwiseNotOp>>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateInsertBroadcastPass() {
  return std::make_unique<InsertBroadcastPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
