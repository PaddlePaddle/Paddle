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

#include "paddle/cinn/hlir/dialect/operator/transforms/fully_insert_broadcast_pass.h"

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

namespace {

pir::Value GetOutputDimTensor(pir::PatternRewriter* rewriter,
                              pir::Value x,
                              pir::Value y) {
  pir::Value x_shape = rewriter->Build<paddle::dialect::ShapeOp>(x).out();
  pir::Value y_shape = rewriter->Build<paddle::dialect::ShapeOp>(y).out();
  return rewriter->Build<paddle::dialect::ShapeBroadcastOp>(x_shape, y_shape)
      .out();
}

bool ProcessOp(pir::Operation* op, pir::PatternRewriter* rewriter) {
  if (op->operand_source(0).defining_op()->isa<paddle::dialect::ExpandOp>() &&
      op->operand_source(1).defining_op()->isa<paddle::dialect::ExpandOp>()) {
    return false;
  }
  pir::Value x = op->operand_source(0);
  pir::Value y = op->operand_source(1);
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

}  // namespace

template <typename OPTYPE>
class FullyInsertBroadcastPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

FullyInsertBroadcastPass::FullyInsertBroadcastPass()
    : pir::PatternRewritePass("fully_insert_broadcast_pass", 1) {}

pir::RewritePatternSet FullyInsertBroadcastPass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  // elementwise ops
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::AddOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::SubtractOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::MultiplyOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::DivideOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::ElementwisePowOp>>(
      context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::RemainderOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::FloorDivideOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::MaximumOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::MinimumOp>>(context);

  // compare ops
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::LessThanOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::LessEqualOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::EqualOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::NotEqualOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::GreaterThanOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::GreaterEqualOp>>(context);

  // bitwise ops
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::BitwiseOrOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::BitwiseXorOp>>(context);
  ps.Add<FullyInsertBroadcastPattern<paddle::dialect::BitwiseNotOp>>(context);

  return ps;
}

bool FullyInsertBroadcastPass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
