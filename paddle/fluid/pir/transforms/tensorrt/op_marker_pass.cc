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

#include "paddle/fluid/pir/transforms/tensorrt/op_marker_pass.h"
#include <memory>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

inline const char kCanRunTrtAttr[] = "__can_run_tensorrt__";

class MatmulOpPattern_0 : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulOpPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    const auto &src_matmul = src.Op(paddle::dialect::MatmulOp::name(),
                                    {{"transpose_x", src.Attr("transpose_x")},
                                     {"transpose_y", src.Attr("transpose_y")}});
    src.Tensor("matmul_out") = src_matmul(src.Tensor("x"), src.Tensor("y"));

    src.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto matmul_op = match_ctx.Tensor("matmul_out").defining_op();
      if (matmul_op->HasAttribute(kCanRunTrtAttr) &&
          matmul_op->attribute<bool>(kCanRunTrtAttr)) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &res_matmul = res.Op(paddle::dialect::MatmulOp::name(),
                                    {{"transpose_x", src.Attr("transpose_x")},
                                     {"transpose_y", src.Attr("transpose_y")}},
                                    {{kCanRunTrtAttr, res.BoolAttr(true)}});
    res.Tensor("matmul_out") = res_matmul(res.Tensor("x"), res.Tensor("y"));
  }
};

class MatmulOpPattern_1
    : public pir::OpRewritePattern<paddle::dialect::MatmulOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::MatmulOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<bool>(kCanRunTrtAttr)) {
      return false;
    }
    auto matmul_op = rewriter.Build<paddle::dialect::MatmulOp>(
        op.x(), op.y(), op->attributes());
    matmul_op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    rewriter.ReplaceAllUsesWith(op.out(), matmul_op.out());
    rewriter.EraseOp(op);
    return true;
  }
};

class OpMarkerPass : public pir::PatternRewritePass {
 public:
  OpMarkerPass() : pir::PatternRewritePass("op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<MatmulOpPattern_0>(context));
    ps.Add(std::make_unique<MatmulOpPattern_1>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateOpMarkerPass() {
  return std::make_unique<OpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(op_marker_pass, OpMarkerPass);
