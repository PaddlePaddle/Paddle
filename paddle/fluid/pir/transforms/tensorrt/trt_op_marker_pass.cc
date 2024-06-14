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

#include "paddle/fluid/pir/transforms/tensorrt/trt_op_marker_pass.h"

#include <memory>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

inline auto kCanRunTrtAttr = paddle::dialect::kCanRunTrtAttr;

class MatmulOpPattern
    : public pir::OpRewritePattern<paddle::dialect::MatmulOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::MatmulOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
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

class TrtOpMarkerPass : public pir::PatternRewritePass {
 public:
  TrtOpMarkerPass() : pir::PatternRewritePass("trt_op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(std::make_unique<MatmulOpPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTrtOpMarkerPass() {
  return std::make_unique<TrtOpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(trt_op_marker_pass, TrtOpMarkerPass);
