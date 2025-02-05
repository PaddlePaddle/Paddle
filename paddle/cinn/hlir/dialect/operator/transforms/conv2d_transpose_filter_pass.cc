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

#include "paddle/cinn/hlir/dialect/operator/transforms/conv2d_transpose_filter_pass.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

COMMON_DECLARE_bool(manually_trans_conv_filter);

namespace cinn::dialect::ir {

class Conv2dTransposeFilter
    : public pir::OpRewritePattern<paddle::dialect::Conv2dOp> {
  using pir::OpRewritePattern<paddle::dialect::Conv2dOp>::OpRewritePattern;

  bool Match(paddle::dialect::Conv2dOp op) const override {
    const std::string& data_format =
        op->attribute("data_format").dyn_cast<pir::StrAttribute>().AsString();
    bool already_transposed =
        op.filter().defining_op()->isa<paddle::dialect::TransposeOp>();
    return FLAGS_manually_trans_conv_filter && data_format == "NHWC" &&
           !already_transposed;
  }

  void Rewrite(paddle::dialect::Conv2dOp op,
               pir::PatternRewriter& rewriter) const override {
    // NCHW -> NHWC
    auto transpose_op = rewriter.Build<paddle::dialect::TransposeOp>(
        op.filter(), std::vector<int>{0, 2, 3, 1});
    auto new_conv_op = rewriter.Build<paddle::dialect::Conv2dOp>(
        op.input(), transpose_op.result(0), op->attributes());
    rewriter.ReplaceAllUsesWith(op.out(), new_conv_op.out());
    rewriter.EraseOp(op);
  }
};

class Conv2dTransposeFilterPass : public pir::PatternRewritePass {
 public:
  Conv2dTransposeFilterPass()
      : pir::PatternRewritePass("conv2d_transpose_filter", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<Conv2dTransposeFilter>(context);
    return ps;
  }
};

std::unique_ptr<pir::Pass> CreateConv2dTransposeFilterPass() {
  return std::make_unique<Conv2dTransposeFilterPass>();
}

}  // namespace cinn::dialect::ir
