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

#include "paddle/cinn/hlir/dialect/operator/transforms/fusion_fallback_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/dialect.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class FusionOpsPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpsPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       ::pir::PatternRewriter& rewriter) const override {
    details::FallbackFusionOpToPdOps(fusion_op, rewriter);
    return true;
  }
};

class FusionFallbackPass : public pir::PatternRewritePass {
 public:
  FusionFallbackPass() : pir::PatternRewritePass("fusion_fallback_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpsPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateFusionFallbackPass() {
  return std::make_unique<FusionFallbackPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
