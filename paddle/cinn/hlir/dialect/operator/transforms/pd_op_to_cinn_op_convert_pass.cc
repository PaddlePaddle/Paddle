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

#include "paddle/cinn/hlir/dialect/operator/transforms/pd_op_to_cinn_op_convert_pass.h"

#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class PDSum2CINNReduceSumPattern
    : public pir::drr::DrrPatternBase<PDSum2CINNReduceSumPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_int_array = pat.Op("pd_op.full_int_array",
                                        {{"value", pat.Attr("axis_info")},
                                         {"dtype", pat.Attr("dtype_2")},
                                         {"place", pat.Attr("place_2")}});

    const auto &sum = pat.Op(
        "pd_op.sum",
        {{"dtype", pat.Attr("dtype")}, {"keepdim", pat.Attr("keep_dim")}});
    pat.Tensor("ret") = sum(pat.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &cinn_reduce_sum = res.Op(
        "cinn_op.reduce_sum",
        {{"axis", pat.Attr("axis_info")}, {"keep_dim", pat.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_sum(res.Tensor("arg0"));
  }
};

PdOpToCinnOpPass::PdOpToCinnOpPass() : pir::Pass("PdOpToCinnOpPass", 1) {}

bool PdOpToCinnOpPass::Initialize(pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add(PDSum2CINNReduceSumPattern().Build(context));

  patterns_ = ::pir::FrozenRewritePatternSet(std::move(ps));
  return true;
}

void PdOpToCinnOpPass::Run(pir::Operation *op) {
  pir::GreedyRewriteConfig cfg;
  cfg.use_top_down_traversal = true;
  cfg.max_iterations = 10;
  pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
}

bool PdOpToCinnOpPass::CanApplyOn(pir::Operation *op) const {
  return op->name() == "builtin.module" && op->num_regions() > 0;
}

void PdOp2CinnOpConverter(::pir::Program *program) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<PdOpToCinnOpPass>());

  pm.Run(program);
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
