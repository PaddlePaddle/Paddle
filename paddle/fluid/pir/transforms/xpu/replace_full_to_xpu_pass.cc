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

#include "paddle/fluid/pir/transforms/xpu/replace_full_to_xpu_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
class ReplaceFullPattern : public paddle::drr::DrrPatternBase {
 public:
  ReplaceFullPattern() {}
  std::string name() const override { return "ReplaceFullPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name());
    full_op({}, {&pat.Tensor("full_out")});
    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &full_op_res = res.Op(paddle::dialect::FullOp::name(),
                                     {{
                                         {"place", res.PlaceAttr("xpu")},
                                     }});

    full_op_res({}, {&pat.Tensor("full_out")});
  }
};

class ReplaceFullPass : public pir::PatternRewritePass {
 public:
  ReplaceFullPass() : pir::PatternRewritePass("replace_full_xpu_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ReplaceFullPattern>(context));
    return ps;
  }

  pir::GreedyRewriteConfig InitializeConfig() override {
    pir::GreedyRewriteConfig config;

    config.use_top_down_traversal = false;

    config.max_iterations = 10;
    return config;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateReplaceFullPass() {
  return std::make_unique<ReplaceFullPass>();
}

}  // namespace pir

REGISTER_IR_PASS(replace_full_xpu_pass, ReplaceFullPass);
