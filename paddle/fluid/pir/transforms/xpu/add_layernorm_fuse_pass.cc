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

#include "paddle/fluid/pir/transforms/xpu/add_layernorm_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class AddLayernormPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "AddLayernormPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    add({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("add_out")});
    layernorm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::vector<int64_t> x_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("x"));
      std::vector<int64_t> y_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (x_shape.size() == y_shape.size()) {
        return true;
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &add_layernorm_xpu =
        res.Op(paddle::dialect::AddLayernormXpuOp::name(),
               {{{"epsilon", pat.Attr("epsilon")},
                 {"begin_norm_axis", pat.Attr("begin_norm_axis")}}});
    add_layernorm_xpu({&res.Tensor("x"),
                       &res.Tensor("y"),
                       &res.Tensor("scale"),
                       &res.Tensor("bias")},
                      {&res.Tensor("layernorm_out")});
  }
};

class AddLayernormXpuFusePass : public pir::PatternRewritePass {
 public:
  AddLayernormXpuFusePass()
      : pir::PatternRewritePass("add_layernorm_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AddLayernormPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAddLayernormXpuFusePass() {
  return std::make_unique<AddLayernormXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(add_layernorm_xpu_fuse_pass, AddLayernormXpuFusePass);
