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

#include "paddle/fluid/pir/transforms/xpu/group_norm_silu_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

/*
fuse gn + activation block in to group_norm_silu op
For example:
graph:
                      X
              Scale   |   Bias
                   \  |  /
                  group norm
                   /  |  \
                  /   |   \
            variance  |   mean
                      |
                     silu
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      X
              Scale   |   Bias
                   \  |  /
                group_norm_silu
                      |
                     Out
*/

namespace {

class GroupNormSiluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "GroupNormSiluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &groupnorm = pat.Op(
        paddle::dialect::GroupNormOp::name(),
        {{"epsilon", pat.Attr("epsilon")}, {"groups", pat.Attr("groups")}});

    const auto &silu = pat.Op(paddle::dialect::SiluOp::name());

    groupnorm({&pat.Tensor("X"), &pat.Tensor("Scale"), &pat.Tensor("Bias")},
              {&pat.Tensor("Y"), &pat.Tensor("Mean"), &pat.Tensor("Variance")});
    silu({&pat.Tensor("Y")}, {&pat.Tensor("Out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &group_norm_silu_xpu = res.Op(
        paddle::dialect::GroupNormSiluXpuOp::name(),
        {{{"epsilon", pat.Attr("epsilon")}, {"groups", pat.Attr("groups")}}});
    group_norm_silu_xpu(
        {&res.Tensor("X"), &res.Tensor("Scale"), &res.Tensor("Bias")},
        {&res.Tensor("Out")});
  }
};

class GroupNormSiluXpuFusePass : public pir::PatternRewritePass {
 public:
  GroupNormSiluXpuFusePass()
      : pir::PatternRewritePass("group_norm_silu_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<GroupNormSiluPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateGroupNormSiluXpuFusePass() {
  return std::make_unique<GroupNormSiluXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(group_norm_silu_xpu_fuse_pass, GroupNormSiluXpuFusePass);
