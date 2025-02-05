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

#include "paddle/fluid/pir/transforms/general/group_norm_silu_fuse_pass.h"

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
After the pass is applied:XPU
                      X
              Scale   |   Bias
                   \  |  /
                group_norm_silu_xpu
                      |
                     Out
------------------------------------------------------
After the pass is applied:GPU
                      X
              Scale   |   Bias
                   \  |  /
                add_group_norm_silu
                      |
                     Out
*/

namespace {

class GroupNormSiluPattern : public paddle::drr::DrrPatternBase {
 private:
  const bool enable_gpu_mixed_;

 public:
  explicit GroupNormSiluPattern(bool enable_gpu_mixed)
      : enable_gpu_mixed_(enable_gpu_mixed) {}

  std::string name() const override { return "GroupNormSiluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &group_norm = pat.Op(paddle::dialect::GroupNormOp::name(),
                                    {{"epsilon", pat.Attr("epsilon")},
                                     {"groups", pat.Attr("groups")},
                                     {"data_format", pat.Attr("data_format")}});

    const auto &silu = pat.Op(paddle::dialect::SiluOp::name());

    group_norm(
        {&pat.Tensor("X"), &pat.Tensor("Scale"), &pat.Tensor("Bias")},
        {&pat.Tensor("Y"), &pat.Tensor("Mean"), &pat.Tensor("Variance")});
    silu({&pat.Tensor("Y")}, {&pat.Tensor("Out")});

#ifdef PADDLE_WITH_CUDA
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("X"));
      if (!this->enable_gpu_mixed_ && !x_dtype.isa<pir::Float16Type>() &&
          !x_dtype.isa<pir::BFloat16Type>()) {
        return false;
      }
      return true;
    });
#endif

#ifdef PADDLE_WITH_CUDA
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &add_group_norm_silu_op =
        res.Op(paddle::dialect::AddGroupNormSiluOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"activation", res.StrAttr("silu")}});
    add_group_norm_silu_op({&res.Tensor("X"),
                            &res.InputNoneTensor(),
                            &res.Tensor("Scale"),
                            &res.Tensor("Bias")},
                           {&res.Tensor("Out"),
                            &res.OutputNoneTensor(),
                            &res.Tensor("Mean"),
                            &res.Tensor("Variance")});
#endif
#ifdef PADDLE_WITH_XPU
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &group_norm_silu_xpu = res.Op(
        paddle::dialect::GroupNormSiluXpuOp::name(),
        {{{"epsilon", pat.Attr("epsilon")}, {"groups", pat.Attr("groups")}}});
    group_norm_silu_xpu(
        {&res.Tensor("X"), &res.Tensor("Scale"), &res.Tensor("Bias")},
        {&res.Tensor("Out")});
#endif
  }
};

class GroupNormSiluFusePass : public pir::PatternRewritePass {
 public:
  GroupNormSiluFusePass()
      : pir::PatternRewritePass("group_norm_silu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    bool enable_gpu_mixed = false;
    if (Has("enable_gpu_mixed")) {
      enable_gpu_mixed = Get<bool>("enable_gpu_mixed");
    }
    ps.Add(
        paddle::drr::Create<GroupNormSiluPattern>(context, enable_gpu_mixed));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateGroupNormSiluFusePass() {
  return std::make_unique<GroupNormSiluFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(group_norm_silu_fuse_pass, GroupNormSiluFusePass);
