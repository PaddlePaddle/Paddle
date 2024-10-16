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

#include "paddle/fluid/pir/transforms/xpu/bn_act_xpu_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

template <int act_type>
class BnActPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "BnActPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string act_op_name;
    if (act_type == xpu::Activation_t::RELU) {
      act_op_name = paddle::dialect::ReluOp::name();
    } else {
      phi::errors::InvalidArgument("Unsupported activation type.");
    }

    const auto &bn =
        pat.Op(paddle::dialect::BatchNorm_Op::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});

    const auto &act = pat.Op(act_op_name);

    bn({&pat.Tensor("x"),
        &pat.Tensor("mean"),
        &pat.Tensor("variance"),
        &pat.Tensor("scale"),
        &pat.Tensor("bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("variance_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    pat.Tensor("out") = act(pat.Tensor("bn_out"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &bn_act_xpu = res.Op(paddle::dialect::BnActXpuOp::name(),
                                    {{"epsilon", pat.Attr("epsilon")},
                                     {"momentum", pat.Attr("momentum")},
                                     {"data_format", pat.Attr("data_format")},
                                     {"act_type", res.Int32Attr(act_type)}});
    bn_act_xpu({&res.Tensor("x"),
                &res.Tensor("bias"),
                &res.Tensor("mean"),
                &res.Tensor("scale"),
                &res.Tensor("variance")},
               {&res.Tensor("out")});
  }
};

class BnActXpuFusePass : public pir::PatternRewritePass {
 public:
  BnActXpuFusePass() : pir::PatternRewritePass("bn_act_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<BnActPattern<xpu::Activation_t::RELU>>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateBnActXpuFusePass() {
  return std::make_unique<BnActXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(bn_act_xpu_fuse_pass, BnActXpuFusePass);
