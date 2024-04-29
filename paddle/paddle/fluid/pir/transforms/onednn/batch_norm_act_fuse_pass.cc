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

#include "paddle/fluid/pir/transforms/onednn/batch_norm_act_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class BatchNormActFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bn_name_;
  std::string fused_bn_name_;

 public:
  BatchNormActFusePattern(const std::string &bn_name,
                          const std::string &fused_bn_name)
      : bn_name_(bn_name), fused_bn_name_(fused_bn_name) {}

  std::string name() const override { return "BatchNormActFusePattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &bn =
        pat.Op(bn_name_,
               {{"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")},
                {"is_test", pat.Attr("is_test")}});
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());
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
    pat.Tensor("relu_out") = relu(pat.Tensor("bn_out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      float epsilon = match_ctx.Attr<float>("epsilon");
      if (epsilon < 0.0 || epsilon > 0.001 ||
          match_ctx.Attr<bool>("trainable_statistics") == true ||
          match_ctx.Attr<bool>("is_test") == false) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_bn =
        res.Op(fused_bn_name_,
               {{
                   {"is_test", res.BoolAttr(true)},
                   {"momentum", pat.Attr("momentum")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"data_format", pat.Attr("data_format")},
                   {"use_global_stats", pat.Attr("use_global_stats")},
                   {"trainable_statistics", res.BoolAttr(false)},
                   {"fuse_with_relu", res.BoolAttr(true)},
               }});

    fused_bn({&res.Tensor("x"),
              &res.Tensor("mean"),
              &res.Tensor("variance"),
              &res.Tensor("scale"),
              &res.Tensor("bias")},
             {&res.Tensor("relu_out"),
              &res.Tensor("mean_out"),
              &res.Tensor("variance_out"),
              &res.Tensor("saved_mean"),
              &res.Tensor("saved_variance"),
              &res.Tensor("reserve_space")});
  }
};

class BatchNormActFusePass : public pir::PatternRewritePass {
 public:
  BatchNormActFusePass()
      : pir::PatternRewritePass("batch_norm_act_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<BatchNormActFusePattern>(
        context,
        paddle::dialect::BatchNormOp::name(),
        paddle::onednn::dialect::BatchNormOp::name()));
    ps.Add(paddle::drr::Create<BatchNormActFusePattern>(
        context,
        paddle::dialect::BatchNorm_Op::name(),
        paddle::onednn::dialect::BatchNorm_Op::name()));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateBatchNormActFusePass() {
  // pd_op.batch_norm + pd_op.relu -> onednn_op.batch_norm
  return std::make_unique<BatchNormActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(batch_norm_act_fuse_pass, BatchNormActFusePass);
