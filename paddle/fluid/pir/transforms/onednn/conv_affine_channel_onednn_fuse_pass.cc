// REGISTER_IR_PASS(onednn_placement_pass, AffineChannelPass);
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

#include "paddle/fluid/pir/transforms/onednn/conv_affine_channel_onednn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

class Conv2dAffineChannelPattern
    : public pir::OpRewritePattern<paddle::dialect::AffineChannelOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::AffineChannelOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AffineChannelOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The prev op should be conv2d op.
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;
    pir::Value conv_weight = conv2d_op.filter();
    pir::Value ac_bias = op.Bias();
    pir::Value ac_scale = op.Scale();

    // Re-compute weight of conv2d from AffineChannel

    auto weight_shape = pir::GetShapeFromValue(conv_weight);
    auto ac_scale_shape = pir::GetShapeFromValue(ac_scale);
    std::vector<int64_t> ac_scale_new_shape(weight_shape.size(), 1);
    ac_scale_new_shape[0] = ac_scale_shape[0];
    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(ac_scale,
                                                   ac_scale_new_shape);

    paddle::dialect::MultiplyOp weight_value_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(conv_weight,
                                                    reshape_scale_op.out());

    auto conv2d_attributes = conv2d_op->attributes();
    auto new_conv2d_op = rewriter.Build<paddle::dialect::Conv2dOp>(
        conv2d_op.input(), weight_value_op.out(), conv2d_attributes);

    // reshape new bias
    auto new_conv2d_out_shape = pir::GetShapeFromValue(new_conv2d_op.out());
    std::vector<int64_t> new_bias_new_shape(new_conv2d_out_shape.size(), 1);
    new_bias_new_shape[1] = new_conv2d_out_shape[1];
    paddle::dialect::ReshapeOp reshape_bias_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(ac_bias, new_bias_new_shape);

    auto elementwise_add = rewriter.Build<paddle::dialect::AddOp>(
        new_conv2d_op.out(), reshape_bias_op.out());

    rewriter.ReplaceAllUsesWith(op.out(), elementwise_add.out());
    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class FusedConv2dAffineChannelPattern
    : public pir::OpRewritePattern<paddle::dialect::AffineChannelOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::AffineChannelOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AffineChannelOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    paddle::onednn::dialect::FusedConv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::onednn::dialect::FusedConv2dOp>();
    if (!conv2d_op) return false;
    pir::Value conv_weight = conv2d_op.filter();
    pir::Value ac_bias = op.Bias();
    pir::Value ac_scale = op.Scale();

    auto weight_shape = pir::GetShapeFromValue(conv_weight);
    auto ac_scale_shape = pir::GetShapeFromValue(ac_scale);
    std::vector<int64_t> ac_scale_new_shape(weight_shape.size(), 1);
    ac_scale_new_shape[0] = ac_scale_shape[0];
    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(ac_scale,
                                                   ac_scale_new_shape);

    paddle::dialect::MultiplyOp weight_value_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(conv_weight,
                                                    reshape_scale_op.out());

    auto conv2d_attributes = conv2d_op->attributes();
    auto new_conv2d_op = rewriter.Build<paddle::onednn::dialect::FusedConv2dOp>(
        conv2d_op.input(),
        weight_value_op.out(),
        conv2d_op.bias(),
        conv2d_op.residual_param(),
        conv2d_attributes);

    auto new_conv2d_out_shape = pir::GetShapeFromValue(new_conv2d_op.output());
    std::vector<int64_t> new_bias_new_shape(new_conv2d_out_shape.size(), 1);
    new_bias_new_shape[1] = new_conv2d_out_shape[1];
    paddle::dialect::ReshapeOp reshape_bias_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(ac_bias, new_bias_new_shape);

    auto elementwise_add = rewriter.Build<paddle::dialect::AddOp>(
        new_conv2d_op.output(), reshape_bias_op.out());

    rewriter.ReplaceAllUsesWith(op.out(), elementwise_add.out());
    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class AffineChannelPass : public pir::PatternRewritePass {
 public:
  AffineChannelPass()
      : pir::PatternRewritePass("conv_affine_channel_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    auto pattern = std::make_unique<Conv2dAffineChannelPattern>(
        context, 1, std::vector<std::string>{});
    ps.Add(std::move(pattern));
    auto fused_pattern = std::make_unique<FusedConv2dAffineChannelPattern>(
        context, 1, std::vector<std::string>{});
    ps.Add(std::move(fused_pattern));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAffineChannelPass() {
  return std::make_unique<AffineChannelPass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv_affine_channel_onednn_fuse_pass, AffineChannelPass);
