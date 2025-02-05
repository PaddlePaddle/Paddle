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

#include "paddle/fluid/pir/transforms/onednn/conv2d_bn_onednn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Conv2dBnOneDNNFusePattern
    : public pir::OpRewritePattern<paddle::dialect::BatchNorm_Op> {
 public:
  using pir::OpRewritePattern<paddle::dialect::BatchNorm_Op>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::BatchNorm_Op op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The prev op should be conv2d op.
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    auto conv2d_attributes = conv2d_op->attributes();
    auto padding_algorithm = conv2d_attributes.at("padding_algorithm")
                                 .dyn_cast<pir::StrAttribute>()
                                 .AsString();
    if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
        padding_algorithm != "VALID") {
      return false;
    }
    auto data_format = conv2d_attributes.at("data_format")
                           .dyn_cast<pir::StrAttribute>()
                           .AsString();
    if (data_format != "NCHW" && data_format != "AnyLayout" &&
        data_format != "NHWC") {
      return false;
    }
    auto groups =
        conv2d_attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();
    if (groups < 1) {
      return false;
    }
    if (!conv2d_op.out().HasOneUse()) return false;
    // The bn outputs(mean_out\variance_out\saved_mean\saved_variance)
    //  cannot be used in conv bn fusion
    if (!op.mean_out().use_empty()) return false;
    if (!op.variance_out().use_empty()) return false;
    if (!op.saved_mean().use_empty()) return false;
    if (!op.saved_variance().use_empty()) return false;

    pir::Value conv2d_filter = conv2d_op.filter();
    pir::Value bn_mean = op.mean();
    pir::Value bn_variance = op.variance();
    pir::Value bn_scale = op.scale();
    pir::Value bn_bias = op.bias();

    // --- deal with filter ---
    auto bn_variance_shape = pir::GetShapeFromValue(bn_variance);
    float epsilon = op.attribute<pir::FloatAttribute>("epsilon").data();
    if (epsilon < 0.0f || epsilon > 0.001f) {
      return false;
    }
    paddle::dialect::FullOp full_op =
        rewriter.Build<paddle::dialect::FullOp>(bn_variance_shape, epsilon);
    paddle::dialect::AddOp add_op =
        rewriter.Build<paddle::dialect::AddOp>(bn_variance, full_op.out());
    paddle::dialect::SqrtOp sqrt_op =
        rewriter.Build<paddle::dialect::SqrtOp>(add_op.out());
    paddle::dialect::DivideOp div_op =
        rewriter.Build<paddle::dialect::DivideOp>(bn_scale, sqrt_op.out());
    // reshape scale
    auto conv2d_filter_shape = pir::GetShapeFromValue(conv2d_filter);
    auto bn_scale_shape = pir::GetShapeFromValue(bn_scale);
    std::vector<int64_t> bn_scale_new_shape(conv2d_filter_shape.size(), 1);
    bn_scale_new_shape[0] = bn_scale_shape[0];
    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(div_op.out(),
                                                   bn_scale_new_shape);

    paddle::onednn::dialect::FusedConv2dOp new_conv2d_op;

    conv2d_attributes["force_fp32_output"] = rewriter.bool_attr(false);
    conv2d_attributes["fuse_residual_connection"] = rewriter.bool_attr(false);
    conv2d_attributes["mkldnn_data_type"] = rewriter.str_attr("float32");
    conv2d_attributes["fuse_activation"] = rewriter.str_attr("");
    conv2d_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    conv2d_attributes["fuse_beta"] = rewriter.float_attr(0.0f);
    conv2d_attributes["scale_in"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_out"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_in_eltwise"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_weights"] =
        rewriter.array_attr({rewriter.float_attr(1.0f)});

    auto conv2d_filter_dtype = pir::GetDataTypeFromValue(conv2d_filter);
    if (conv2d_filter_dtype.isa<pir::Float16Type>()) {
      return false;
    }
    auto mul_op = rewriter.Build<paddle::dialect::MultiplyOp>(
        conv2d_filter, reshape_scale_op.out());

    // --- deal with bias ---
    paddle::dialect::MultiplyOp mul_bias_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(bn_mean, div_op.out());
    // new bias --> sub_op.out()
    paddle::dialect::SubtractOp sub_op =
        rewriter.Build<paddle::dialect::SubtractOp>(bn_bias, mul_bias_op.out());
    // fuse new bias to fused_conv2d
    new_conv2d_op = rewriter.Build<paddle::onednn::dialect::FusedConv2dOp>(
        conv2d_op.input(),
        mul_op.out(),
        sub_op.out(),
        pir::Value{},
        conv2d_attributes);

    rewriter.ReplaceAllUsesWith(op.out(), new_conv2d_op.output());

    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dBiasBnOneDNNFusePattern
    : public pir::OpRewritePattern<paddle::dialect::BatchNorm_Op> {
 public:
  using pir::OpRewritePattern<paddle::dialect::BatchNorm_Op>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::BatchNorm_Op op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The prev op should be add op.
    paddle::dialect::AddOp add_op =
        pir::GetDefiningOpForInput(op, 0)->dyn_cast<paddle::dialect::AddOp>();
    if (!add_op) return false;
    // The prev prev op should be conv2d op.
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(add_op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    auto conv2d_attributes = conv2d_op->attributes();
    auto padding_algorithm = conv2d_attributes.at("padding_algorithm")
                                 .dyn_cast<pir::StrAttribute>()
                                 .AsString();
    if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
        padding_algorithm != "VALID") {
      return false;
    }
    auto data_format = conv2d_attributes.at("data_format")
                           .dyn_cast<pir::StrAttribute>()
                           .AsString();
    if (data_format != "NCHW" && data_format != "AnyLayout" &&
        data_format != "NHWC") {
      return false;
    }
    auto groups =
        conv2d_attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();
    if (groups < 1) {
      return false;
    }
    if (!conv2d_op.out().HasOneUse()) return false;
    if (!add_op.out().HasOneUse()) return false;
    // The bn outputs(mean_out\variance_out\saved_mean\saved_variance)
    //  cannot be used in conv bn fusion
    if (!op.mean_out().use_empty()) return false;
    if (!op.variance_out().use_empty()) return false;
    if (!op.saved_mean().use_empty()) return false;
    if (!op.saved_variance().use_empty()) return false;

    pir::Value add_y = add_op.y();
    if (!pir::ValueIsPersistable(add_y)) return false;
    pir::Value conv2d_filter = conv2d_op.filter();
    pir::Value bn_mean = op.mean();
    pir::Value bn_variance = op.variance();
    pir::Value bn_scale = op.scale();
    pir::Value bn_bias = op.bias();

    auto add_y_shape = pir::GetShapeFromValue(add_y);
    auto bn_bias_shape = pir::GetShapeFromValue(bn_bias);
    std::vector<int64_t> add_y_new_shape{add_y_shape[0]};
    // Support both per_tensor & per_channel addition
    if (add_y_shape.size() != 1) {
      size_t idx;
      if (data_format == "NHWC") {
        idx = add_y_shape.size() - 1;
      } else {
        idx = 1;
      }
      if (add_y_shape[idx] != bn_bias_shape[0]) return false;
      bool is_ok = true;
      for (size_t i = 0; i < add_y_shape.size(); i++) {
        if (i == idx) continue;
        if (add_y_shape[i] != 1) {
          is_ok = false;
          break;
        }
      }
      if (!is_ok) return false;
      // reshape add_y from [1, X, 1, 1] (NCHW) to [X]
      add_y_new_shape[0] = add_y_shape[idx];
    } else if (add_y_shape[0] != 1) {
      return false;
    }
    paddle::dialect::ReshapeOp reshape_add_y_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(add_op.y(), add_y_new_shape);

    // --- deal with filter ---
    auto bn_variance_shape = pir::GetShapeFromValue(bn_variance);
    float epsilon = op.attribute<pir::FloatAttribute>("epsilon").data();
    if (epsilon < 0.0f || epsilon > 0.001f) {
      return false;
    }
    paddle::dialect::FullOp full_op =
        rewriter.Build<paddle::dialect::FullOp>(bn_variance_shape, epsilon);
    paddle::dialect::AddOp add_op_1 =
        rewriter.Build<paddle::dialect::AddOp>(bn_variance, full_op.out());
    paddle::dialect::SqrtOp sqrt_op =
        rewriter.Build<paddle::dialect::SqrtOp>(add_op_1.out());
    paddle::dialect::DivideOp div_op =
        rewriter.Build<paddle::dialect::DivideOp>(bn_scale, sqrt_op.out());
    // reshape scale
    auto conv2d_filter_shape = pir::GetShapeFromValue(conv2d_filter);
    auto bn_scale_shape = pir::GetShapeFromValue(bn_scale);
    std::vector<int64_t> bn_scale_new_shape(conv2d_filter_shape.size(), 1);
    bn_scale_new_shape[0] = bn_scale_shape[0];
    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(div_op.out(),
                                                   bn_scale_new_shape);

    paddle::onednn::dialect::FusedConv2dOp new_conv2d_op;

    conv2d_attributes["force_fp32_output"] = rewriter.bool_attr(false);
    conv2d_attributes["fuse_residual_connection"] = rewriter.bool_attr(false);
    conv2d_attributes["mkldnn_data_type"] = rewriter.str_attr("float32");
    conv2d_attributes["fuse_activation"] = rewriter.str_attr("");
    conv2d_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    conv2d_attributes["fuse_beta"] = rewriter.float_attr(0.0f);
    conv2d_attributes["scale_in"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_out"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_in_eltwise"] = rewriter.float_attr(1.0f);
    conv2d_attributes["scale_weights"] =
        rewriter.array_attr({rewriter.float_attr(1.0f)});

    auto conv2d_filter_dtype = pir::GetDataTypeFromValue(conv2d_filter);
    if (conv2d_filter_dtype.isa<pir::Float16Type>()) {
      return false;
    }
    auto mul_op = rewriter.Build<paddle::dialect::MultiplyOp>(
        conv2d_filter, reshape_scale_op.out());

    // --- deal with bias ---
    // (add_op.y()(reshaped) - bn_mean)*scale + bn_bias
    paddle::dialect::SubtractOp sub_op_1 =
        rewriter.Build<paddle::dialect::SubtractOp>(reshape_add_y_op.out(),
                                                    bn_mean);
    paddle::dialect::MultiplyOp mul_bias_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(sub_op_1.out(),
                                                    div_op.out());
    paddle::dialect::AddOp add_op_2 =
        rewriter.Build<paddle::dialect::AddOp>(bn_bias, mul_bias_op.out());
    // fuse new bias to fused_conv2d
    new_conv2d_op = rewriter.Build<paddle::onednn::dialect::FusedConv2dOp>(
        conv2d_op.input(),
        mul_op.out(),
        add_op_2.out(),
        pir::Value{},
        conv2d_attributes);

    rewriter.ReplaceAllUsesWith(op.out(), new_conv2d_op.output());

    rewriter.EraseOp(op);
    rewriter.EraseOp(add_op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dBnOneDNNFusePass : public pir::PatternRewritePass {
 public:
  Conv2dBnOneDNNFusePass()
      : pir::PatternRewritePass("conv2d_bn_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    auto conv_bn_onednn_pattern = std::make_unique<Conv2dBnOneDNNFusePattern>(
        context,
        1,
        std::vector<std::string>{
            paddle::dialect::FullOp::name(),
            paddle::dialect::AddOp::name(),
            paddle::dialect::SqrtOp::name(),
            paddle::dialect::DivideOp::name(),
            paddle::dialect::ReshapeOp::name(),
            paddle::dialect::MultiplyOp::name(),
            paddle::dialect::SubtractOp::name(),
            paddle::dialect::Conv2dOp::name(),
            paddle::onednn::dialect::FusedConv2dOp::name(),
        });

    // Currently BN on graph are all BN_, so no need to add BN replace pattern
    // conv2d+bn_->conv2d
    ps.Add(std::move(conv_bn_onednn_pattern));
    return ps;
  }
};

class Conv2dBiasBnOneDNNFusePass : public pir::PatternRewritePass {
 public:
  Conv2dBiasBnOneDNNFusePass()
      : pir::PatternRewritePass("conv2d_bias_bn_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    auto conv_bias_bn_onednn_pattern =
        std::make_unique<Conv2dBiasBnOneDNNFusePattern>(
            context,
            1,
            std::vector<std::string>{
                paddle::dialect::FullOp::name(),
                paddle::dialect::AddOp::name(),
                paddle::dialect::SqrtOp::name(),
                paddle::dialect::DivideOp::name(),
                paddle::dialect::ReshapeOp::name(),
                paddle::dialect::MultiplyOp::name(),
                paddle::dialect::SubtractOp::name(),
                paddle::dialect::Conv2dOp::name(),
                paddle::onednn::dialect::FusedConv2dOp::name(),
            });

    // Currently BN on graph are all BN_, so no need to add BN replace pattern
    // conv2d+bias+bn_->conv2d
    ps.Add(std::move(conv_bias_bn_onednn_pattern));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dBnOneDNNFusePass() {
  return std::make_unique<Conv2dBnOneDNNFusePass>();
}

std::unique_ptr<Pass> CreateConv2dBiasBnOneDNNFusePass() {
  return std::make_unique<Conv2dBiasBnOneDNNFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_bn_onednn_fuse_pass, Conv2dBnOneDNNFusePass);
REGISTER_IR_PASS(conv2d_bias_bn_onednn_fuse_pass, Conv2dBiasBnOneDNNFusePass);
