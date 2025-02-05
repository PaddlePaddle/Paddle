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

#include "paddle/fluid/pir/transforms/onednn/conv_elementwise_add_onednn_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class ConvElementwiseAddPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string conv_name_;
  std::string fused_conv_name_;

 public:
  ConvElementwiseAddPattern(std::string conv_name, std::string fused_conv_name)
      : conv_name_(std::move(conv_name)),
        fused_conv_name_(std::move(fused_conv_name)) {}

  std::string name() const override { return "ConvElementwiseAddPattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &conv =
        pat.Op(conv_name_,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv2d_out")});

    pat.Tensor("add_out") =
        add(pat.Tensor("conv2d_out"), pat.Tensor("residual_param"));
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout") {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto conv2d_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto residual_param_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual_param"));
      // conv_elementwise_add_onednn_fuse_pass does not support broadcast
      if (conv2d_out_shape != residual_param_shape) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add =
        res.Op(fused_conv_name_,
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_residual_connection", res.BoolAttr(true)},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    fused_conv2d_add({&res.Tensor("input"),
                      &res.Tensor("filter"),
                      &res.InputNoneTensor(),
                      &res.Tensor("residual_param")},
                     {&res.Tensor("add_out")});
  }
};

class ConvElementwiseAddAsYPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string conv_name_;
  std::string fused_conv_name_;

 public:
  ConvElementwiseAddAsYPattern(std::string conv_name,
                               std::string fused_conv_name)
      : conv_name_(std::move(conv_name)),
        fused_conv_name_(std::move(fused_conv_name)) {}

  std::string name() const override { return "ConvElementwiseAddAsYPattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &conv =
        pat.Op(conv_name_,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv2d_out")});
    pat.Tensor("add_out") =
        add(pat.Tensor("residual_param"), pat.Tensor("conv2d_out"));

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout") {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto conv2d_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto residual_param_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual_param"));
      // conv_elementwise_add_onednn_fuse_pass does not support broadcast
      if (conv2d_out_shape != residual_param_shape) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add =
        res.Op(fused_conv_name_,
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_residual_connection", res.BoolAttr(true)},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    fused_conv2d_add({&res.Tensor("input"),
                      &res.Tensor("filter"),
                      &res.InputNoneTensor(),
                      &res.Tensor("residual_param")},
                     {&res.Tensor("add_out")});
  }
};

class FusedConvBiasElementwiseAddPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string conv_name_;
  std::string fused_conv_name_;

 public:
  FusedConvBiasElementwiseAddPattern(std::string conv_name,
                                     std::string fused_conv_name)
      : conv_name_(std::move(conv_name)),
        fused_conv_name_(std::move(fused_conv_name)) {}

  std::string name() const override {
    return "FusedConvBiasElementwiseAddPattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv = pat.Op(
        conv_name_,
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
            {"fuse_activation", pat.Attr("fuse_activation")},
            {"fuse_residual_connection", pat.Attr("fuse_residual_connection")},
            {"force_fp32_output", pat.Attr("force_fp32_output")},
            {"fuse_alpha", pat.Attr("fuse_alpha")},
            {"fuse_beta", pat.Attr("fuse_beta")},
            {"scale_in", pat.Attr("scale_in")},
            {"scale_out", pat.Attr("scale_out")},
            {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
            {"scale_weights", pat.Attr("scale_weights")},
        }});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("bias"),
          &pat.InputNoneTensor()},
         {&pat.Tensor("conv2d_out")});

    pat.Tensor("add_out") =
        add(pat.Tensor("conv2d_out"), pat.Tensor("residual_param"));
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout") {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto fuse_activation = match_ctx.Attr<std::string>("fuse_activation");
      auto fuse_residual_connection =
          match_ctx.Attr<bool>("fuse_residual_connection");
      if (!fuse_activation.empty() || fuse_residual_connection) {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto conv2d_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto residual_param_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual_param"));
      // conv_elementwise_add_onednn_fuse_pass does not support broadcast
      if (conv2d_out_shape != residual_param_shape) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add =
        res.Op(fused_conv_name_,
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"fuse_activation", pat.Attr("fuse_activation")},
                   {"fuse_residual_connection", res.BoolAttr(true)},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
                   {"fuse_alpha", pat.Attr("fuse_alpha")},
                   {"fuse_beta", pat.Attr("fuse_beta")},
                   {"scale_in", pat.Attr("scale_in")},
                   {"scale_out", pat.Attr("scale_out")},
                   {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
                   {"scale_weights", pat.Attr("scale_weights")},
               }});

    fused_conv2d_add({&res.Tensor("input"),
                      &res.Tensor("filter"),
                      &res.Tensor("bias"),
                      &res.Tensor("residual_param")},
                     {&res.Tensor("add_out")});
  }
};

class FusedConvBiasElementwiseAddAsYPattern
    : public paddle::drr::DrrPatternBase {
 private:
  std::string conv_name_;
  std::string fused_conv_name_;

 public:
  FusedConvBiasElementwiseAddAsYPattern(std::string conv_name,
                                        std::string fused_conv_name)
      : conv_name_(std::move(conv_name)),
        fused_conv_name_(std::move(fused_conv_name)) {}

  std::string name() const override {
    return "FusedConvBiasElementwiseAddAsYPattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &conv = pat.Op(
        conv_name_,
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
            {"fuse_activation", pat.Attr("fuse_activation")},
            {"fuse_residual_connection", pat.Attr("fuse_residual_connection")},
            {"force_fp32_output", pat.Attr("force_fp32_output")},
            {"fuse_alpha", pat.Attr("fuse_alpha")},
            {"fuse_beta", pat.Attr("fuse_beta")},
            {"scale_in", pat.Attr("scale_in")},
            {"scale_out", pat.Attr("scale_out")},
            {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
            {"scale_weights", pat.Attr("scale_weights")},
        }});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("bias"),
          &pat.InputNoneTensor()},
         {&pat.Tensor("conv2d_out")});

    pat.Tensor("add_out") =
        add(pat.Tensor("residual_param"), pat.Tensor("conv2d_out"));
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout") {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto fuse_activation = match_ctx.Attr<std::string>("fuse_activation");
      auto fuse_residual_connection =
          match_ctx.Attr<bool>("fuse_residual_connection");
      if (!fuse_activation.empty() || fuse_residual_connection) {
        return false;
      }
      return true;
    });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto conv2d_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto residual_param_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual_param"));
      // conv_elementwise_add_onednn_fuse_pass does not support broadcast
      if (conv2d_out_shape != residual_param_shape) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add =
        res.Op(fused_conv_name_,
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"fuse_activation", pat.Attr("fuse_activation")},
                   {"fuse_residual_connection", res.BoolAttr(true)},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
                   {"fuse_alpha", pat.Attr("fuse_alpha")},
                   {"fuse_beta", pat.Attr("fuse_beta")},
                   {"scale_in", pat.Attr("scale_in")},
                   {"scale_out", pat.Attr("scale_out")},
                   {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
                   {"scale_weights", pat.Attr("scale_weights")},
               }});

    fused_conv2d_add({&res.Tensor("input"),
                      &res.Tensor("filter"),
                      &res.Tensor("bias"),
                      &res.Tensor("residual_param")},
                     {&res.Tensor("add_out")});
  }
};

class ConvElementwiseAddFusePass : public pir::PatternRewritePass {
 public:
  ConvElementwiseAddFusePass()
      : pir::PatternRewritePass("conv_elementwise_add_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ConvElementwiseAddPattern>(
        context,
        paddle::dialect::Conv2dOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()));
    ps.Add(paddle::drr::Create<ConvElementwiseAddAsYPattern>(
        context,
        paddle::dialect::Conv2dOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()));
    // conv + bias -> fused_conv2d, fused_conv2d + residual -> fused_conv2d
    ps.Add(paddle::drr::Create<FusedConvBiasElementwiseAddPattern>(
        context,
        paddle::onednn::dialect::FusedConv2dOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()));
    ps.Add(paddle::drr::Create<FusedConvBiasElementwiseAddAsYPattern>(
        context,
        paddle::onednn::dialect::FusedConv2dOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConvElementwiseAddFusePass() {
  return std::make_unique<ConvElementwiseAddFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv_elementwise_add_onednn_fuse_pass,
                 ConvElementwiseAddFusePass);
