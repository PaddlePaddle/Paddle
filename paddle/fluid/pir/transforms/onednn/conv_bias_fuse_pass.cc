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

#include "paddle/fluid/pir/transforms/onednn/conv_bias_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {
class ConvBiasFusePattern : public paddle::drr::DrrPatternBase {
 public:
  ConvBiasFusePattern(const std::string &conv_name,
                      const std::string &fused_conv_name)
      : conv_name_(conv_name), fused_conv_name_(fused_conv_name) {}
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
         {&pat.Tensor("conv_out")});
    const auto &parameter_bias = pat.Op(
        pir::ParameterOp::name(), {{"parameter_name", pat.Attr("param_name")}});
    pat.Tensor("bias") = parameter_bias();
    pat.Tensor("add_out") = add(pat.Tensor("conv_out"), pat.Tensor("bias"));

    if (conv_name_ == paddle::dialect::Conv2dOp::name() ||
        conv_name_ == paddle::onednn::dialect::FusedConv2dOp::name()) {
      pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
        std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
        std::set<std::string> data_format = {"NCHW", "NHWC", "AnyLayout"};
        if (padding_algorithm.count(
                match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
            data_format.count(match_ctx.Attr<std::string>("data_format")) ==
                0 ||
            match_ctx.Attr<int>("groups") < 1) {
          return false;
        }
        return true;
      });
    } else {
      pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
        std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
        std::set<std::string> data_format = {"NDHWC", "NCDHW"};
        if (padding_algorithm.count(
                match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
            data_format.count(match_ctx.Attr<std::string>("data_format")) ==
                0 ||
            match_ctx.Attr<int>("groups") < 1) {
          return false;
        }
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv =
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
                   {"fuse_residual_connection", res.BoolAttr(false)},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    fused_conv({&res.Tensor("input"),
                &res.Tensor("filter"),
                &res.Tensor("bias"),
                &res.NoneTensor()},
               {&res.Tensor("add_out")});
  }

  std::string name() const override { return "ConvBiasFusePattern"; }

  uint32_t benefit() const override { return 2; }

 private:
  std::string conv_name_;
  std::string fused_conv_name_;
};

class FusedConvAddFusePattern : public paddle::drr::DrrPatternBase {
 public:
  FusedConvAddFusePattern(const std::string &conv_name,
                          const std::string &fused_conv_name)
      : conv_name_(conv_name), fused_conv_name_(fused_conv_name) {}
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
    const auto &add2 = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv_out")});
    const auto &parameter_bias = pat.Op(
        pir::ParameterOp::name(), {{"parameter_name", pat.Attr("param_name")}});
    pat.Tensor("bias") = parameter_bias();

    pat.Tensor("add_out") = add(pat.Tensor("conv_out"), pat.Tensor("bias"));

    const auto &parameter = pat.Op(
        pir::ParameterOp::name(), {{"parameter_name", pat.Attr("param_name")}});
    pat.Tensor("other_param") = parameter();
    pat.Tensor("result") =
        add2(pat.Tensor("add_out"), pat.Tensor("other_param"));

    if (conv_name_ == paddle::dialect::Conv2dOp::name() ||
        conv_name_ == paddle::onednn::dialect::FusedConv2dOp::name()) {
      pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
        std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
        std::set<std::string> data_format = {"NCHW", "NHWC", "AnyLayout"};
        if (padding_algorithm.count(
                match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
            data_format.count(match_ctx.Attr<std::string>("data_format")) ==
                0 ||
            match_ctx.Attr<int>("groups") < 1) {
          return false;
        }
        return true;
      });
    } else {
      pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
        std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
        std::set<std::string> data_format = {"NDHWC", "NCDHW"};
        if (padding_algorithm.count(
                match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
            data_format.count(match_ctx.Attr<std::string>("data_format")) ==
                0 ||
            match_ctx.Attr<int>("groups") < 1) {
          return false;
        }
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("bias2") =
        fused_add(res.Tensor("bias"), res.Tensor("other_param"));

    const auto &fused_conv =
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
                   {"fuse_residual_connection", res.BoolAttr(false)},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    fused_conv({&res.Tensor("input"),
                &res.Tensor("filter"),
                &res.Tensor("bias2"),
                &res.NoneTensor()},
               {&res.Tensor("result")});
  }

  std::string name() const override { return "FusedConvAddFusePattern"; }

  uint32_t benefit() const override { return 3; }

 private:
  std::string conv_name_;
  std::string fused_conv_name_;
};

class Conv2dBiasFusePass : public pir::PatternRewritePass {
 public:
  Conv2dBiasFusePass() : pir::PatternRewritePass("conv2d_bias_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(ConvBiasFusePattern(paddle::dialect::Conv2dOp::name(),
                               paddle::onednn::dialect::FusedConv2dOp::name())
               .Build(context));
    ps.Add(
        FusedConvAddFusePattern(paddle::dialect::Conv2dOp::name(),
                                paddle::onednn::dialect::FusedConv2dOp::name())
            .Build(context));
    return ps;
  }
};

// class Conv2dTransposeBiasFusePass : public pir::PatternRewritePass {
//  public:
//   Conv2dTransposeBiasFusePass() :
//   pir::PatternRewritePass("conv2d_transpose_bias_fuse_pass", 2) {}

//   pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override
//   {
//     pir::RewritePatternSet ps(context);
//     ps.Add(Conv2dBiasFusePattern().Build(context));
//     return ps;
//   }
// };

class Conv3dBiasFusePass : public pir::PatternRewritePass {
 public:
  Conv3dBiasFusePass() : pir::PatternRewritePass("conv3d_bias_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(ConvBiasFusePattern(paddle::dialect::Conv3dOp::name(),
                               paddle::onednn::dialect::FusedConv3dOp::name())
               .Build(context));
    ps.Add(
        FusedConvAddFusePattern(paddle::dialect::Conv3dOp::name(),
                                paddle::onednn::dialect::FusedConv3dOp::name())
            .Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dBiasFusePass() {
  // pd_op.conv2d + pd_op.add -> onednn_op.fused_conv2d
  // onednn_op.fused_conv2d + pd_op.add -> onednn_op.fused_conv2d + pd_op.add
  return std::make_unique<Conv2dBiasFusePass>();
}

// std::unique_ptr<Pass> CreateConv2dTransposeBiasFusePass() {
//   // pd_op.conv2d_transpose + pd_op.add -> onednn_op.fused_conv2d
//   return std::make_unique<Conv2dTransposeBiasFusePass>();
// }

std::unique_ptr<Pass> CreateConv3dBiasFusePass() {
  // pd_op.conv3d + pd_op.add -> onednn_op.fused_conv3d
  // onednn_op.fused_conv3d + pd_op.add -> onednn_op.fused_conv3d
  return std::make_unique<Conv3dBiasFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(conv2d_bias_fuse_pass, Conv2dBiasFusePass);
// REGISTER_IR_PASS(conv2d_transpose_bias_fuse_pass,
// Conv2dTransposeBiasFusePass);
REGISTER_IR_PASS(conv3d_bias_fuse_pass, Conv3dBiasFusePass);
