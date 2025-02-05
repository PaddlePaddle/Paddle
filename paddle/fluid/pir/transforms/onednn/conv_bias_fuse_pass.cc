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

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class ConvBiasFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string conv_name_;
  std::string fused_conv_name_;

 public:
  ConvBiasFusePattern(std::string conv_name, std::string fused_conv_name)
      : conv_name_(std::move(conv_name)),
        fused_conv_name_(std::move(fused_conv_name)) {}

  std::string name() const override { return "ConvBiasFusePattern"; }

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
         {&pat.Tensor("conv_out")});

    pat.Tensor("add_out") = add(pat.Tensor("conv_out"), pat.Tensor("bias"));

    if (conv_name_ == paddle::dialect::Conv2dOp::name()) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
          return false;
        }

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
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
          return false;
        }

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
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      auto output_shape = pir::GetShapeFromValue(match_ctx.Tensor("conv_out"));
      if (bias_shape.size() != 1) {
        if (bias_shape[1] != output_shape[1]) return false;
        bool is_ok = true;
        for (size_t i = 0; i < bias_shape.size(); i++) {
          if (i == 1) continue;
          if (bias_shape[i] != 1) {
            is_ok = false;
            break;
          }
        }
        return is_ok;
      }
      return true;
    });

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
                &res.InputNoneTensor()},
               {&res.Tensor("add_out")});
  }
};

class ConvTransposeBiasFusePattern : public paddle::drr::DrrPatternBase {
  std::string name() const override { return "ConvTransposeBiasFusePattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &conv =
        pat.Op(paddle::dialect::Conv2dTransposeOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"output_padding", pat.Attr("output_padding")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("output_size")},
         {&pat.Tensor("conv_out")});

    pat.Tensor("add_out") = add(pat.Tensor("conv_out"), pat.Tensor("bias"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
        return false;
      }

      std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
      std::set<std::string> data_format = {"NCHW", "NHWC", "AnyLayout"};
      if (padding_algorithm.count(
              match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
          data_format.count(match_ctx.Attr<std::string>("data_format")) == 0 ||
          match_ctx.Attr<int>("groups") < 1) {
        return false;
      }
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      if (bias_shape.size() != 1) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"output_padding", pat.Attr("output_padding")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_relu", res.BoolAttr(false)},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"is_test", res.BoolAttr(true)},
               }});

    fused_conv({&res.Tensor("input"),
                &res.Tensor("filter"),
                &res.Tensor("bias"),
                &res.Tensor("output_size")},
               {&res.Tensor("add_out")});
  }
};

class FusedConvTransposeAddFusePattern : public paddle::drr::DrrPatternBase {
  std::string name() const override {
    return "FusedConvTransposeAddFusePattern";
  }

  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv =
        pat.Op(paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"output_padding", pat.Attr("output_padding")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"force_fp32_output", pat.Attr("force_fp32_output")},
                {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                {"fuse_relu", pat.Attr("fuse_relu")},
                {"fuse_activation", pat.Attr("fuse_activation")},
                {"fuse_alpha", pat.Attr("fuse_alpha")},
                {"fuse_beta", pat.Attr("fuse_beta")},
                {"is_test", pat.Attr("is_test")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("bias"),
          &pat.Tensor("output_size")},
         {&pat.Tensor("conv_out")});

    pat.Tensor("result") =
        add(pat.Tensor("conv_out"), pat.Tensor("other_param"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("other_param"))) {
        return false;
      }

      std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
      std::set<std::string> data_format = {"NCHW", "NHWC", "AnyLayout"};
      if (padding_algorithm.count(
              match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
          data_format.count(match_ctx.Attr<std::string>("data_format")) == 0 ||
          match_ctx.Attr<int>("groups") < 1) {
        return false;
      }
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      auto other_param_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("other_param"));
      if (bias_shape != other_param_shape) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("bias2") =
        fused_add(res.Tensor("bias"), res.Tensor("other_param"));

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"output_padding", pat.Attr("output_padding")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"fuse_relu", pat.Attr("fuse_relu")},
                   {"fuse_activation", pat.Attr("fuse_activation")},
                   {"fuse_alpha", pat.Attr("fuse_alpha")},
                   {"fuse_beta", pat.Attr("fuse_beta")},
                   {"is_test", pat.Attr("is_test")},
               }});

    fused_conv({&res.Tensor("input"),
                &res.Tensor("filter"),
                &res.Tensor("bias2"),
                &res.Tensor("output_size")},
               {&res.Tensor("result")});
  }
};

class Conv2dBiasFusePass : public pir::PatternRewritePass {
 public:
  Conv2dBiasFusePass() : pir::PatternRewritePass("conv2d_bias_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ConvBiasFusePattern>(
        context,
        paddle::dialect::Conv2dOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()));
    return ps;
  }
};

class Conv2dTransposeBiasFusePass : public pir::PatternRewritePass {
 public:
  Conv2dTransposeBiasFusePass()
      : pir::PatternRewritePass("conv2d_transpose_bias_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ConvTransposeBiasFusePattern>(context));
    ps.Add(paddle::drr::Create<FusedConvTransposeAddFusePattern>(context));
    return ps;
  }
};

class Conv3dBiasFusePass : public pir::PatternRewritePass {
 public:
  Conv3dBiasFusePass() : pir::PatternRewritePass("conv3d_bias_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ConvBiasFusePattern>(
        context,
        paddle::dialect::Conv3dOp::name(),
        paddle::onednn::dialect::FusedConv3dOp::name()));
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

std::unique_ptr<Pass> CreateConv2dTransposeBiasFusePass() {
  // pd_op.conv2d_transpose + pd_op.add -> onednn_op.conv2d_transpose_bias
  // onednn_op.conv2d_transpose_bias + pd_op.add ->
  // onednn_op.conv2d_transpose_bias + pd_op.add
  return std::make_unique<Conv2dTransposeBiasFusePass>();
}

std::unique_ptr<Pass> CreateConv3dBiasFusePass() {
  // pd_op.conv3d + pd_op.add -> onednn_op.fused_conv3d
  // onednn_op.fused_conv3d + pd_op.add -> onednn_op.fused_conv3d
  return std::make_unique<Conv3dBiasFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(conv2d_bias_fuse_pass, Conv2dBiasFusePass);
REGISTER_IR_PASS(conv2d_transpose_bias_fuse_pass, Conv2dTransposeBiasFusePass);
REGISTER_IR_PASS(conv3d_bias_fuse_pass, Conv3dBiasFusePass);
