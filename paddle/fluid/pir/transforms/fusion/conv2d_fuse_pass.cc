// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"
#include "paddle/phi/core/ddim.h"

namespace {

class Conv2dBnFusePattern
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

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    pir::Value conv2d_filter = conv2d_op.filter();

    pir::OpResult conv2d_filter_result =
        conv2d_filter.dyn_cast<pir::OpResult>();
    IR_ENFORCE(conv2d_filter_result);

    pir::Value bn_input = op.x();
    IR_ENFORCE(bn_input == conv2d_out);

    pir::Value bn_mean = op.mean();
    pir::Value bn_variance = op.variance();
    pir::Value bn_scale = op.scale();
    pir::Value bn_bias = op.bias();

    // --- deal with filter ---
    rewriter.set_insertion_point(op);
    phi::DDim bn_variance_shape =
        bn_variance.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    float epsilon = op.attribute<pir::FloatAttribute>("epsilon").data();
    paddle::dialect::FullOp full_op = rewriter.Build<paddle::dialect::FullOp>(
        phi::vectorize(bn_variance_shape), epsilon);
    paddle::dialect::AddOp add_op = rewriter.Build<paddle::dialect::AddOp>(
        bn_variance.dyn_cast<pir::OpResult>(), full_op.out());
    paddle::dialect::SqrtOp sqrt_op =
        rewriter.Build<paddle::dialect::SqrtOp>(add_op.out());
    paddle::dialect::DivideOp div_op =
        rewriter.Build<paddle::dialect::DivideOp>(
            bn_scale.dyn_cast<pir::OpResult>(), sqrt_op.out());
    // reshape scale
    phi::DDim conv2d_filter_shape = pir::GetShapeFromValue(conv2d_filter);
    phi::DDim bn_scale_shape =
        bn_scale.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    std::vector<int64_t> bn_scale_new_shape(conv2d_filter_shape.size(), 1);
    bn_scale_new_shape[0] = bn_scale_shape[0];
    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(div_op.out(),
                                                   bn_scale_new_shape);
    // new filter --> mul_op.out()
    paddle::dialect::MultiplyOp mul_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(conv2d_filter_result,
                                                    reshape_scale_op.out());

    auto conv2d_attributes = conv2d_op->attributes();
    auto new_conv2d_op = rewriter.Build<paddle::dialect::Conv2dOp>(
        conv2d_op.input().dyn_cast<pir::OpResult>(),
        mul_op.out(),
        conv2d_attributes);

    // --- deal with bias ---
    paddle::dialect::MultiplyOp mul_bias_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(
            bn_mean.dyn_cast<pir::OpResult>(), div_op.out());
    // new bias --> sub_op.out()
    paddle::dialect::SubtractOp sub_op =
        rewriter.Build<paddle::dialect::SubtractOp>(
            bn_bias.dyn_cast<pir::OpResult>(), mul_bias_op.out());
    // reshape new bias
    phi::DDim new_conv2d_out_shape =
        pir::GetShapeFromValue(new_conv2d_op.out());
    std::vector<int64_t> new_bias_new_shape(new_conv2d_out_shape.size(), 1);
    std::string data_format =
        new_conv2d_op.attribute<pir::StrAttribute>("data_format").AsString();
    if (data_format != "NCHW") {
      return false;
    }
    new_bias_new_shape[1] = new_conv2d_out_shape[1];
    paddle::dialect::ReshapeOp reshape_bias_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(sub_op.out(),
                                                   new_bias_new_shape);
    paddle::dialect::AddOp add_bias_op = rewriter.Build<paddle::dialect::AddOp>(
        new_conv2d_op.out(), reshape_bias_op.out());

    rewriter.ReplaceAllUsesWith(op.out(), add_bias_op.out());

    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dAddFusePattern
    : public pir::drr::DrrPatternBase<Conv2dAddFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("bias"));

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &conv2d_fusion = res.Op(
        paddle::dialect::Conv2dFusionOp::name(),
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"activation",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::string { return "identity"; })},
            {"split_channels",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::vector<int> { return {}; })},
            {"exhaustive_search",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
               return true;
             })},
            {"workspace_size_MB",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> int {
               return 32;
             })},
            {"fuse_alpha",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
               return 0.0f;
             })},
        }});

    conv2d_fusion({&res.Tensor("input"),
                   &res.Tensor("filter"),
                   &res.Tensor("bias"),
                   &res.NoneTensor()},
                  {&res.Tensor("add_out")});
  }
};

class Conv2dAddActFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AddOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;
    pir::Value conv2d_filter = conv2d_op.filter();

    pir::OpResult conv2d_filter_result =
        conv2d_filter.dyn_cast<pir::OpResult>();
    IR_ENFORCE(conv2d_filter_result);

    pir::Value add_input = op.x();
    IR_ENFORCE(add_input == conv2d_out);

    pir::OpResult add_out = op.out();
    if (!add_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    if (next_op_list.size() == 0) return false;

    auto next_op = next_op_list[0];
    std::string act_name = "";
    if (next_op->dyn_cast<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->dyn_cast<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->dyn_cast<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") return false;

    auto conv2d_fusion_attributes = conv2d_op->attributes();
    conv2d_fusion_attributes["activation"] = rewriter.str_attr(act_name);
    conv2d_fusion_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    conv2d_fusion_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    conv2d_fusion_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    conv2d_fusion_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op = rewriter.Build<paddle::dialect::Conv2dFusionOp>(
        conv2d_op.input().dyn_cast<pir::OpResult>(),
        conv2d_op.filter().dyn_cast<pir::OpResult>(),
        op.y().dyn_cast<pir::OpResult>(),
        pir::Value{}.dyn_cast<pir::OpResult>(),
        conv2d_fusion_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});
    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dDoubleAddActFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AddOp add2_op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    paddle::dialect::AddOp add1_op = pir::GetDefiningOpForInput(add2_op, 1)
                                         ->dyn_cast<paddle::dialect::AddOp>();
    if (!add1_op) return false;

    pir::OpResult add1_out = add1_op.out();
    if (!add1_out.HasOneUse()) return false;

    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(add1_op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(add2_op, 0);
    if (next_op_list.size() == 0) return false;
    auto next_op = next_op_list[0];
    std::string act_name = "";
    if (next_op->dyn_cast<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->dyn_cast<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->dyn_cast<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") {
      return false;
    }

    auto conv2d_fusion_attributes = conv2d_op->attributes();
    conv2d_fusion_attributes["activation"] = rewriter.str_attr(act_name);
    conv2d_fusion_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    conv2d_fusion_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    conv2d_fusion_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    conv2d_fusion_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op = rewriter.Build<paddle::dialect::Conv2dFusionOp>(
        conv2d_op.input().dyn_cast<pir::OpResult>(),
        conv2d_op.filter().dyn_cast<pir::OpResult>(),
        add1_op.y().dyn_cast<pir::OpResult>(),
        add2_op.x().dyn_cast<pir::OpResult>(),
        conv2d_fusion_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});

    rewriter.EraseOp(add2_op);
    rewriter.EraseOp(add1_op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class BatchNormReplacePattern
    : public pir::OpRewritePattern<paddle::dialect::BatchNormOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::BatchNormOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::BatchNormOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto bn_op = rewriter.Build<paddle::dialect::BatchNorm_Op>(
        op.x().dyn_cast<pir::OpResult>(),
        op.mean().dyn_cast<pir::OpResult>(),
        op.variance().dyn_cast<pir::OpResult>(),
        op.scale().dyn_cast<pir::OpResult>(),
        op.bias().dyn_cast<pir::OpResult>(),
        op->attributes());
    rewriter.ReplaceAllUsesWith(op.out(), bn_op.out());
    rewriter.EraseOp(op);
    return true;
  }
};

class Conv2dFusePass : public pir::PatternRewritePass {
 public:
  Conv2dFusePass() : pir::PatternRewritePass("conv2d_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    auto conv_bn_pattern = std::make_unique<Conv2dBnFusePattern>(
        context,
        1,
        std::vector<std::string>{paddle::dialect::FullOp::name(),
                                 paddle::dialect::AddOp::name(),
                                 paddle::dialect::SqrtOp::name(),
                                 paddle::dialect::DivideOp::name(),
                                 paddle::dialect::ReshapeOp::name(),
                                 paddle::dialect::MultiplyOp::name(),
                                 paddle::dialect::SubtractOp::name(),
                                 paddle::dialect::Conv2dOp::name()});
    auto bn_replace_pattern = std::make_unique<BatchNormReplacePattern>(
        context,
        1,
        std::vector<std::string>{paddle::dialect::BatchNorm_Op::name()});
    auto conv2d_add_act_fuse_pattern =
        std::make_unique<Conv2dAddActFusePattern>(
            context,
            1,
            std::vector<std::string>{paddle::dialect::Conv2dFusionOp::name()});
    auto conv2d_doublue_add_act_fuse_pattern =
        std::make_unique<Conv2dDoubleAddActFusePattern>(
            context,
            1,
            std::vector<std::string>{paddle::dialect::Conv2dFusionOp::name()});

    // bn->bn_replace
    ps.Add(std::move(bn_replace_pattern));
    // conv2d+bn->conv2d
    ps.Add(std::move(conv_bn_pattern));
    // conv2d+add+add+act->conv2d_fusion
    ps.Add(std::move(conv2d_doublue_add_act_fuse_pattern));
    // conv2d+add+act->conv2d_fusion
    ps.Add(std::move(conv2d_add_act_fuse_pattern));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dFusePass() {
  return std::make_unique<Conv2dFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_fuse_pass, Conv2dFusePass);
