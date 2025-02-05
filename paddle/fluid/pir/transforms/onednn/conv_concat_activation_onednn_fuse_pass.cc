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

#include "paddle/fluid/pir/transforms/onednn/conv_concat_activation_onednn_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class NConvConcatActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t concat_count_;
  std::string activation_name_;
  /*
   * fused_level_ = 0 : conv2d + activation
    fused_level_ = 1 : conv2d + bias + activation
                       conv2d + residual + activation
                       conv2d + bias + residual + activation
  */
  const int fused_level_;
  const int benefit_;

 public:
  NConvConcatActivationFusePattern(size_t concat_count,
                                   std::string activation_name,
                                   int fused_level,
                                   int benefit)
      : concat_count_(concat_count),
        activation_name_(std::move(activation_name)),
        fused_level_(fused_level),
        benefit_(benefit) {}

  std::string name() const override {
    return "Conv" + std::to_string(concat_count_) + "Concat" + "Level" +
           std::to_string(fused_level_) + activation_name_ + "Pattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }
    std::vector<const paddle::drr::Tensor *> combine_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &conv =
          fused_level_ == 0
              ? pat.Op(
                    conv_name,
                    {{"strides", pat.Attr("strides" + std::to_string(i))},
                     {"paddings", pat.Attr("paddings" + std::to_string(i))},
                     {"padding_algorithm",
                      pat.Attr("padding_algorithm" + std::to_string(i))},
                     {"dilations", pat.Attr("dilations" + std::to_string(i))},
                     {"groups", pat.Attr("groups" + std::to_string(i))},
                     {"data_format",
                      pat.Attr("data_format" + std::to_string(i))}})
              : pat.Op(
                    conv_name,
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation",
                         pat.Attr("fuse_activation" + std::to_string(i))},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha",
                         pat.Attr("fuse_alpha" + std::to_string(i))},
                        {"fuse_beta",
                         pat.Attr("fuse_beta" + std::to_string(i))},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i)),
              &pat.Tensor("__@bias" + std::to_string(i) + "@__"),
              &pat.Tensor("__@residual" + std::to_string(i) + "@__")},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});

      } else {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i))},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});
      }

      combine_in.push_back(&pat.Tensor("conv2d_out_" + std::to_string(i)));
    }
    const auto &combine_op = pat.Op(pir::CombineOp::name());
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});

    combine_op(combine_in, {&pat.Tensor("combine_out")});
    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    concat_op({&pat.Tensor("combine_out"), &full_op()},
              {&pat.Tensor("concat_out")});

    std::string activation_name_op = "pd_op." + activation_name_;
    if (activation_name_ == "hard_swish") {
      // oneDNN use hard_swish, paddle use hardswish
      activation_name_op = "pd_op.hardswish";
    }
    const auto &activation =
        activation_name_op != "pd_op.leaky_relu"
            ? pat.Op(activation_name_op)
            : pat.Op(activation_name_op,
                     {{"negative_slope", pat.Attr("negative_slope")}});
    pat.Tensor("activation_out") = activation(pat.Tensor("concat_out"));

    if (fused_level_ > 0) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto act_type = match_ctx.Attr<std::string>("fuse_activation");
        if (act_type != "") {
          return false;
        }
        return true;
      });
    }
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (activation_name_ == "leaky_relu") {
        float negative_slope = match_ctx.Attr<float>("negative_slope");
        // leaky relu alpha is a positive number
        if (negative_slope <= 0.0) {
          return false;
        }
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_beta = res.Float32Attr(0.0f);
    auto fuse_alpha = res.Float32Attr(0.0f);
    if (activation_name_ == "relu6") {
      fuse_beta = res.Float32Attr(6.0f);
    } else if (activation_name_ == "hard_swish") {
      // hard swish have not attr float threshold = 6.0f, float scale = 6.0f,
      // float offset = 3.0f attr But in previous implementation hard swish,
      // fuse_alpha=1.f / 6.f， fuse_beta=1.f / 2.f, it has fixed
      fuse_beta = res.Float32Attr(1.f / 2.f);
      fuse_alpha = res.Float32Attr(1.f / 6.f);
    } else if (activation_name_ == "swish") {
      fuse_alpha = res.Float32Attr(1.0f);
    } else if (activation_name_ == "leaky_relu") {
      fuse_alpha = pat.Attr("negative_slope");
    }

    std::vector<const paddle::drr::Tensor *> combine_result_in;
    // int input_num = 1;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &fused_conv =
          fused_level_ == 0
              ? res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type", res.StrAttr("float32")},
                        {"fuse_activation", res.StrAttr(activation_name_)},
                        {"fuse_residual_connection", res.BoolAttr(false)},
                        {"force_fp32_output", res.BoolAttr(false)},
                        {"fuse_alpha", fuse_alpha},
                        {"fuse_beta", fuse_beta},
                        {"scale_in", res.Float32Attr(1.0f)},
                        {"scale_out", res.Float32Attr(1.0f)},
                        {"scale_in_eltwise", res.Float32Attr(1.0f)},
                        {"scale_weights", res.VectorFloatAttr({1.0f})},
                    }})
              : res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation", res.StrAttr(activation_name_)},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha", fuse_alpha},
                        {"fuse_beta", fuse_beta},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.Tensor("__@bias" + std::to_string(i) + "@__"),
                    &res.Tensor("__@residual" + std::to_string(i) + "@__")},
                   {&res.Tensor("act_out_" + std::to_string(i))});

      } else {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.InputNoneTensor(),
                    &res.InputNoneTensor()},
                   {&res.Tensor("act_out_" + std::to_string(i))});
      }
      combine_result_in.push_back(&res.Tensor("act_out_" + std::to_string(i)));
    }

    const auto &combine = res.Op(pir::CombineOp::name());

    combine(combine_result_in, {&res.Tensor("combine_result_out")});

    // const auto &concat_result_op =
    // res.Op(paddle::dialect::ConcatOp::name(),{{"axis", res.Int32Attr(0) }});
    const auto &full_result_op = res.Op(paddle::dialect::FullOp::name(),
                                        {{"shape", pat.Attr("shape")},
                                         {"value", pat.Attr("value")},
                                         {"dtype", pat.Attr("dtype")},
                                         {"place", pat.Attr("place")}});

    const auto &concat_result_op = res.Op(paddle::dialect::ConcatOp::name());
    concat_result_op({&res.Tensor("combine_result_out"), &full_result_op()},
                     {&res.Tensor("activation_out")});

    // concat_result_op(combine_result_in, {&res.Tensor("concat_out")});
  }
};

class NConvConcatHardSigmoidFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t concat_count_;
  std::string activation_name_;
  /*
   * fused_level_ = 0 : conv2d + activation
    fused_level_ = 1 : conv2d + bias + activation
                       conv2d + residual + activation
                       conv2d + bias + residual + activation
  */
  const int fused_level_;

 public:
  NConvConcatHardSigmoidFusePattern(size_t concat_count,
                                    std::string activation_name,
                                    int fused_level)
      : concat_count_(concat_count),
        activation_name_(std::move(activation_name)),
        fused_level_(fused_level) {}

  std::string name() const override {
    return "Conv" + std::to_string(concat_count_) + "Concat" + "Level" +
           std::to_string(fused_level_) + "HardSigmoidPattern";
  }

  uint32_t benefit() const override { return concat_count_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }
    std::vector<const paddle::drr::Tensor *> combine_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &conv =
          fused_level_ == 0
              ? pat.Op(
                    conv_name,
                    {{"strides", pat.Attr("strides" + std::to_string(i))},
                     {"paddings", pat.Attr("paddings" + std::to_string(i))},
                     {"padding_algorithm",
                      pat.Attr("padding_algorithm" + std::to_string(i))},
                     {"dilations", pat.Attr("dilations" + std::to_string(i))},
                     {"groups", pat.Attr("groups" + std::to_string(i))},
                     {"data_format",
                      pat.Attr("data_format" + std::to_string(i))}})
              : pat.Op(
                    conv_name,
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation",
                         pat.Attr("fuse_activation" + std::to_string(i))},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha",
                         pat.Attr("fuse_alpha" + std::to_string(i))},
                        {"fuse_beta",
                         pat.Attr("fuse_beta" + std::to_string(i))},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i)),
              &pat.Tensor("__@bias" + std::to_string(i) + "@__"),
              &pat.Tensor("__@residual" + std::to_string(i) + "@__")},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});

      } else {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i))},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});
      }

      combine_in.push_back(&pat.Tensor("conv2d_out_" + std::to_string(i)));
    }
    const auto &combine_op = pat.Op(pir::CombineOp::name());
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});

    combine_op(combine_in, {&pat.Tensor("combine_out")});
    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    concat_op({&pat.Tensor("combine_out"), &full_op()},
              {&pat.Tensor("concat_out")});

    const auto &activation =
        pat.Op(activation_name_,
               {{"slope", pat.Attr("slope")}, {"offset", pat.Attr("offset")}});
    pat.Tensor("activation_out") = activation(pat.Tensor("concat_out"));

    if (fused_level_ > 0) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto act_type = match_ctx.Attr<std::string>("fuse_activation");
        if (act_type != "") {
          return false;
        }
        return true;
      });
    }
    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::vector<const paddle::drr::Tensor *> combine_result_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &fused_conv =
          fused_level_ == 0
              ? res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type", res.StrAttr("float32")},
                        {"fuse_activation", res.StrAttr("hard_sigmoid")},
                        {"fuse_residual_connection", res.BoolAttr(false)},
                        {"force_fp32_output", res.BoolAttr(false)},
                        {"fuse_alpha", pat.Attr("slope")},
                        {"fuse_beta", pat.Attr("offset")},
                        {"scale_in", res.Float32Attr(1.0f)},
                        {"scale_out", res.Float32Attr(1.0f)},
                        {"scale_in_eltwise", res.Float32Attr(1.0f)},
                        {"scale_weights", res.VectorFloatAttr({1.0f})},
                    }})
              : res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation", res.StrAttr("hard_sigmoid")},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha", pat.Attr("slope")},
                        {"fuse_beta", pat.Attr("offset")},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.Tensor("__@bias" + std::to_string(i) + "@__"),
                    &res.Tensor("__@residual" + std::to_string(i) + "@__")},
                   {&res.Tensor("act_out_" + std::to_string(i))});

      } else {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.InputNoneTensor(),
                    &res.InputNoneTensor()},
                   {&res.Tensor("act_out_" + std::to_string(i))});
      }
      combine_result_in.push_back(&res.Tensor("act_out_" + std::to_string(i)));
    }

    const auto &combine = res.Op(pir::CombineOp::name());

    combine(combine_result_in, {&res.Tensor("combine_result_out")});

    // const auto &concat_result_op =
    // res.Op(paddle::dialect::ConcatOp::name(),{{"axis", res.Int32Attr(0) }});
    const auto &full_result_op = res.Op(paddle::dialect::FullOp::name(),
                                        {{"shape", pat.Attr("shape")},
                                         {"value", pat.Attr("value")},
                                         {"dtype", pat.Attr("dtype")},
                                         {"place", pat.Attr("place")}});

    const auto &concat_result_op = res.Op(paddle::dialect::ConcatOp::name());
    concat_result_op({&res.Tensor("combine_result_out"), &full_result_op()},
                     {&res.Tensor("activation_out")});

    // concat_result_op(combine_result_in, {&res.Tensor("concat_out")});
  }
};

class NConvConcatGeluFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t concat_count_;
  std::string activation_name_;
  /*
   * fused_level_ = 0 : conv2d + activation
    fused_level_ = 1 : conv2d + bias + activation
                       conv2d + residual + activation
                       conv2d + bias + residual + activation
  */
  const int fused_level_;

 public:
  NConvConcatGeluFusePattern(size_t concat_count,
                             std::string activation_name,
                             int fused_level)
      : concat_count_(concat_count),
        activation_name_(std::move(activation_name)),
        fused_level_(fused_level) {}

  std::string name() const override {
    return "Conv" + std::to_string(concat_count_) + "Concat" + "Level" +
           std::to_string(fused_level_) + "GeluPattern";
  }

  uint32_t benefit() const override { return concat_count_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }
    std::vector<const paddle::drr::Tensor *> combine_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &conv =
          fused_level_ == 0
              ? pat.Op(
                    conv_name,
                    {{"strides", pat.Attr("strides" + std::to_string(i))},
                     {"paddings", pat.Attr("paddings" + std::to_string(i))},
                     {"padding_algorithm",
                      pat.Attr("padding_algorithm" + std::to_string(i))},
                     {"dilations", pat.Attr("dilations" + std::to_string(i))},
                     {"groups", pat.Attr("groups" + std::to_string(i))},
                     {"data_format",
                      pat.Attr("data_format" + std::to_string(i))}})
              : pat.Op(
                    conv_name,
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation",
                         pat.Attr("fuse_activation" + std::to_string(i))},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha",
                         pat.Attr("fuse_alpha" + std::to_string(i))},
                        {"fuse_beta",
                         pat.Attr("fuse_beta" + std::to_string(i))},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i)),
              &pat.Tensor("__@bias" + std::to_string(i) + "@__"),
              &pat.Tensor("__@residual" + std::to_string(i) + "@__")},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});

      } else {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i))},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});
      }

      combine_in.push_back(&pat.Tensor("conv2d_out_" + std::to_string(i)));
    }
    const auto &combine_op = pat.Op(pir::CombineOp::name());
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});

    combine_op(combine_in, {&pat.Tensor("combine_out")});
    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    concat_op({&pat.Tensor("combine_out"), &full_op()},
              {&pat.Tensor("concat_out")});

    const auto &activation =
        pat.Op(activation_name_, {{"approximate", pat.Attr("approximate")}});
    pat.Tensor("activation_out") = activation(pat.Tensor("concat_out"));

    if (fused_level_ > 0) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto act_type = match_ctx.Attr<std::string>("fuse_activation");
        if (act_type != "") {
          return false;
        }
        return true;
      });
    }
    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::vector<const paddle::drr::Tensor *> combine_result_in;
    const auto &gelu = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::string {
          bool approximate = match_ctx.Attr<bool>("approximate");
          if (approximate) return "gelu_tanh";
          return "gelu_erf";
        });

    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &fused_conv =
          fused_level_ == 0
              ? res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type", res.StrAttr("float32")},
                        {"fuse_activation", gelu},
                        {"fuse_residual_connection", res.BoolAttr(false)},
                        {"force_fp32_output", res.BoolAttr(false)},
                        {"fuse_alpha", res.Float32Attr(0.0f)},
                        {"fuse_beta", res.Float32Attr(0.0f)},
                        {"scale_in", res.Float32Attr(1.0f)},
                        {"scale_out", res.Float32Attr(1.0f)},
                        {"scale_in_eltwise", res.Float32Attr(1.0f)},
                        {"scale_weights", res.VectorFloatAttr({1.0f})},
                    }})
              : res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation", gelu},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha", res.Float32Attr(0.0f)},
                        {"fuse_beta", res.Float32Attr(0.0f)},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.Tensor("__@bias" + std::to_string(i) + "@__"),
                    &res.Tensor("__@residual" + std::to_string(i) + "@__")},
                   {&res.Tensor("act_out_" + std::to_string(i))});

      } else {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.InputNoneTensor(),
                    &res.InputNoneTensor()},
                   {&res.Tensor("act_out_" + std::to_string(i))});
      }
      combine_result_in.push_back(&res.Tensor("act_out_" + std::to_string(i)));
    }

    const auto &combine = res.Op(pir::CombineOp::name());

    combine(combine_result_in, {&res.Tensor("combine_result_out")});

    // const auto &concat_result_op =
    // res.Op(paddle::dialect::ConcatOp::name(),{{"axis", res.Int32Attr(0) }});
    const auto &full_result_op = res.Op(paddle::dialect::FullOp::name(),
                                        {{"shape", pat.Attr("shape")},
                                         {"value", pat.Attr("value")},
                                         {"dtype", pat.Attr("dtype")},
                                         {"place", pat.Attr("place")}});

    const auto &concat_result_op = res.Op(paddle::dialect::ConcatOp::name());
    concat_result_op({&res.Tensor("combine_result_out"), &full_result_op()},
                     {&res.Tensor("activation_out")});

    // concat_result_op(combine_result_in, {&res.Tensor("concat_out")});
  }
};

class NConvConcatClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t concat_count_;
  std::string activation_name_;
  /*
   * fused_level_ = 0 : conv2d + activation
    fused_level_ = 1 : conv2d + bias + activation
                       conv2d + residual + activation
                       conv2d + bias + residual + activation
  */
  const int fused_level_;

 public:
  NConvConcatClipFusePattern(size_t concat_count,
                             std::string activation_name,
                             int fused_level)
      : concat_count_(concat_count),
        activation_name_(std::move(activation_name)),
        fused_level_(fused_level) {}

  std::string name() const override {
    return "Conv" + std::to_string(concat_count_) + "Concat" + "Level" +
           std::to_string(fused_level_) + "ClipPattern";
  }

  uint32_t benefit() const override { return concat_count_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }

    std::vector<const paddle::drr::Tensor *> combine_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &conv =
          fused_level_ == 0
              ? pat.Op(
                    conv_name,
                    {{"strides", pat.Attr("strides" + std::to_string(i))},
                     {"paddings", pat.Attr("paddings" + std::to_string(i))},
                     {"padding_algorithm",
                      pat.Attr("padding_algorithm" + std::to_string(i))},
                     {"dilations", pat.Attr("dilations" + std::to_string(i))},
                     {"groups", pat.Attr("groups" + std::to_string(i))},
                     {"data_format",
                      pat.Attr("data_format" + std::to_string(i))}})
              : pat.Op(
                    conv_name,
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation",
                         pat.Attr("fuse_activation" + std::to_string(i))},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha",
                         pat.Attr("fuse_alpha" + std::to_string(i))},
                        {"fuse_beta",
                         pat.Attr("fuse_beta" + std::to_string(i))},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i)),
              &pat.Tensor("__@bias" + std::to_string(i) + "@__"),
              &pat.Tensor("__@residual" + std::to_string(i) + "@__")},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});

      } else {
        conv({&pat.Tensor("input" + std::to_string(i)),
              &pat.Tensor("filter" + std::to_string(i))},
             {&pat.Tensor("conv2d_out_" + std::to_string(i))});
      }

      combine_in.push_back(&pat.Tensor("conv2d_out_" + std::to_string(i)));
    }
    const auto &combine_op = pat.Op(pir::CombineOp::name());
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});

    combine_op(combine_in, {&pat.Tensor("combine_out")});
    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    concat_op({&pat.Tensor("combine_out"), &full_op()},
              {&pat.Tensor("concat_out")});

    const auto &full_1 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &full_2 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_2_value")}});
    pat.Tensor("min") = full_1();
    pat.Tensor("max") = full_2();

    const auto &activation = pat.Op(activation_name_);

    pat.Tensor("activation_out") = activation(
        pat.Tensor("concat_out"), pat.Tensor("min"), pat.Tensor("max"));

    if (fused_level_ > 0) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto act_type = match_ctx.Attr<std::string>("fuse_activation");
        if (act_type != "") {
          return false;
        }
        return true;
      });
    }
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fuse_alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_1_value");
        });
    const auto &fuse_beta = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_2_value");
        });

    std::vector<const paddle::drr::Tensor *> combine_result_in;
    for (size_t i = 1; i <= concat_count_; i++) {
      const auto &fused_conv =
          fused_level_ == 0
              ? res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type", res.StrAttr("float32")},
                        {"fuse_activation", res.StrAttr("clip")},
                        {"fuse_residual_connection", res.BoolAttr(false)},
                        {"force_fp32_output", res.BoolAttr(false)},
                        {"fuse_alpha", fuse_alpha},
                        {"fuse_beta", fuse_beta},
                        {"scale_in", res.Float32Attr(1.0f)},
                        {"scale_out", res.Float32Attr(1.0f)},
                        {"scale_in_eltwise", res.Float32Attr(1.0f)},
                        {"scale_weights", res.VectorFloatAttr({1.0f})},
                    }})
              : res.Op(
                    paddle::onednn::dialect::FusedConv2dOp::name(),
                    {{
                        {"strides", pat.Attr("strides" + std::to_string(i))},
                        {"paddings", pat.Attr("paddings" + std::to_string(i))},
                        {"padding_algorithm",
                         pat.Attr("padding_algorithm" + std::to_string(i))},
                        {"dilations",
                         pat.Attr("dilations" + std::to_string(i))},
                        {"groups", pat.Attr("groups" + std::to_string(i))},
                        {"data_format",
                         pat.Attr("data_format" + std::to_string(i))},
                        {"mkldnn_data_type",
                         pat.Attr("mkldnn_data_type" + std::to_string(i))},
                        {"fuse_activation", res.StrAttr("clip")},
                        {"fuse_residual_connection",
                         pat.Attr("fuse_residual_connection" +
                                  std::to_string(i))},
                        {"force_fp32_output",
                         pat.Attr("force_fp32_output" + std::to_string(i))},
                        {"fuse_alpha", fuse_alpha},
                        {"fuse_beta", fuse_beta},
                        {"scale_in", pat.Attr("scale_in" + std::to_string(i))},
                        {"scale_out",
                         pat.Attr("scale_out" + std::to_string(i))},
                        {"scale_in_eltwise",
                         pat.Attr("scale_in_eltwise" + std::to_string(i))},
                        {"scale_weights",
                         pat.Attr("scale_weights" + std::to_string(i))},
                    }});

      if (fused_level_ > 0) {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.Tensor("__@bias" + std::to_string(i) + "@__"),
                    &res.Tensor("__@residual" + std::to_string(i) + "@__")},
                   {&res.Tensor("act_out_" + std::to_string(i))});

      } else {
        fused_conv({&res.Tensor("input" + std::to_string(i)),
                    &res.Tensor("filter" + std::to_string(i)),
                    &res.InputNoneTensor(),
                    &res.InputNoneTensor()},
                   {&res.Tensor("act_out_" + std::to_string(i))});
      }
      combine_result_in.push_back(&res.Tensor("act_out_" + std::to_string(i)));
    }

    const auto &combine = res.Op(pir::CombineOp::name());

    combine(combine_result_in, {&res.Tensor("combine_result_out")});

    // const auto &concat_result_op =
    // res.Op(paddle::dialect::ConcatOp::name(),{{"axis", res.Int32Attr(0) }});
    const auto &full_result_op = res.Op(paddle::dialect::FullOp::name(),
                                        {{"shape", pat.Attr("shape")},
                                         {"value", pat.Attr("value")},
                                         {"dtype", pat.Attr("dtype")},
                                         {"place", pat.Attr("place")}});

    const auto &concat_result_op = res.Op(paddle::dialect::ConcatOp::name());
    concat_result_op({&res.Tensor("combine_result_out"), &full_result_op()},
                     {&res.Tensor("activation_out")});

    // concat_result_op(combine_result_in, {&res.Tensor("concat_out")});
  }
};

class ConvConcatActFusePass : public pir::PatternRewritePass {
 public:
  ConvConcatActFusePass()
      : pir::PatternRewritePass("conv_concat_activation_mkldnn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    std::vector<std::string> supported_activations_name = {"abs",
                                                           "sqrt",
                                                           "mish",
                                                           "relu",
                                                           "sigmoid",
                                                           "tanh",
                                                           "relu6",
                                                           "hard_swish",
                                                           "swish",
                                                           "leaky_relu"};
    int benefit = 1;
    /**
     * To avoid many for loop patterns to reduce efficiency
     * We just support 6 conv2d concat now
     * And concat in OneDNN with a large number of concat ops
     * performance is worse than CPU kernel.
     */
    /**
     * fused_level 0:  conv2d + activation
     *             1:  fused_conv2d + activation
     */
    for (size_t concat_num = 1; concat_num <= 6; concat_num++) {
      for (auto activation : supported_activations_name) {
        ps.Add(paddle::drr::Create<NConvConcatActivationFusePattern>(
            context, concat_num, activation, 0, benefit++));
        ps.Add(paddle::drr::Create<NConvConcatActivationFusePattern>(
            context, concat_num, activation, 1, benefit++));
      }
    }

    /**
     * These activation use separate pattern to avoid to too large of benefit
     */
    for (size_t concat_num = 1; concat_num <= 6; concat_num++) {
      ps.Add(paddle::drr::Create<NConvConcatHardSigmoidFusePattern>(
          context, concat_num, paddle::dialect::HardsigmoidOp::name(), 0));
      ps.Add(paddle::drr::Create<NConvConcatHardSigmoidFusePattern>(
          context, concat_num, paddle::dialect::HardsigmoidOp::name(), 1));
    }

    for (size_t concat_num = 1; concat_num <= 6; concat_num++) {
      ps.Add(paddle::drr::Create<NConvConcatGeluFusePattern>(
          context, concat_num, paddle::dialect::GeluOp::name(), 0));
      ps.Add(paddle::drr::Create<NConvConcatGeluFusePattern>(
          context, concat_num, paddle::dialect::GeluOp::name(), 1));
    }

    for (size_t concat_num = 1; concat_num <= 6; concat_num++) {
      ps.Add(paddle::drr::Create<NConvConcatClipFusePattern>(
          context, concat_num, paddle::dialect::ClipOp::name(), 0));
      ps.Add(paddle::drr::Create<NConvConcatClipFusePattern>(
          context, concat_num, paddle::dialect::ClipOp::name(), 1));
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dConcatActFusePass() {
  // /**
  //  * This pass must execution before conv_activation_mkldnn_fuse_pass
  //  *  conv   conv   conv      conv     conv    conv        fused_conv
  //  fused_conv  fused_conv
  //  *     \   /  ...             |        |   ...              \           /
  //  .....
  //  *     concat      ->        act      act          ->           concat
  //  *       |                      \       /
  //  *      act                      concat
  // */
  return std::make_unique<ConvConcatActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv_concat_activation_onednn_fuse_pass,
                 ConvConcatActFusePass);
