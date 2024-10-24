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

#include "paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FusedDotProductAttentionPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedDotProductAttentionPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();

    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &q_transpose = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = q_transpose(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    const auto &k_transpose = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = k_transpose(src.Tensor("k"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &v_transpose = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = v_transpose(src.Tensor("v"));

    const auto &q_scale_full =
        src.Op("pd_op.full", {{"value", src.Attr("q_scale_value")}});
    src.Tensor("q_scale_full_out") = q_scale_full();
    const auto &q_scale = src.Op("pd_op.scale");
    src.Tensor("q_scale_out") =
        q_scale(src.Tensor("q_transpose_out"), src.Tensor("q_scale_full_out"));

    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("qk_matmul_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_transpose_y")}});
    src.Tensor("qk_matmul_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose_out"));

    // mask(int) -> cast -> cast -> scale -> scale -> mask(fp16)
    const auto &mask_cast1 = src.Op("pd_op.cast");
    src.Tensor("mask_cast1_out") = mask_cast1(src.Tensor("mask"));
    const auto &mask_full1 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale1_value")}});
    const auto &mask_scale1 = src.Op("pd_op.scale");
    src.Tensor("mask_scale1_out") =
        mask_scale1(src.Tensor("mask_cast1_out"), mask_full1());
    const auto &mask_full2 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale2_value")}});
    const auto &mask_scale2 = src.Op("pd_op.scale");
    src.Tensor("mask_scale2_out") =
        mask_scale2(src.Tensor("mask_scale1_out"), mask_full2());

    // softmax(qk)v
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_matmul_out"), src.Tensor("mask_scale2_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool qk_matmul_transpose_x =
          match_ctx.Attr<bool>("qk_matmul_transpose_x");
      bool qk_matmul_transpose_y =
          match_ctx.Attr<bool>("qk_matmul_transpose_y");
      if (qk_matmul_transpose_x || !qk_matmul_transpose_y) return false;

      bool context_matmul_transpose_x =
          match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool context_matmul_transpose_y =
          match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (context_matmul_transpose_x || context_matmul_transpose_y)
        return false;

      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("q_scale_value");
        });

    const auto &dot_product_attention =
        res.Op(paddle::dialect::FusedDotProductAttentionOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});

    dot_product_attention({&res.Tensor("q"),
                           &res.Tensor("k"),
                           &res.Tensor("v"),
                           &res.Tensor("mask_scale2_out"),
                           &res.InputNoneTensor(),   // cu_seqlen_q is none
                           &res.InputNoneTensor()},  // cu_seqlen_k is none
                          {&res.Tensor("out"),
                           &res.Tensor("softmax_aux"),
                           &res.Tensor("rng_state")});
  }
};

class FusedDotProductAttentionGradPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedDotProductAttentionGradPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();

    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &q_transpose = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = q_transpose(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    const auto &k_transpose = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = k_transpose(src.Tensor("k"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &v_transpose = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = v_transpose(src.Tensor("v"));

    const auto &q_scale_full =
        src.Op("pd_op.full", {{"value", src.Attr("q_scale_value")}});
    src.Tensor("q_scale_full_out") = q_scale_full();
    const auto &q_scale = src.Op("pd_op.scale");
    src.Tensor("q_scale_out") =
        q_scale(src.Tensor("q_transpose_out"), src.Tensor("q_scale_full_out"));

    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("qk_matmul_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_transpose_y")}});
    src.Tensor("qk_matmul_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose_out"));

    // mask(int) -> cast -> cast -> scale -> scale -> mask(fp16)
    const auto &mask_cast1 = src.Op("pd_op.cast");
    src.Tensor("mask_cast1_out") = mask_cast1(src.Tensor("mask"));
    const auto &mask_full1 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale1_value")}});
    const auto &mask_scale1 = src.Op("pd_op.scale");
    src.Tensor("mask_scale1_out") =
        mask_scale1(src.Tensor("mask_cast1_out"), mask_full1());
    const auto &mask_full2 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale2_value")}});
    const auto &mask_scale2 = src.Op("pd_op.scale");
    src.Tensor("mask_scale2_out") =
        mask_scale2(src.Tensor("mask_scale1_out"), mask_full2());

    // softmax(qk)v
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_matmul_out"), src.Tensor("mask_scale2_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // backward
    const auto &o_transpose_grad = src.Op("pd_op.transpose_grad");
    src.Tensor("o_transpose_grad_out") =
        o_transpose_grad(src.Tensor("out_grad"));
    const auto &context_matmul_grad =
        src.Op("pd_op.matmul_grad",
               {{"transpose_x", src.Attr("context_matmul_grad_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_grad_transpose_y")}});
    context_matmul_grad(
        {&src.Tensor("softmax_out"),
         &src.Tensor("v_transpose_out"),
         &src.Tensor("o_transpose_grad_out")},
        {&src.Tensor("softmax_out_grad"), &src.Tensor("v_transpose_out_grad")});
    const auto &softmax_grad = src.Op("pd_op.softmax_grad");
    softmax_grad({&src.Tensor("softmax_out"), &src.Tensor("softmax_out_grad")},
                 {&src.Tensor("mask_add_out_grad")});
    const auto &v_transpose_grad = src.Op("pd_op.transpose_grad");
    v_transpose_grad({&src.Tensor("v_transpose_out_grad")},
                     {&src.Tensor("v_grad")});
    const auto &mask_add_grad = src.Op("pd_op.add_grad");
    mask_add_grad({&src.Tensor("qk_matmul_out"),
                   &src.Tensor("mask_scale2_out"),
                   &src.Tensor("mask_add_out_grad")},
                  {&src.Tensor("qk_matmul_out_grad"),
                   &src.Tensor("mask_scale2_out_grad")});
    const auto &qk_matmul_grad =
        src.Op("pd_op.matmul_grad",
               {{"transpose_x", src.Attr("qk_matmul_grad_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_grad_transpose_y")}});
    qk_matmul_grad(
        {&src.Tensor("q_scale_out"),
         &src.Tensor("k_transpose_out"),
         &src.Tensor("qk_matmul_out_grad")},
        {&src.Tensor("q_scale_out_grad"), &src.Tensor("k_transpose_out_grad")});
    const auto &q_scale_grad = src.Op("pd_op.scale");
    src.Tensor("q_transpose_out_grad") = q_scale_grad(
        src.Tensor("q_scale_out_grad"), src.Tensor("q_scale_full_out"));
    const auto &q_transpose_grad = src.Op("pd_op.transpose_grad");
    q_transpose_grad({&src.Tensor("q_transpose_out_grad")},
                     {&src.Tensor("q_grad")});
    const auto &k_transpose_grad = src.Op("pd_op.transpose_grad");
    k_transpose_grad({&src.Tensor("k_transpose_out_grad")},
                     {&src.Tensor("k_grad")});

    // Constraints
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool qk_matmul_transpose_x =
          match_ctx.Attr<bool>("qk_matmul_transpose_x");
      bool qk_matmul_transpose_y =
          match_ctx.Attr<bool>("qk_matmul_transpose_y");
      if (qk_matmul_transpose_x || !qk_matmul_transpose_y) return false;

      bool context_matmul_transpose_x =
          match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool context_matmul_transpose_y =
          match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (context_matmul_transpose_x || context_matmul_transpose_y)
        return false;

      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("q_scale_value");
        });

    const auto &dot_product_attention =
        res.Op(paddle::dialect::FusedDotProductAttentionOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});

    dot_product_attention({&res.Tensor("q"),
                           &res.Tensor("k"),
                           &res.Tensor("v"),
                           &res.Tensor("mask_scale2_out"),
                           &res.InputNoneTensor(),   // cu_seqlen_q is none
                           &res.InputNoneTensor()},  // cu_seqlen_k is none
                          {&res.Tensor("out"),
                           &res.Tensor("softmax_aux"),
                           &res.Tensor("rng_state")});
    const auto &dot_product_attention_grad =
        res.Op(paddle::dialect::FusedDotProductAttentionGradOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});
    dot_product_attention_grad(
        {&res.Tensor("q"),
         &res.Tensor("k"),
         &res.Tensor("v"),
         &res.Tensor("mask_scale2_out"),
         &res.InputNoneTensor(),  // cu_seqlen_q is none
         &res.InputNoneTensor(),  // cu_seqlen_k is none
         &res.Tensor("out"),
         &res.Tensor("softmax_aux"),
         &res.Tensor("rng_state"),
         &res.Tensor("out_grad")},
        {&res.Tensor("q_grad"), &res.Tensor("k_grad"), &res.Tensor("v_grad")});
  }
};

class FusedDotProductAttentionWithDropoutPattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedDotProductAttentionWithDropoutPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();

    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &q_transpose = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = q_transpose(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    const auto &k_transpose = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = k_transpose(src.Tensor("k"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &v_transpose = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = v_transpose(src.Tensor("v"));

    const auto &q_scale_full =
        src.Op("pd_op.full", {{"value", src.Attr("q_scale_value")}});
    src.Tensor("q_scale_full_out") = q_scale_full();
    const auto &q_scale = src.Op("pd_op.scale");
    src.Tensor("q_scale_out") =
        q_scale(src.Tensor("q_transpose_out"), src.Tensor("q_scale_full_out"));

    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("qk_matmul_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_transpose_y")}});
    src.Tensor("qk_matmul_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose_out"));

    // mask(int) -> cast -> cast -> scale -> scale -> mask(fp16)
    const auto &mask_cast1 = src.Op("pd_op.cast");
    src.Tensor("mask_cast1_out") = mask_cast1(src.Tensor("mask"));
    const auto &mask_full1 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale1_value")}});
    const auto &mask_scale1 = src.Op("pd_op.scale");
    src.Tensor("mask_scale1_out") =
        mask_scale1(src.Tensor("mask_cast1_out"), mask_full1());
    const auto &mask_full2 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale2_value")}});
    const auto &mask_scale2 = src.Op("pd_op.scale");
    src.Tensor("mask_scale2_out") =
        mask_scale2(src.Tensor("mask_scale1_out"), mask_full2());

    // softmax(qk)v
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_matmul_out"), src.Tensor("mask_scale2_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    const auto &dropout = src.Op("pd_op.dropout",
                                 {{"p", src.Attr("dropout_prob")},
                                  {"is_test", src.Attr("is_test")},
                                  {"mode", src.Attr("mode")},
                                  {"seed", src.Attr("seed")},
                                  {"fix_seed", src.Attr("fix_seed")}});
    dropout({&src.Tensor("softmax_out"), &src.Tensor("seed_tensor")},
            {&src.Tensor("dropout_out"), &src.Tensor("dropout_mask")});
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("dropout_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool qk_matmul_transpose_x =
          match_ctx.Attr<bool>("qk_matmul_transpose_x");
      bool qk_matmul_transpose_y =
          match_ctx.Attr<bool>("qk_matmul_transpose_y");
      if (qk_matmul_transpose_x || !qk_matmul_transpose_y) return false;

      bool context_matmul_transpose_x =
          match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool context_matmul_transpose_y =
          match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (context_matmul_transpose_x || context_matmul_transpose_y)
        return false;

      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("q_scale_value");
        });

    const auto &dot_product_attention =
        res.Op(paddle::dialect::FusedDotProductAttentionOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", src.Attr("dropout_prob")},
                 {"is_training", res.BoolAttr(true)},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});

    dot_product_attention({&res.Tensor("q"),
                           &res.Tensor("k"),
                           &res.Tensor("v"),
                           &res.Tensor("mask_scale2_out"),
                           &res.InputNoneTensor(),   // cu_seqlen_q is none
                           &res.InputNoneTensor()},  // cu_seqlen_k is none
                          {&res.Tensor("out"),
                           &res.Tensor("softmax_aux"),
                           &res.Tensor("rng_state")});
  }
};

class FusedDotProductAttentionGradWithDropoutPattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedDotProductAttentionGradWithDropoutPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();

    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &q_transpose = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = q_transpose(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    const auto &k_transpose = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = k_transpose(src.Tensor("k"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &v_transpose = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = v_transpose(src.Tensor("v"));

    const auto &q_scale_full =
        src.Op("pd_op.full", {{"value", src.Attr("q_scale_value")}});
    src.Tensor("q_scale_full_out") = q_scale_full();
    const auto &q_scale = src.Op("pd_op.scale");
    src.Tensor("q_scale_out") =
        q_scale(src.Tensor("q_transpose_out"), src.Tensor("q_scale_full_out"));

    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("qk_matmul_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_transpose_y")}});
    src.Tensor("qk_matmul_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose_out"));

    // mask(int) -> cast -> cast -> scale -> scale -> mask(fp16)
    const auto &mask_cast1 = src.Op("pd_op.cast");
    src.Tensor("mask_cast1_out") = mask_cast1(src.Tensor("mask"));
    const auto &mask_full1 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale1_value")}});
    const auto &mask_scale1 = src.Op("pd_op.scale");
    src.Tensor("mask_scale1_out") =
        mask_scale1(src.Tensor("mask_cast1_out"), mask_full1());
    const auto &mask_full2 =
        src.Op("pd_op.full", {{"value", src.Attr("mask_scale2_value")}});
    const auto &mask_scale2 = src.Op("pd_op.scale");
    src.Tensor("mask_scale2_out") =
        mask_scale2(src.Tensor("mask_scale1_out"), mask_full2());

    // softmax(qk)v
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_matmul_out"), src.Tensor("mask_scale2_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    const auto &dropout = src.Op("pd_op.dropout",
                                 {{"p", src.Attr("dropout_prob")},
                                  {"is_test", src.Attr("is_test")},
                                  {"mode", src.Attr("mode")},
                                  {"seed", src.Attr("seed")},
                                  {"fix_seed", src.Attr("fix_seed")}});
    dropout({&src.Tensor("softmax_out"), &src.Tensor("seed_tensor")},
            {&src.Tensor("dropout_out"), &src.Tensor("dropout_mask")});
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("dropout_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // backward
    const auto &o_transpose_grad = src.Op("pd_op.transpose_grad");
    src.Tensor("o_transpose_grad_out") =
        o_transpose_grad(src.Tensor("out_grad"));
    const auto &context_matmul_grad =
        src.Op("pd_op.matmul_grad",
               {{"transpose_x", src.Attr("context_matmul_grad_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_grad_transpose_y")}});
    context_matmul_grad(
        {&src.Tensor("dropout_out"),
         &src.Tensor("v_transpose_out"),
         &src.Tensor("o_transpose_grad_out")},
        {&src.Tensor("dropout_out_grad"), &src.Tensor("v_transpose_out_grad")});
    const auto &dropout_grad = src.Op("pd_op.dropout_grad",
                                      {{"p", src.Attr("dropout_prob")},
                                       {"is_test", src.Attr("is_test")},
                                       {"mode", src.Attr("mode")}});
    dropout_grad({&src.Tensor("dropout_mask"), &src.Tensor("dropout_out_grad")},
                 {&src.Tensor("softmax_out_grad")});
    const auto &softmax_grad = src.Op("pd_op.softmax_grad");
    softmax_grad({&src.Tensor("softmax_out"), &src.Tensor("softmax_out_grad")},
                 {&src.Tensor("mask_add_out_grad")});
    const auto &v_transpose_grad = src.Op("pd_op.transpose_grad");
    v_transpose_grad({&src.Tensor("v_transpose_out_grad")},
                     {&src.Tensor("v_grad")});
    const auto &mask_add_grad = src.Op("pd_op.add_grad");
    mask_add_grad({&src.Tensor("qk_matmul_out"),
                   &src.Tensor("mask_scale2_out"),
                   &src.Tensor("mask_add_out_grad")},
                  {&src.Tensor("qk_matmul_out_grad"),
                   &src.Tensor("mask_scale2_out_grad")});
    const auto &qk_matmul_grad =
        src.Op("pd_op.matmul_grad",
               {{"transpose_x", src.Attr("qk_matmul_grad_transpose_x")},
                {"transpose_y", src.Attr("qk_matmul_grad_transpose_y")}});
    qk_matmul_grad(
        {&src.Tensor("q_scale_out"),
         &src.Tensor("k_transpose_out"),
         &src.Tensor("qk_matmul_out_grad")},
        {&src.Tensor("q_scale_out_grad"), &src.Tensor("k_transpose_out_grad")});
    const auto &q_scale_grad = src.Op("pd_op.scale");
    src.Tensor("q_transpose_out_grad") = q_scale_grad(
        src.Tensor("q_scale_out_grad"), src.Tensor("q_scale_full_out"));
    const auto &q_transpose_grad = src.Op("pd_op.transpose_grad");
    q_transpose_grad({&src.Tensor("q_transpose_out_grad")},
                     {&src.Tensor("q_grad")});
    const auto &k_transpose_grad = src.Op("pd_op.transpose_grad");
    k_transpose_grad({&src.Tensor("k_transpose_out_grad")},
                     {&src.Tensor("k_grad")});

    // Constraints
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool qk_matmul_transpose_x =
          match_ctx.Attr<bool>("qk_matmul_transpose_x");
      bool qk_matmul_transpose_y =
          match_ctx.Attr<bool>("qk_matmul_transpose_y");
      if (qk_matmul_transpose_x || !qk_matmul_transpose_y) return false;

      bool context_matmul_transpose_x =
          match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool context_matmul_transpose_y =
          match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (context_matmul_transpose_x || context_matmul_transpose_y)
        return false;

      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("q_scale_value");
        });

    const auto &dot_product_attention =
        res.Op(paddle::dialect::FusedDotProductAttentionOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", src.Attr("dropout_prob")},
                 {"is_training", res.BoolAttr(true)},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});

    dot_product_attention({&res.Tensor("q"),
                           &res.Tensor("k"),
                           &res.Tensor("v"),
                           &res.Tensor("mask_scale2_out"),
                           &res.InputNoneTensor(),   // cu_seqlen_q is none
                           &res.InputNoneTensor()},  // cu_seqlen_k is none
                          {&res.Tensor("out"),
                           &res.Tensor("softmax_aux"),
                           &res.Tensor("rng_state")});
    const auto &dot_product_attention_grad =
        res.Op(paddle::dialect::FusedDotProductAttentionGradOp::name(),
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", src.Attr("dropout_prob")},
                 {"mask_type_str", res.StrAttr("none")},
                 {"bias_type_str", res.StrAttr("post_scale_bias")}}});
    dot_product_attention_grad(
        {&res.Tensor("q"),
         &res.Tensor("k"),
         &res.Tensor("v"),
         &res.Tensor("mask_scale2_out"),
         &res.InputNoneTensor(),  // cu_seqlen_q is none
         &res.InputNoneTensor(),  // cu_seqlen_k is none
         &res.Tensor("out"),
         &res.Tensor("softmax_aux"),
         &res.Tensor("rng_state"),
         &res.Tensor("out_grad")},
        {&res.Tensor("q_grad"), &res.Tensor("k_grad"), &res.Tensor("v_grad")});
  }
};

class FusedDotProductAttentionPass : public pir::PatternRewritePass {
 public:
  FusedDotProductAttentionPass()
      : pir::PatternRewritePass("fused_dot_product_attention_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedDotProductAttentionPattern>(context));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionGradPattern>(context));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionWithDropoutPattern>(
        context));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionGradWithDropoutPattern>(
        context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedDotProductAttentionPass() {
  return std::make_unique<FusedDotProductAttentionPass>();
}
}  // namespace pir

REGISTER_IR_PASS(fused_dot_product_attention_pass,
                 FusedDotProductAttentionPass);
