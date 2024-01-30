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

#include "paddle/fluid/pir/transforms/fusion/embedding_eltwise_layernorm_fuse_pass.h"

#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class Fused2EmbeddingEltwiseLayernormPattern
    : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &embedding_1 = pat.Op(paddle::dialect::EmbeddingOp::name(),
                                     {{{"padding_idx", pat.Attr("padding_idx")},
                                       {"sparse", pat.Attr("sparse")}}});
    const auto &embedding_2 = pat.Op(paddle::dialect::EmbeddingOp::name(),
                                     {{{"padding_idx", pat.Attr("padding_idx")},
                                       {"sparse", pat.Attr("sparse")}}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    embedding_1({&pat.Tensor("x1"), &pat.Tensor("w1")},
                {&pat.Tensor("embedding_1_out")});
    embedding_2({&pat.Tensor("x2"), &pat.Tensor("w2")},
                {&pat.Tensor("embedding_2_out")});
    pat.Tensor("add_out") =
        add(pat.Tensor("embedding_1_out"), pat.Tensor("embedding_2_out"));
    layernorm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto x1_shape = pir::GetShapeFromValue(match_ctx.Tensor("x1"));
      auto x2_shape = pir::GetShapeFromValue(match_ctx.Tensor("x2"));

      if (x1_shape.size() != x2_shape.size()) {
        return false;
      }

      for (size_t i = 0; i < x1_shape.size(); i++) {
        if (x1_shape.at(i) != x2_shape.at(i)) {
          return false;
        }
      }

      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    auto &combine_op_1 = res.Op(pir::CombineOp::name());
    combine_op_1({&res.Tensor("x1"), &res.Tensor("x2")},
                 {&res.Tensor("combine1_out")});
    auto &combine_op_2 = res.Op(pir::CombineOp::name());
    combine_op_2({&res.Tensor("w1"), &res.Tensor("w2")},
                 {&res.Tensor("combine2_out")});

    const auto &fused_embedding_eltwise_layernorm_op =
        res.Op(paddle::dialect::FusedEmbeddingEltwiseLayernormOp::name(),
               {{
                   {"epsilon", pat.Attr("epsilon")},
               }});
    fused_embedding_eltwise_layernorm_op({&res.Tensor("combine1_out"),
                                          &res.Tensor("combine2_out"),
                                          &res.Tensor("bias"),
                                          &res.Tensor("scale")},
                                         {&res.Tensor("layernorm_out")});
  }
  std::string name() const override {
    return "Fused2EmbeddingEltwiseLayernormPattern";
  }
};

class Fused3EmbeddingEltwiseLayernormPattern
    : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &embedding_1 = pat.Op(paddle::dialect::EmbeddingOp::name(),
                                     {{{"padding_idx", pat.Attr("padding_idx")},
                                       {"sparse", pat.Attr("sparse")}}});
    const auto &embedding_2 = pat.Op(paddle::dialect::EmbeddingOp::name(),
                                     {{{"padding_idx", pat.Attr("padding_idx")},
                                       {"sparse", pat.Attr("sparse")}}});
    const auto &embedding_3 = pat.Op(paddle::dialect::EmbeddingOp::name(),
                                     {{{"padding_idx", pat.Attr("padding_idx")},
                                       {"sparse", pat.Attr("sparse")}}});
    const auto &add1 = pat.Op(paddle::dialect::AddOp::name());
    const auto &add2 = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    embedding_1({&pat.Tensor("x1"), &pat.Tensor("w1")},
                {&pat.Tensor("embedding_1_out")});
    embedding_2({&pat.Tensor("x2"), &pat.Tensor("w2")},
                {&pat.Tensor("embedding_2_out")});
    pat.Tensor("add1_out") =
        add1(pat.Tensor("embedding_1_out"), pat.Tensor("embedding_2_out"));
    embedding_3({&pat.Tensor("x3"), &pat.Tensor("w3")},
                {&pat.Tensor("embedding_3_out")});
    pat.Tensor("add2_out") =
        add2(pat.Tensor("add1_out"), pat.Tensor("embedding_3_out"));
    layernorm(
        {&pat.Tensor("add2_out"), &pat.Tensor("scale"), &pat.Tensor("bias")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto x1_shape = pir::GetShapeFromValue(match_ctx.Tensor("x1"));
      auto x2_shape = pir::GetShapeFromValue(match_ctx.Tensor("x2"));
      auto x3_shape = pir::GetShapeFromValue(match_ctx.Tensor("x3"));
      if (x1_shape.size() != x2_shape.size() ||
          x1_shape.size() != x3_shape.size()) {
        return false;
      }
      for (size_t i = 0; i < x1_shape.size(); i++) {
        if (x1_shape.at(i) != x2_shape.at(i) ||
            x1_shape.at(i) != x2_shape.at(i)) {
          return false;
        }
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto &combine_op_1 = res.Op(pir::CombineOp::name());
    combine_op_1({&res.Tensor("x1"), &res.Tensor("x2"), &res.Tensor("x3")},
                 {&res.Tensor("combine1_out")});

    auto &combine_op_2 = res.Op(pir::CombineOp::name());
    combine_op_2({&res.Tensor("w1"), &res.Tensor("w2"), &res.Tensor("w3")},
                 {&res.Tensor("combine2_out")});

    const auto &fused_embedding_eltwise_layernorm_op =
        res.Op(paddle::dialect::FusedEmbeddingEltwiseLayernormOp::name(),
               {{
                   {"epsilon", pat.Attr("epsilon")},
               }});
    fused_embedding_eltwise_layernorm_op({&res.Tensor("combine1_out"),
                                          &res.Tensor("combine2_out"),
                                          &res.Tensor("bias"),
                                          &res.Tensor("scale")},
                                         {&res.Tensor("layernorm_out")});
  }
  std::string name() const override {
    return "Fused3EmbeddingEltwiseLayernormPattern";
  }
};

class EmbeddingEltwiseLayernormFusePass : public pir::PatternRewritePass {
 public:
  EmbeddingEltwiseLayernormFusePass()
      : pir::PatternRewritePass("embedding_eltwise_layernorm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(Fused2EmbeddingEltwiseLayernormPattern().Build(context));
    ps.Add(Fused3EmbeddingEltwiseLayernormPattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedEmbeddingEltwiseLayerNormPass() {
  return std::make_unique<EmbeddingEltwiseLayernormFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(embedding_eltwise_layernorm_fuse_pass,
                 EmbeddingEltwiseLayernormFusePass);
