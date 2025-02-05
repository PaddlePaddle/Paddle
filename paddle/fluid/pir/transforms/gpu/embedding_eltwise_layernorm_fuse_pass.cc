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

#include "paddle/fluid/pir/transforms/gpu/embedding_eltwise_layernorm_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Fused2EmbeddingEltwiseLayernormPattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "Fused2EmbeddingEltwiseLayernormPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &embedding_1 = pat.Op(paddle::dialect::EmbeddingOp::name());
    const auto &embedding_2 = pat.Op(paddle::dialect::EmbeddingOp::name());
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    const auto &layernorm = pat.Op(paddle::dialect::LayerNormOp::name(),
                                   {{"epsilon", pat.Attr("epsilon")}});

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

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      auto w1_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w1"));
      auto w2_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w2"));
      if (w1_dtype != w2_dtype || (!w1_dtype.isa<pir::Float16Type>() &&
                                   !w1_dtype.isa<pir::Float32Type>())) {
        return false;
      }

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

    const auto &combine_op_1 = res.Op(pir::CombineOp::name());
    combine_op_1({&res.Tensor("x1"), &res.Tensor("x2")},
                 {&res.Tensor("combine1_out")});
    const auto &combine_op_2 = res.Op(pir::CombineOp::name());
    combine_op_2({&res.Tensor("w1"), &res.Tensor("w2")},
                 {&res.Tensor("combine2_out")});

    const auto &cast_op_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto w1_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w1"));
          return paddle::dialect::TransToPhiDataType(w1_dtype);
        });
    const auto &cast_op_1 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_bias") = cast_op_1(res.Tensor("bias"));
    const auto &cast_op_2 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_scale") = cast_op_2(res.Tensor("scale"));

    const auto &fused_embedding_eltwise_layernorm_op =
        res.Op(paddle::dialect::FusedEmbeddingEltwiseLayernormOp::name(),
               {{
                   {"epsilon", pat.Attr("epsilon")},
               }});
    fused_embedding_eltwise_layernorm_op({&res.Tensor("combine1_out"),
                                          &res.Tensor("combine2_out"),
                                          &res.Tensor("casted_bias"),
                                          &res.Tensor("casted_scale")},
                                         {&res.Tensor("layernorm_out")});
  }
};

class Fused3EmbeddingEltwiseLayernormPattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "Fused3EmbeddingEltwiseLayernormPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &embedding_1 = pat.Op(paddle::dialect::EmbeddingOp::name());
    const auto &embedding_2 = pat.Op(paddle::dialect::EmbeddingOp::name());
    const auto &embedding_3 = pat.Op(paddle::dialect::EmbeddingOp::name());
    const auto &add1 = pat.Op(paddle::dialect::AddOp::name());
    const auto &add2 = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm = pat.Op(paddle::dialect::LayerNormOp::name(),
                                   {{"epsilon", pat.Attr("epsilon")}});

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

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      auto w1_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w1"));
      auto w2_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w2"));
      auto w3_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w3"));
      if (w1_dtype != w2_dtype || w1_dtype != w3_dtype ||
          (!w1_dtype.isa<pir::Float16Type>() &&
           !w1_dtype.isa<pir::Float32Type>())) {
        return false;
      }

      auto x1_shape = pir::GetShapeFromValue(match_ctx.Tensor("x1"));
      auto x2_shape = pir::GetShapeFromValue(match_ctx.Tensor("x2"));
      auto x3_shape = pir::GetShapeFromValue(match_ctx.Tensor("x3"));
      if (x1_shape.size() != x2_shape.size() ||
          x1_shape.size() != x3_shape.size()) {
        return false;
      }
      for (size_t i = 0; i < x1_shape.size(); i++) {
        if (x1_shape.at(i) != x2_shape.at(i) ||
            x1_shape.at(i) != x3_shape.at(i)) {
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
    const auto &cast_op_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto w1_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w1"));
          return paddle::dialect::TransToPhiDataType(w1_dtype);
        });
    const auto &cast_op_1 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_bias") = cast_op_1(res.Tensor("bias"));
    const auto &cast_op_2 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_scale") = cast_op_2(res.Tensor("scale"));

    const auto &fused_embedding_eltwise_layernorm_op =
        res.Op(paddle::dialect::FusedEmbeddingEltwiseLayernormOp::name(),
               {{
                   {"epsilon", pat.Attr("epsilon")},
               }});
    fused_embedding_eltwise_layernorm_op({&res.Tensor("combine1_out"),
                                          &res.Tensor("combine2_out"),
                                          &res.Tensor("casted_bias"),
                                          &res.Tensor("casted_scale")},
                                         {&res.Tensor("layernorm_out")});
  }
};

class EmbeddingEltwiseLayernormFusePass : public pir::PatternRewritePass {
 public:
  EmbeddingEltwiseLayernormFusePass()
      : pir::PatternRewritePass("embedding_eltwise_layernorm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(
        paddle::drr::Create<Fused2EmbeddingEltwiseLayernormPattern>(context));
    ps.Add(
        paddle::drr::Create<Fused3EmbeddingEltwiseLayernormPattern>(context));
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
