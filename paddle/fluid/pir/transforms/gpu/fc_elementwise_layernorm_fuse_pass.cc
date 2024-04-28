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

#include "paddle/fluid/pir/transforms/gpu/fc_elementwise_layernorm_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FcElementwiseLayerNormFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FcElementwiseLayerNormFusePattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
               });
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias0")},
       {&pat.Tensor("fc_out")});
    add({&pat.Tensor("fc_out"), &pat.Tensor("y")}, {&pat.Tensor("add_out")});
    layernorm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias1")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
      if (!x_dtype.isa<pir::Float16Type>() &&
          !x_dtype.isa<pir::Float32Type>()) {
        return false;
      }

      int64_t layer_norm_x = 1;
      auto fc_out_dims = pir::GetShapeFromValue(match_ctx.Tensor("fc_out"));
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      for (size_t i = match_ctx.Attr<int>("begin_norm_axis");
           i < fc_out_dims.size();
           i++) {
        layer_norm_x *= fc_out_dims.at(i);
      }
      if (layer_norm_x == w_dims.at(1)) {
        return true;
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &cast_op_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          return paddle::dialect::TransToPhiDataType(x_dtype);
        });
    const auto &cast_op_1 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_bias1") = cast_op_1(res.Tensor("bias1"));
    const auto &cast_op_2 =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    res.Tensor("casted_scale") = cast_op_2(res.Tensor("scale"));

    const auto &fused_fc_elementwise_op =
        res.Op(paddle::dialect::FusedFcElementwiseLayernormOp::name(),
               {{
                   {"x_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
               }});
    fused_fc_elementwise_op({&res.Tensor("x"),
                             &res.Tensor("w"),
                             &res.Tensor("y"),
                             &res.Tensor("bias0"),
                             &res.Tensor("casted_scale"),
                             &res.Tensor("casted_bias1")},
                            {&res.Tensor("layernorm_out"),
                             &res.Tensor("layernorm_mean"),
                             &res.Tensor("layernorm_variance")});
  }
};

class FcElementwiseLayerNormFusePass : public pir::PatternRewritePass {
 public:
  FcElementwiseLayerNormFusePass()
      : pir::PatternRewritePass("fc_elementwise_layernorm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FcElementwiseLayerNormFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcElementwiseLayerNormFusePass() {
  return std::make_unique<FcElementwiseLayerNormFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_elementwise_layernorm_fuse_pass,
                 FcElementwiseLayerNormFusePass);
