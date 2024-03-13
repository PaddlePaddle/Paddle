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

#include "paddle/fluid/pir/transforms/onednn/matmul_elementwise_add_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class MatmulElementwiseAddFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;

 public:
  MatmulElementwiseAddFusePattern(const std::string &matmul_name,
                                  const std::string &fused_matmul_name)
      : matmul_name_(matmul_name), fused_matmul_name_(fused_matmul_name) {}

  std::string name() const override {
    return "MatmulElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    pat.Tensor("add_out") = add(pat.Tensor("Out"), pat.Tensor("residual"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      std::set<bool> bool_sets = {true, false};
      auto result_x = match_ctx.Attr<bool>("transpose_x");
      auto result_y = match_ctx.Attr<bool>("transpose_y");
      if (bool_sets.count(result_x) == 0 || bool_sets.count(result_y) == 0) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_matmul =
        res.Op(fused_matmul_name_,
               {{
                   {"trans_x", pat.Attr("transpose_x")},
                   {"trans_y", pat.Attr("transpose_y")},
                   {"matmul_alpha", res.Float32Attr(1.0f)},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"fused_output_scale", res.Float32Attr(1.0f)},
                   {"fused_reshape_x", res.VectorInt32Attr({})},
                   {"fused_transpose_x", res.VectorInt32Attr({})},
                   {"fused_reshape_y", res.VectorInt32Attr({})},
                   {"fused_transpose_y", res.VectorInt32Attr({})},
                   {"fused_reshape_out", res.VectorInt32Attr({})},
                   {"fused_transpose_out", res.VectorInt32Attr({})},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"scale_x", res.Float32Attr(1.0f)},
                   {"scale_y", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(0.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"force_fp32_output", res.BoolAttr(false)},
               }});

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("add_out")});
  }
};

class MatmulElementwiseAddFusePass : public pir::PatternRewritePass {
 public:
  MatmulElementwiseAddFusePass()
      : pir::PatternRewritePass("matmul_elementwise_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<MatmulElementwiseAddFusePattern>(
        context,
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name()));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulElementwiseAddFusePass() {
  // pd_op.batch_norm + pd_op.relu -> onednn_op.batch_norm
  return std::make_unique<MatmulElementwiseAddFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(matmul_elementwise_add_fuse_pass,
                 MatmulElementwiseAddFusePass);
