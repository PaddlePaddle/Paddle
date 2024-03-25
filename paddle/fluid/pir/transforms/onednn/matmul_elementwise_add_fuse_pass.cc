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
  uint32_t benefit_;
  bool as_x_;  // Decide input direction of add

 public:
  MatmulElementwiseAddFusePattern(const std::string &matmul_name,
                                  const std::string &fused_matmul_name,
                                  uint32_t benefit,
                                  bool as_x)
      : matmul_name_(matmul_name),
        fused_matmul_name_(fused_matmul_name),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override {
    return "MatmulElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    pat.Tensor("add_out") =
        as_x_ ? add(pat.Tensor("Out"), pat.Tensor("residual"))
              : add(pat.Tensor("residual"), pat.Tensor("Out"));

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

class FusedMatmulElementwiseAddFusePattern
    : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;   // Decide input direction of 1st add
  bool as_x2_;  // Decide input direction of 2nd add

 public:
  FusedMatmulElementwiseAddFusePattern(const std::string &matmul_name,
                                       const std::string &fused_matmul_name,
                                       uint32_t benefit,
                                       bool as_x,
                                       bool as_x2)
      : matmul_name_(matmul_name),
        fused_matmul_name_(fused_matmul_name),
        benefit_(benefit),
        as_x_(as_x),
        as_x2_(as_x2) {}

  std::string name() const override {
    return "FusedMatmulElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add2 = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    pat.Tensor("add_out") =
        as_x_ ? add(pat.Tensor("Out"), pat.Tensor("residual"))
              : add(pat.Tensor("residual"), pat.Tensor("Out"));
    pat.Tensor("add_out_end") =
        as_x2_ ? add2(pat.Tensor("add_out"), pat.Tensor("residual2"))
               : add2(pat.Tensor("residual2"), pat.Tensor("add_out"));

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

    const auto &fused_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("residual3") =
        fused_add(res.Tensor("residual1"), res.Tensor("residual2"));

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

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual3")},
                 {&res.Tensor("add_out_end")});
  }
};

class MatmulElementwiseAddFusePass : public pir::PatternRewritePass {
 public:
  MatmulElementwiseAddFusePass()
      : pir::PatternRewritePass("matmul_elementwise_add_fuse_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    std::vector<bool> bool_set = {false, true};
    int benefit_idx = 1;
    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<MatmulElementwiseAddFusePattern>(
          context,
          paddle::dialect::MatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          as_x));
      benefit_idx++;
    }

    for (auto as_x : bool_set)
      for (auto as_x2 : bool_set) {
        ps.Add(paddle::drr::Create<FusedMatmulElementwiseAddFusePattern>(
            context,
            paddle::dialect::MatmulOp::name(),
            paddle::onednn::dialect::FusedMatmulOp::name(),
            benefit_idx,
            as_x,
            as_x2));
        benefit_idx++;
      }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulElementwiseAddFusePass() {
  // pd_op.matmul + pd_op.add -> onednn_op.fused_matmul
  // pd_op.matmul + pd_op.add + pd_op.add -> pd_op.add + onednn_op.fused_matmul
  // -> onednn_op.fused_matmul
  return std::make_unique<MatmulElementwiseAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_elementwise_add_fuse_pass,
                 MatmulElementwiseAddFusePass);
