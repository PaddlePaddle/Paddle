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

#include "paddle/fluid/pir/transforms/gpu/matmul_horizontal_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class MatmulHorizontalPattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t count_;

 public:
  explicit MatmulHorizontalPattern(size_t count) : count_(count) {}

  uint32_t benefit() const override { return count_; }
  std::string name() const override { return "MatmulHorizontalPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // std::vector<const paddle::drr::Tensor *> out;
    // for (size_t i = 0; i < 3; i++) {
    //     const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
    //     {{"w", pat.Attr("w_" + std::to_string(i))}});

    //     matmul_op({&pat.Tensor("x"), &pat.Tensor("w")},
    //     {&pat.Tensor("matmul_out")}); out.push_back(&pat.Tensor("matmul_out_"
    //     + std::to_string(i)));
    // }

    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul_op_q = pat.Op(paddle::dialect::MatmulOp::name());
    const auto &matmul_op_k = pat.Op(paddle::dialect::MatmulOp::name());
    const auto &matmul_op_v = pat.Op(paddle::dialect::MatmulOp::name());

    matmul_op_q({&pat.Tensor("x"), &pat.Tensor("w_q")}, {&pat.Tensor("q_out")});
    matmul_op_k({&pat.Tensor("x"), &pat.Tensor("w_k")}, {&pat.Tensor("k_out")});
    matmul_op_v({&pat.Tensor("x"), &pat.Tensor("w_v")}, {&pat.Tensor("v_out")});

    pat.AddConstraint(
        [this](const paddle::drr::MatchContext &match_ctx) -> bool {
          std::cout << "AddConstraint done" << std::endl;
          return true;
        });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &combine_1 = res.Op("builtin.combine");
    const auto &concat_1 = res.Op("pd_op.concat", {{"axis", res.Int32Attr(1)}});
    const auto &fused_matmul_op =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", res.BoolAttr(false)},
                {"transpose_y", res.BoolAttr(false)}});

    const auto &split_secs_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("w_q"));
          int64_t last_dim = x_dims[x_dims.size() - 1];
          std::cout << "lastdim : " << last_dim << std::endl;
          return {last_dim, last_dim, last_dim};
        });

    const auto &split_op = res.Op(paddle::dialect::SplitOp::name(),
                                  {
                                      {"sections", split_secs_attr},
                                      {"axis", res.Int32Attr(1)},
                                  });

    std::vector<const paddle::drr::Tensor *> concat_in = {
        &res.Tensor("w_q"), &res.Tensor("w_k"), &res.Tensor("w_v")};

    combine_1({&res.Tensor("w_q"), &res.Tensor("w_k"), &res.Tensor("w_v")},
              {&res.Tensor("combine_1_out")});

    res.Tensor("concat_out") = concat_1(res.Tensor("combine_1_out"));

    fused_matmul_op({&res.Tensor("x"), &res.Tensor("concat_out")},
                    {&res.Tensor("fused_matmul_out")});

    split_op(
        {&res.Tensor("fused_matmul_out")},
        {&res.Tensor("q_out"), &res.Tensor("k_out"), &res.Tensor("v_out")});

    std::cout << "test done" << std::endl;
  }
};

class MatmulHorizontalFusePass : public pir::PatternRewritePass {
 public:
  MatmulHorizontalFusePass()
      : pir::PatternRewritePass("matmul_horizontal_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    size_t count = 3;
    ps.Add(paddle::drr::Create<MatmulHorizontalPattern>(context, count));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulHorizontalFusePass() {
  return std::make_unique<MatmulHorizontalFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(matmul_horizontal_fuse_pass, MatmulHorizontalFusePass);
