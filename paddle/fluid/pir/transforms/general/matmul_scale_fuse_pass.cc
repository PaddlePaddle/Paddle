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

#include "paddle/fluid/pir/transforms/general/matmul_scale_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class MatmulScaleFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulScaleFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("transpose_x")},
                                    {"transpose_y", pat.Attr("transpose_y")}});

    matmul_op({&pat.Tensor("x"), &pat.Tensor("w")},
              {&pat.Tensor("matmul_out")});
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &scale_op =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    scale_op({&pat.Tensor("matmul_out"), &full_op()},
             {&pat.Tensor("scale_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("w"))) {
        return false;
      }
      return std::abs(match_ctx.Attr<float>("bias")) <= 1e-6;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &full_op_res = res.Op(paddle::dialect::FullOp::name(),
                                     {{"shape", pat.Attr("shape")},
                                      {"value", pat.Attr("value")},
                                      {"dtype", pat.Attr("dtype")},
                                      {"place", pat.Attr("place")}});
    const auto &scale_op_res =
        res.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", res.Float32Attr(0.0)},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    const auto &matmul_op_res =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("transpose_x")},
                {"transpose_y", pat.Attr("transpose_y")}});
    scale_op_res({&res.Tensor("w"), &full_op_res()},
                 {&res.Tensor("scale_res_out")});
    matmul_op_res({&res.Tensor("x"), &res.Tensor("scale_res_out")},
                  {&res.Tensor("scale_out")});
  }
};

class MatmulScaleFusePass : public pir::PatternRewritePass {
 public:
  MatmulScaleFusePass()
      : pir::PatternRewritePass("matmul_scale_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<MatmulScaleFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulScaleFusePass() {
  return std::make_unique<MatmulScaleFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_scale_fuse_pass, MatmulScaleFusePass);
