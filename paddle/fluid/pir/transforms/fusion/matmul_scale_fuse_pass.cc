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

#include "paddle/fluid/pir/transforms/fusion/matmul_scale_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/common/ddim.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class MatmulScaleFusePattern
    : public pir::drr::DrrPatternBase<MatmulScaleFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("transpose_x")},
                                    {"transpose_y", pat.Attr("transpose_y")}});

    matmul_op({&pat.Tensor("x"), &pat.Tensor("y")},
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

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return std::abs(match_ctx.Attr<float>("bias")) <= 1e-6;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &full_op_res = res.Op(paddle::dialect::FullOp::name(),
                                     {{"shape", pat.Attr("shape")},
                                      {"value", pat.Attr("value")},
                                      {"dtype", pat.Attr("dtype")},
                                      {"place", pat.Attr("place")}});
    const auto &scale_op_res =
        res.Op(paddle::dialect::ScaleOp::name(),
               {{"bias",
                 res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
                   return 0.0;
                 })},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    const auto &matmul_op_res =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("transpose_x")},
                {"transpose_y", pat.Attr("transpose_y")}});
    scale_op_res({&res.Tensor("y"), &full_op_res()},
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
    ps.Add(MatmulScaleFusePattern().Build(context));
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
