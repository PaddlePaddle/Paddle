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

#include "paddle/fluid/pir/transforms/fusion/matmul_transpose_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class MatmulOutTransposeFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulOutTransposeFusePattern"; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
   paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x")},
                                   {"transpose_y", pat.Attr("transpose_y")}});
    
    const auto &transpose_op = pat.Op(paddle::dialect::TransposeOp::name(),
                                    {{"perm", pat.Attr("perm")}});

    matmul_op({&pat.Tensor("x"), &pat.Tensor("y")},
              {&pat.Tensor("matmul_out")});

    transpose_op({&pat.Tensor("matmul_out")}, 
                 {&pat.Tensor("transpose_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor(x));
      auto y_shape = pir::GetShapeFromValue(match_ctx.Tensor(y));
      if (x_shape.size() < 2 || y_shape.size() < 2)
        return false;
      const auto &perm = match_ctx.Attr<std::vector<int64_t>>("perm")
      const auto perm_size = perm.size()
      for (size_t i = 0; i < perm_size - 2; ++i) {
        if (perm[i] != i):
          return false; 
      }
      if ((perm[perm_size - 1] != perm_size - 2) && (perm[perm_size - 2] != perm_size - 1))
        return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_matmul_transpose_op = res.Op(paddle::dialect::MatmulOp::name(),
                                                   {{"transpose_x", !pat.Attr("transpose_y")},
                                                    {"transpose_y", !pat.Attr("transpose_x")}});
    fused_matmul_transpose_op({&res.Tensor("y"), &res.Tensor("x")});    
 }
};


class MatmulXTransposeFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulXTransposeFusePattern"; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
   paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x")},
                                   {"transpose_y", pat.Attr("transpose_y")}});
    
    const auto &transpose_op = pat.Op(paddle::dialect::TransposeOp::name(),
                                    {{"perm", pat.Attr("perm")}});

    transpose_op({&pat.Tensor("x")}, 
                 {&pat.Tensor("x_transpose")});

    matmul_op({&pat.Tensor("x_transpose"), &pat.Tensor("y")},
              {&pat.Tensor("matmul_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor(x));
      auto y_shape = pir::GetShapeFromValue(match_ctx.Tensor(y));
      if (x_shape.size() < 2 || y_shape.size() < 2)
        return false;
      const auto &perm = match_ctx.Attr<std::vector<int64_t>>("perm")
      const auto perm_size = perm.size()
      for (size_t i = 0; i < perm_size - 2; ++i) {
        if (perm[i] != i):
          return false; 
      }
      if ((perm[perm_size - 1] != perm_size - 2) && (perm[perm_size - 2] != perm_size - 1))
        return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_matmul_transpose_op = res.Op(paddle::dialect::MatmulOp::name(),
                                                   {{"transpose_x", !pat.Attr("transpose_x")},
                                                    {"transpose_y", pat.Attr("transpose_y")}});
    fused_matmul_transpose_op({&res.Tensor("x"), &res.Tensor("y")});    
 }
};


class MatmulYTransposeFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulYTransposeFusePattern"; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
   paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x")},
                                   {"transpose_y", pat.Attr("transpose_y")}});
    
    const auto &transpose_op = pat.Op(paddle::dialect::TransposeOp::name(),
                                    {{"perm", pat.Attr("perm")}});

    transpose_op({&pat.Tensor("y")}, 
                 {&pat.Tensor("y_transpose")});

    matmul_op({&pat.Tensor("x"), &pat.Tensor("y_transpose")},
              {&pat.Tensor("matmul_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor(x));
      auto y_shape = pir::GetShapeFromValue(match_ctx.Tensor(y));
      if (x_shape.size() < 2 || y_shape.size() < 2)
        return false;
      const auto &perm = match_ctx.Attr<std::vector<int64_t>>("perm")
      const auto perm_size = perm.size()
      for (size_t i = 0; i < perm_size - 2; ++i) {
        if (perm[i] != i):
          return false; 
      }
      if ((perm[perm_size - 1] != perm_size - 2) && (perm[perm_size - 2] != perm_size - 1))
        return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_matmul_transpose_op = res.Op(paddle::dialect::MatmulOp::name(),
                                                   {{"transpose_x", pat.Attr("transpose_x")},
                                                    {"transpose_y", !pat.Attr("transpose_y")}});
    fused_matmul_transpose_op({&res.Tensor("x"), &res.Tensor("y")});    
 }
};


class MatmulTransposeFusePass : public pir::PatternRewritePass {
 public:
  MatmulTransposeFusePass()
      : pir::PatternRewritePass("matmul_transpose_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<MatmulTransposeOutFusePattern>(context));
    ps.Add(paddle::drr::Create<MatmulTransposeXFusePattern>(context));
    ps.Add(paddle::drr::Create<MatmulTransposeYFusePattern>(context));
    // Add three pattern here
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulTransposeFusePass() {
  return std::make_unique<MatmulTransposeFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_transpose_fuse_pass, MatmulTransposeFusePass);