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

#include "paddle/fluid/pir/transforms/gpu/fc_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include <iostream>

namespace {

// 我们希望cutlass是可选的
// 如果关闭cutlass 那么使用FcOp，只能匹配[M, N]+[1,N]的模式，激活只支持["","relu"]
// 如果开启cutlass 那么使用GemmEpilogueOp, 额外支持[M, N]+[M, N]的模式，激活支持["", "relu", "gelu", "sigmoid", "silu"]
// 可能扩展的包括 其他Layout(如RCR)，其他激活(如leakyRelu)，详尽的输入参数。
// 需要注意的是：两种Op共用FCInferMeta函数，我们放宽了该函数的约束以匹配额外模式。也就是说FcOp不能处理的模式，目前只在pass的约束中过滤，FCInferMeta中的check被取消了。
#define USE_CUTLASS

class MatmulAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = add(pat.Tensor("matmul_out"), pat.Tensor("y"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto y_dims = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (w_dims.size() != 2 || x_dims.size() < 2) {
        return false;
      }
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0) ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }
      
      if (y_dims.size() == 1) {
        return y_dims.at(0) == w_dims.at(1);
      }
#ifdef USE_CUTLASS
      // kai mod 要融合 M*N + N 和 M*N + M*N 两种elementwiseAdd模式。
      // 要求bias的维度：如果是1，那么需要是[N]。如果是2，那么要么是[1,N],要么是[M,N]。如果是大于2，则除最后一维外，和x相同；最后一维和w_dims.at[1]相同。
      if (y_dims.size() == x_dims.size()){
        if (y_dims.size() == 2) {
          return ((y_dims.at(0) == 1) || (y_dims.at(0) == x_dims.at(0))) && y_dims.at(1) == w_dims.at(1);
        }
        for(size_t ii = 0; ii < x_dims.size()-1; ii++){
          if(y_dims.at(ii) != x_dims.at(ii)){
            return false;
          }
        }
        return y_dims.at(y_dims.size() - 1) == w_dims.at(1);
      }
#else
      if (y_dims.size() == 2) {
        return y_dims.at(0) == 1 && y_dims.at(1) == w_dims.at(1);
      }
#endif
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return static_cast<int>(x_dims.size()) - 1;
        });

#ifdef USE_CUTLASS
    const auto &fc = res.Op(paddle::dialect::GemmEpilogueOp::name(),
                            {{
                                {"in_num_col_dims", in_num_col_dims_attr},
                                {"activation_type", res.StrAttr("")},
                                {"padding_weights", res.BoolAttr(false)},
                            }});
#else
    const auto &fc = res.Op(paddle::dialect::FcOp::name(),
                            {{
                                {"in_num_col_dims", in_num_col_dims_attr},
                                {"activation_type", res.StrAttr("")},
                                {"padding_weights", res.BoolAttr(false)},
                            }});
#endif
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
       {&res.Tensor("add_out")});
  }
};

class FcWithReluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FcWithReluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
#ifdef USE_CUTLASS
    const auto &fc =
        pat.Op(paddle::dialect::GemmEpilogueOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
#else
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
#endif
    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("y")},
       {&pat.Tensor("fc_out")});
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());
    relu({&pat.Tensor("fc_out")}, {&pat.Tensor("relu_out")});

    // Constrains the activation is none
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("activation_type").empty();
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

#ifdef USE_CUTLASS
    const auto &fc_with_relu =
        res.Op(paddle::dialect::GemmEpilogueOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", res.StrAttr("relu")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
#else
    const auto &fc_with_relu =
        res.Op(paddle::dialect::FcOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", res.StrAttr("relu")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
#endif
    fc_with_relu({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
                 {&res.Tensor("relu_out")});
  }
};

#ifdef USE_CUTLASS
// 这个Pattern是为了匹配 将matmul_out作为AddOp右操作数 的模式
class RightMatmulAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RightMatmulAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = add(pat.Tensor("y"), pat.Tensor("matmul_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto y_dims = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (w_dims.size() != 2 || x_dims.size() < 2) {
        return false;
      }
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0) ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }

      if (y_dims.size() == 1) {
        return y_dims.at(0) == w_dims.at(1);
      }
      // kai mod 要融合 M*N + N 和 M*N + M*N 两种elementwiseAdd模式。
      // 要求bias的维度：如果是1，那么需要是[N]。如果是2，那么要么是[1,N],要么是[M,N]。如果是大于2，则除最后一维外，和x相同；最后一维和w_dims.at[1]相同。
      if (y_dims.size() == x_dims.size()){
        if (y_dims.size() == 2) {
          return ((y_dims.at(0) == 1) || (y_dims.at(0) == x_dims.at(0))) && y_dims.at(1) == w_dims.at(1);
        }
        for(size_t ii = 0; ii < x_dims.size()-1; ii++){
          if(y_dims.at(ii) != x_dims.at(ii)){
            return false;
          }
        }
        return y_dims.at(y_dims.size() - 1) == w_dims.at(1);
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &in_num_col_dims_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return static_cast<int>(x_dims.size()) - 1;
        });
    const auto &fc = res.Op(paddle::dialect::GemmEpilogueOp::name(),
                            {{
                                {"in_num_col_dims", in_num_col_dims_attr},
                                {"activation_type", res.StrAttr("")},
                                {"padding_weights", res.BoolAttr(false)},
                            }});
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
       {&res.Tensor("add_out")});
  }
};

// gelu 有一个approximate属性，待写入
class FcWithGeluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FcWithGeluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::GemmEpilogueOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("y")},
       {&pat.Tensor("fc_out")});
    const auto &gelu = pat.Op(paddle::dialect::GeluOp::name());
    gelu({&pat.Tensor("fc_out")}, {&pat.Tensor("gelu_out")});

    // Constrains the activation is none
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("activation_type").empty();
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fc_with_gelu =
        res.Op(paddle::dialect::GemmEpilogueOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", res.StrAttr("gelu")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
    fc_with_gelu({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
                 {&res.Tensor("gelu_out")});
  }
};
/// other Act

#endif

class FcFusePass : public pir::PatternRewritePass {
 public:
  FcFusePass() : pir::PatternRewritePass("fc_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<MatmulAddPattern>(context));
    ps.Add(paddle::drr::Create<FcWithReluPattern>(context));
#ifdef USE_CUTLASS
    ps.Add(paddle::drr::Create<RightMatmulAddPattern>(context));
    ps.Add(paddle::drr::Create<FcWithGeluPattern>(context));

#endif
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcFusePass() {
  return std::make_unique<FcFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_fuse_pass, FcFusePass);
