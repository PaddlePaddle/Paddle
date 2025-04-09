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

#include "paddle/fluid/pir/transforms/onednn/matmul_reshape_add_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class MatmulReshapeElementwiseAddFusePattern
    : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;  // Decide input direction of add

 public:
  MatmulReshapeElementwiseAddFusePattern(const std::string &matmul_name,
                                         const std::string &fused_matmul_name,
                                         uint32_t benefit,
                                         bool as_x)
      : matmul_name_(matmul_name),
        fused_matmul_name_(fused_matmul_name),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override {
    return "MatmulReshapeElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("value")},
                {"dtype", pat.Attr("dtype")},
                {"place", pat.Attr("place")}});
    const auto &reshape1 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape1({&pat.Tensor("x"), &full_int_array1()},
             {&pat.Tensor("reshape_x")});

    const auto &full_int_array2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("value2")},
                {"dtype", pat.Attr("dtype2")},
                {"place", pat.Attr("place2")}});
    const auto &reshape2 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape2({&pat.Tensor("w"), &full_int_array2()},
             {&pat.Tensor("reshape_w")});

    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    pat.Tensor("matmul_out") =
        matmul(pat.Tensor("reshape_x"), pat.Tensor("reshape_w"));

    const auto &full_int_array3 =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    pat.Tensor("shape") = full_int_array3();
    const auto &reshape3 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape3({&pat.Tensor("matmul_out"), &pat.Tensor("shape")},
             {&pat.Tensor("out")});

    const auto &add_ = pat.Op(paddle::dialect::AddOp::name());
    pat.Tensor("add_out") = as_x_ ? add_(pat.Tensor("out"), pat.Tensor("y"))
                                  : add_(pat.Tensor("y"), pat.Tensor("out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
      if (!w_dtype.isa<pir::Float16Type>() &&
          !w_dtype.isa<pir::Float32Type>() &&
          !w_dtype.isa<pir::Float64Type>()) {
        return false;
      }

      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("reshape_w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("reshape_x"));
      auto y_dims = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      auto origin_x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));

      if (origin_x_dims.size() < 2) {
        return false;
      }

      if (w_dims.size() != 2 || x_dims.size() < 2) {
        return false;
      }
      // Currently，FcOp and GemmEpilogueOp support only RRR format
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0) ||
          match_ctx.Attr<bool>("trans_x") == true ||
          match_ctx.Attr<bool>("trans_y") == true) {
        return false;
      }

      if (y_dims.size() == 1) {
        return y_dims.at(0) == w_dims.at(1);
      }

      if (y_dims.size() == 2) {
        return y_dims.at(0) == 1 && y_dims.at(1) == w_dims.at(1);
      }

      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &in_num_col_dims_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("reshape_x"));
          return static_cast<int>(x_dims.size()) - 1;
        });

    const auto &full_1 = res.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("value")},
                                 {"dtype", pat.Attr("dtype")},
                                 {"place", pat.Attr("place")}});
    const auto &reshape1_ = res.Op(paddle::dialect::ReshapeOp::name());
    reshape1_({&res.Tensor("x"), &full_1()},
              {&res.Tensor("reshape_x"), &res.Tensor("reshape_x_xshape")});

    const auto &full_2 = res.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("value2")},
                                 {"dtype", pat.Attr("dtype2")},
                                 {"place", pat.Attr("place2")}});
    const auto &reshape2_ = res.Op(paddle::dialect::ReshapeOp::name());
    reshape2_({&res.Tensor("w"), &full_2()},
              {&res.Tensor("reshape_w"), &res.Tensor("reshape_w_xshape")});

    const auto &fc_op = res.Op(fused_matmul_name_,
                               {{
                                   {"in_num_col_dims", in_num_col_dims_attr},
                                   {"activation_type", res.StrAttr("")},
                                   {"padding_weights", res.BoolAttr(false)},
                               }});
    fc_op(
        {&res.Tensor("reshape_x"), &res.Tensor("reshape_w"), &res.Tensor("y")},
        {&res.Tensor("add_out")});
  }
};

class MatmulReshapeAddFusePass : public pir::PatternRewritePass {
 public:
  MatmulReshapeAddFusePass()
      : pir::PatternRewritePass("matmul_reshape_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    std::vector<bool> bool_set = {false, true};
    int benefit_idx = 1;
    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<MatmulReshapeElementwiseAddFusePattern>(
          context,
          paddle::dialect::MatmulOp::name(),
          paddle::dialect::FcOp::name(),
          benefit_idx,
          as_x));
      benefit_idx++;
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulReshapeAddPass() {
  // pd_op.matmul + reshape + pd_op.add -> onednn_op.fused_matmul
  return std::make_unique<MatmulReshapeAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_reshape_add_fuse_pass, MatmulReshapeAddFusePass);
