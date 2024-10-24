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

#include "paddle/fluid/pir/transforms/xpu/fc_xpu_fuse_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

int ConvertActivationType(const std::string &act_type) {
  if (act_type == "") {
    return static_cast<int>(xpu::Activation_t::LINEAR);
  } else if (act_type == "relu") {
    return static_cast<int>(xpu::Activation_t::RELU);
  } else if (act_type == "sigmoid") {
    return static_cast<int>(xpu::Activation_t::SIGMOID);
  } else if (act_type == "tanh") {
    return static_cast<int>(xpu::Activation_t::TANH);
  } else if (act_type == "gelu") {
    return static_cast<int>(xpu::Activation_t::GELU);
  } else if (act_type == "leaky_relu") {
    return static_cast<int>(xpu::Activation_t::LEAKY_RELU);
  } else if (act_type == "exp") {
    return static_cast<int>(xpu::Activation_t::EXP);
  } else if (act_type == "hard_swish") {
    return static_cast<int>(xpu::Activation_t::HARD_SWISH);
  } else if (act_type == "hard_sigmoid") {
    return static_cast<int>(xpu::Activation_t::HARD_SIGMOID);
  } else if (act_type == "swish") {
    return static_cast<int>(xpu::Activation_t::SWISH);
  } else if (act_type == "relu6") {
    return static_cast<int>(xpu::Activation_t::RELU6);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Not support convert activation_type(%s).", act_type));
  }
  return -1;
}

class FCXpuFusePattern : public paddle::drr::DrrPatternBase {
 private:
  bool w_transpose_;
  bool with_bias_;

 public:
  FCXpuFusePattern(bool w_transpose, bool with_bias)
      : w_transpose_(w_transpose), with_bias_(with_bias) {}
  std::string name() const override { return "FCXpuFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &mul = pat.Op(paddle::dialect::MatmulOp::name(),
                             {{"transpose_x", pat.Attr("transpose_x")},
                              {"transpose_y", pat.Attr("transpose_y")}});

    mul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("mul_out")});
    if (with_bias_) {
      const auto &add = pat.Op(paddle::dialect::AddOp::name());
      add({&pat.Tensor("mul_out"), &pat.Tensor("bias")},
          {&pat.Tensor("add_out")});
    }
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      if (w_shape.size() != 2) {
        return false;
      }
      if (with_bias_) {
        auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
        auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
        if (w_shape.back() != bias_shape.back()) {
          return false;
        }
      }
      if (w_transpose_ != match_ctx.Attr<bool>("transpose_y")) {
        return false;
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    // x shape
    const auto &in_num_col_dims_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return x_shape.size() - 1;
        });
    // reshape scale shape
    const auto &expand_1_shape =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx)
                            -> std::vector<int64_t> {
          return {static_cast<int64_t>(
              phi::backends::xpu::get_xpu_max_ptr_size(-1))};
        });

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    // find fp32 w_max
    const auto &max_op1 =
        res.Op(paddle::dialect::MaxOp::name(),
               {{"axis", res.VectorInt64Attr(std::vector<int64_t>{})},
                {"keepdim", res.BoolAttr(false)}});
    const auto &abs_op = res.Op(paddle::dialect::AbsOp::name());
    res.Tensor("w_abs") = abs_op(res.Tensor("w"));
    res.Tensor("filter_fp_max") = max_op1(res.Tensor("w_abs"));
    // set w_max shape
    const auto &w_max_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          return {1};
        });
    // full 32767 or 127
    const auto &full1 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", w_max_shape_attr},
                                {"value", res.Int32Attr(32767)},
                                {"dtype", res.DataTypeAttr("float32")},
                                {"place", res.PlaceAttr("cpu")}});
    const auto &div = res.Op(paddle::dialect::DivideOp::name());
    // scale1 = fp32_w_max/32767
    res.Tensor("scale_fp32_max") = div(res.Tensor("filter_fp_max"), full1());
    // int16_weight = w / scale1
    res.Tensor("w_quant_tmp") =
        div(res.Tensor("w"), res.Tensor("scale_fp32_max"));
    // cast fp32->int16
    const auto &cast_op = res.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", res.DataTypeAttr("int16")}});
    res.Tensor("w_quant") = cast_op(res.Tensor("w_quant_tmp"));
    // expand to 4 or 6 pointer size
    const auto &expand =
        res.Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
    res.Tensor("res_w_max") = expand(res.Tensor("filter_fp_max"));
    const auto &transpose_y_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return match_ctx.Attr<bool>("transpose_y");
        });
    if (!w_transpose_) {
      // perm
      const auto &perm_attr = res.ComputeAttr(
          [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
            auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
            if (w_shape.size() == 2) {
              return {1, 0};
            } else if (w_shape.size() == 3) {
              return {0, 2, 1};
            } else if (w_shape.size() == 4) {
              return {0, 1, 3, 2};
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Not support convert w_shape.size()(%d).", w_shape.size()));
            }
          });
      const auto &transpose_y =
          res.Op(paddle::dialect::TransposeOp::name(), {{"perm", perm_attr}});
      res.Tensor("w_quant_trans") = transpose_y(res.Tensor("w_quant"));
    }

    const auto &fc_xpu =
        res.Op(paddle::dialect::FcXpuOp::name(),
               {{
                   {"in_num_col_dims", in_num_col_dims_attr},
                   {"transpose_x", pat.Attr("transpose_x")},
                   {"alpha", res.Float32Attr(1.0f)},
                   {"beta", res.Float32Attr(0.f)},
                   {"act_type", res.Int32Attr(ConvertActivationType(""))},
                   {"act_alpha", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    fc_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            w_transpose_ ? &res.Tensor("w_quant")
                         : &res.Tensor("w_quant_trans"),
            &res.Tensor("res_w_max"),
            with_bias_ ? &res.Tensor("bias") : &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {with_bias_ ? &res.Tensor("add_out") : &res.Tensor("mul_out"),
         &res.Tensor("out_max")});
  }
};

class FCXpuFusePass : public pir::PatternRewritePass {
 public:
  FCXpuFusePass() : pir::PatternRewritePass("fc_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    for (auto w_transpose : {true, false}) {
      for (auto with_bias : {true, false}) {
        ps.Add(paddle::drr::Create<FCXpuFusePattern>(
            context, w_transpose, with_bias));
      }
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFCXpuFusePass() {
  return std::make_unique<FCXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_xpu_fuse_pass, FCXpuFusePass);
