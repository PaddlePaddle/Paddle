// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/xpu/conv2d_bn_add_act_xpu_fuse_pass.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/backends/xpu/xpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// Fuse pattern:
//   y = conv2d(x, w);
//   y = batch_norm(y, mean, var, scale, bias);
//   y = add(y, residual);   (or add(residual, y))
//   y = act(y);             // act in {relu, swish, hardswish}
// =>
//   y = conv2d_xpu(x, w_folded, w_max, b_folded,
//                  branch=residual, act=ACT_TYPE);
//
// Switches:
//   - bn_inplace_     : match BatchNorm_Op vs BatchNormOp
//   - is_depthwise_   : match DepthwiseConv2dOp vs Conv2dOp
//   - residual_first_ : add(residual, bn_out) vs add(bn_out, residual)
//   - act_op_name_    : pd_op.relu / swish / hardswish
//   - act_type_       : xpu::Activation_t value
class Conv2dBnAddActFusePattern : public paddle::drr::DrrPatternBase {
 private:
  bool bn_inplace_;
  bool is_depthwise_;
  bool residual_first_;
  std::string act_op_name_;
  int act_type_;

 public:
  Conv2dBnAddActFusePattern(bool bn_inplace,
                            bool is_depthwise,
                            bool residual_first,
                            std::string act_op_name,
                            int act_type)
      : bn_inplace_(bn_inplace),
        is_depthwise_(is_depthwise),
        residual_first_(residual_first),
        act_op_name_(std::move(act_op_name)),
        act_type_(act_type) {}

  std::string name() const override {
    std::string s = "Conv2dBnAddActFusePattern_";
    s += (is_depthwise_ ? "depthwise_" : "conv_");
    s += (bn_inplace_ ? "bn_inplace_" : "bn_");
    s += (residual_first_ ? "resfirst_" : "resafter_");
    auto pos = act_op_name_.rfind('.');
    s += (pos == std::string::npos) ? act_op_name_
                                    : act_op_name_.substr(pos + 1);
    return s;
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(is_depthwise_ ? paddle::dialect::DepthwiseConv2dOp::name()
                             : paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &bn = pat.Op(bn_inplace_ ? paddle::dialect::BatchNorm_Op::name()
                                        : paddle::dialect::BatchNormOp::name(),
                            {
                                {"epsilon", pat.Attr("epsilon")},
                            });

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &relu = pat.Op(act_op_name_);

    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    bn({&pat.Tensor("conv2d_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    if (residual_first_) {
      add({&pat.Tensor("residual"), &pat.Tensor("bn_out")},
          {&pat.Tensor("add_out")});
    } else {
      add({&pat.Tensor("bn_out"), &pat.Tensor("residual")},
          {&pat.Tensor("add_out")});
    }
    relu({&pat.Tensor("add_out")}, {&pat.Tensor("relu_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::vector<int64_t> conv_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto paddings_size = match_ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> bn_bias_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("bn_bias"));
      std::vector<int64_t> filter_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("filter"));
      std::vector<int64_t> bn_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("bn_out"));
      std::vector<int64_t> residual_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual"));

      if (conv_input_shape.size() != 4) return false;
      if (!pir::ValueIsPersistable(match_ctx.Tensor("bn_mean")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_var")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_scale")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_bias"))) {
        return false;
      }
      if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
        return false;
      }
      if (bn_bias_shape.at(0) != filter_shape.at(0)) return false;
      // residual shape must match bn_out shape exactly (element-wise add only)
      if (bn_out_shape.size() != residual_shape.size()) return false;
      for (size_t i = 0; i < bn_out_shape.size(); ++i) {
        if (bn_out_shape[i] != residual_shape[i]) return false;
      }
      // residual must NOT be the conv2d_out itself (avoid self-loop).
      if (match_ctx.Tensor("residual") == match_ctx.Tensor("conv2d_out")) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &bn_var_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          return pir::GetShapeFromValue(match_ctx.Tensor("bn_var"));
        });
    const auto &scale_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto bn_scale_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bn_scale"));
          return {bn_scale_shape[0], 1, 1, 1};
        });
    const auto &expand_1_shape =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx)
                            -> std::vector<int64_t> {
          return {static_cast<int64_t>(
              phi::backends::xpu::get_xpu_max_ptr_size(-1))};
        });
    const auto &paddings_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          auto paddings = match_ctx.Attr<std::vector<int>>("paddings");
          if (paddings.size() == 2) {
            return {paddings[0], paddings[0], paddings[1], paddings[1]};
          } else {
            return paddings;
          }
        });
    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("input"));
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    // BN fold
    const auto &full1 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", bn_var_shape_attr},
                                {"value", pat.Attr("epsilon")},
                                {"dtype", res.DataTypeAttr("float32")},
                                {"place", res.PlaceAttr("cpu")}});
    const auto &var_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("var_add_out") = var_add(res.Tensor("bn_var"), full1());
    const auto &sqrt = res.Op(paddle::dialect::SqrtOp::name());
    res.Tensor("sqrt_out") = sqrt(res.Tensor("var_add_out"));
    const auto &div = res.Op(paddle::dialect::DivideOp::name());
    res.Tensor("new_scale") =
        div(res.Tensor("bn_scale"), res.Tensor("sqrt_out"));
    const auto &reshape_scale = res.Op(paddle::dialect::ReshapeOp::name(),
                                       {{"shape", scale_shape_attr}});
    res.Tensor("res_scale") = reshape_scale(res.Tensor("new_scale"));

    const auto &mul_filter_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("res_filter") =
        mul_filter_op(res.Tensor("filter"), res.Tensor("res_scale"));

    const auto &bn_mean_mul_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("bn_mean_mul_out") =
        bn_mean_mul_op(res.Tensor("bn_mean"), res.Tensor("new_scale"));
    const auto &sub_bias_op = res.Op(paddle::dialect::SubtractOp::name());
    res.Tensor("res_bias") =
        sub_bias_op(res.Tensor("bn_bias"), res.Tensor("bn_mean_mul_out"));

    const auto &abs_op = res.Op(paddle::dialect::AbsOp::name());
    const auto &max_op1 =
        res.Op(paddle::dialect::MaxOp::name(),
               {{"axis", res.VectorInt64Attr(std::vector<int64_t>{})},
                {"keepdim", res.BoolAttr(false)}});
    res.Tensor("res_filter_abs") = abs_op(res.Tensor("res_filter"));
    res.Tensor("filter_max") = max_op1(res.Tensor("res_filter_abs"));
    const auto &expand =
        res.Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
    res.Tensor("res_filter_max") = expand(res.Tensor("filter_max"));

    const auto &conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type", res.Int32Attr(act_type_)},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("res_filter"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("res_bias"),
            &res.Tensor("residual"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("relu_out"), &res.Tensor("out_max")});
  }
};

class Conv2dBnAddActFuseXpuPass : public pir::PatternRewritePass {
 public:
  Conv2dBnAddActFuseXpuPass()
      : pir::PatternRewritePass("conv2d_bn_add_act_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::vector<std::pair<std::string, int>> acts = {
        {paddle::dialect::ReluOp::name(),
         static_cast<int>(xpu::Activation_t::RELU)},
        {paddle::dialect::SwishOp::name(),
         static_cast<int>(xpu::Activation_t::SWISH)},
        {paddle::dialect::HardswishOp::name(),
         static_cast<int>(xpu::Activation_t::HARD_SWISH)},
    };
    for (bool bn_inplace : {true, false}) {
      for (bool is_depthwise : {false, true}) {
        for (bool residual_first : {false, true}) {
          for (const auto &act : acts) {
            ps.Add(
                paddle::drr::Create<Conv2dBnAddActFusePattern>(context,
                                                               bn_inplace,
                                                               is_depthwise,
                                                               residual_first,
                                                               act.first,
                                                               act.second));
          }
        }
      }
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dBnAddActFuseXpuPass() {
  return std::make_unique<Conv2dBnAddActFuseXpuPass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_bn_add_act_xpu_fuse_pass, Conv2dBnAddActFuseXpuPass);
