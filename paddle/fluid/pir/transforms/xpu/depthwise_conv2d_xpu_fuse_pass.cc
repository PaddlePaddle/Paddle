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

#include "paddle/fluid/pir/transforms/xpu/depthwise_conv2d_xpu_fuse_pass.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/backends/xpu/xpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// Fallback pattern: bare DepthwiseConv2d (no BN, no act, no residual)
//   y = depthwise_conv2d(x, w);
// =>
//   y = conv2d_xpu(x, w, w_max, act=LINEAR);  // no bias, no branch
//
// This pass should run AFTER all other conv2d_*_xpu_fuse_pass variants
// so that fusion-eligible depthwise_conv2d ops are first absorbed by the
// stronger patterns.
class DepthwiseConv2dXpuFusePattern : public paddle::drr::DrrPatternBase {
 public:
  DepthwiseConv2dXpuFusePattern() = default;

  std::string name() const override { return "DepthwiseConv2dXpuFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &dw_conv =
        pat.Op(paddle::dialect::DepthwiseConv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    dw_conv({&pat.Tensor("input"), &pat.Tensor("filter")},
            {&pat.Tensor("dw_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::vector<int64_t> conv_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto paddings_size = match_ctx.Attr<std::vector<int>>("paddings");
      if (conv_input_shape.size() != 4) return false;
      if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
        return false;
      }
      // filter must be persistable (weights)
      if (!pir::ValueIsPersistable(match_ctx.Tensor("filter"))) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

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

    // filter_max = expand( max( abs( filter ) ) )
    const auto &abs_op = res.Op(paddle::dialect::AbsOp::name());
    const auto &max_op =
        res.Op(paddle::dialect::MaxOp::name(),
               {{"axis", res.VectorInt64Attr(std::vector<int64_t>{})},
                {"keepdim", res.BoolAttr(false)}});
    res.Tensor("filter_abs") = abs_op(res.Tensor("filter"));
    res.Tensor("filter_max_scalar") = max_op(res.Tensor("filter_abs"));
    const auto &expand =
        res.Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
    res.Tensor("filter_max") = expand(res.Tensor("filter_max_scalar"));

    const auto &conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type",
                    res.Int32Attr(static_cast<int>(xpu::Activation_t::LINEAR))},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("filter"),
            &res.Tensor("filter_max"),
            &res.InputNoneTensor(),  // no bias
            &res.InputNoneTensor(),  // no branch
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("dw_out"), &res.Tensor("out_max")});
  }
};

class DepthwiseConv2dXpuFusePass : public pir::PatternRewritePass {
 public:
  DepthwiseConv2dXpuFusePass()
      : pir::PatternRewritePass("depthwise_conv2d_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DepthwiseConv2dXpuFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDepthwiseConv2dXpuFusePass() {
  return std::make_unique<DepthwiseConv2dXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(depthwise_conv2d_xpu_fuse_pass, DepthwiseConv2dXpuFusePass);
