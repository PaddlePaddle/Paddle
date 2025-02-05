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

#include "paddle/fluid/pir/transforms/xpu/add_activation_xpu_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

/*
fuse ele_add + activation block in to xpu_ele_fusion op
For example:
graph:
                    ele_x
                      |
                      |
                 elementwise_add -----ele_y
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                    Input
                      |     ele_y
                      |    /
                      |   /
  Input_max ---- add_act_fusion ---- ele_y_max
                      |    \
                      |     \
                      |      OutputMax
                    Output
*/
namespace {

template <int act_type>
class AddActivationPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "AddActivationPattern"; }

  // rewrite pattern operator()
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string act_op_name;
    if (act_type == 1) {
      act_op_name = paddle::dialect::ReluOp::name();
    } else if (act_type == 4) {
      act_op_name = paddle::dialect::GeluOp::name();
    } else {
      common::errors::InvalidArgument("Unsupported activation type.");
    }

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &act = pat.Op(act_op_name);
    add({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("add_out")});
    act({&pat.Tensor("add_out")}, {&pat.Tensor("out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::vector<int64_t> x_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("x"));
      std::vector<int64_t> y_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (x_shape.size() == y_shape.size()) {
        return true;
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &add_act_xpu = res.Op(paddle::dialect::AddActXpuOp::name(),
                                     {{{"act_type", res.Int32Attr(act_type)}}});
    add_act_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.Tensor("y"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("out"), &res.Tensor("out_max")});
  }
};

class AddActivationXpuFusePass : public pir::PatternRewritePass {
 public:
  AddActivationXpuFusePass()
      : pir::PatternRewritePass("add_activation_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    // "RELU": 1, "GELU": 4
    ps.Add(paddle::drr::Create<AddActivationPattern<1>>(context));
    ps.Add(paddle::drr::Create<AddActivationPattern<4>>(context));
    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAddActivationXpuFusePass() {
  return std::make_unique<AddActivationXpuFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(add_activation_xpu_fuse_pass, AddActivationXpuFusePass);
