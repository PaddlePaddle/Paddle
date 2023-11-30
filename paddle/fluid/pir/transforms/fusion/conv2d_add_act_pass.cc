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
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_add_act_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"
#include "paddle/phi/core/ddim.h"

namespace {

class Conv2dAddActFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AddOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;
    pir::Value conv2d_filter = conv2d_op.filter();

    pir::OpResult conv2d_filter_result =
        conv2d_filter.dyn_cast<pir::OpResult>();
    IR_ENFORCE(conv2d_filter_result);

    pir::Value add_input = op.x();
    IR_ENFORCE(add_input == conv2d_out);

    pir::OpResult add_out = op.out();
    if (!add_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    if (next_op_list.size() == 0) return false;

    auto next_op = next_op_list[0];
    std::string act_name = "";
    if (next_op->dyn_cast<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->dyn_cast<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->dyn_cast<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") return false;

    auto conv2d_fusion_attributes = conv2d_op->attributes();
    conv2d_fusion_attributes["activation"] = rewriter.str_attr(act_name);
    conv2d_fusion_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    conv2d_fusion_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    conv2d_fusion_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    conv2d_fusion_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op = rewriter.Build<paddle::dialect::Conv2dFusionOp>(
        conv2d_op.input().dyn_cast<pir::OpResult>(),
        conv2d_op.filter().dyn_cast<pir::OpResult>(),
        op.y().dyn_cast<pir::OpResult>(),
        pir::Value{}.dyn_cast<pir::OpResult>(),
        conv2d_fusion_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});
    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dDoubleAddActFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AddOp add2_op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    paddle::dialect::AddOp add1_op = pir::GetDefiningOpForInput(add2_op, 1)
                                         ->dyn_cast<paddle::dialect::AddOp>();
    if (!add1_op) return false;

    pir::OpResult add1_out = add1_op.out();
    if (!add1_out.HasOneUse()) return false;

    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(add1_op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(add2_op, 0);
    if (next_op_list.size() == 0) return false;
    auto next_op = next_op_list[0];
    std::string act_name = "";
    if (next_op->dyn_cast<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->dyn_cast<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->dyn_cast<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") {
      return false;
    }

    auto conv2d_fusion_attributes = conv2d_op->attributes();
    conv2d_fusion_attributes["activation"] = rewriter.str_attr(act_name);
    conv2d_fusion_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    conv2d_fusion_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    conv2d_fusion_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    conv2d_fusion_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op = rewriter.Build<paddle::dialect::Conv2dFusionOp>(
        conv2d_op.input().dyn_cast<pir::OpResult>(),
        conv2d_op.filter().dyn_cast<pir::OpResult>(),
        add1_op.y().dyn_cast<pir::OpResult>(),
        add2_op.x().dyn_cast<pir::OpResult>(),
        conv2d_fusion_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});

    rewriter.EraseOp(add2_op);
    rewriter.EraseOp(add1_op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dAddActPass : public pir::PatternRewritePass {
 public:
  Conv2dAddActPass() : pir::PatternRewritePass("conv2d_add_act_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    auto conv2d_add_act_fuse_pattern =
        std::make_unique<Conv2dAddActFusePattern>(
            context,
            1,
            std::vector<std::string>{paddle::dialect::Conv2dFusionOp::name()});
    auto conv2d_doublue_add_act_fuse_pattern =
        std::make_unique<Conv2dDoubleAddActFusePattern>(
            context,
            1,
            std::vector<std::string>{paddle::dialect::Conv2dFusionOp::name()});

    // conv2d+add+add+act->conv2d_fusion
    ps.Add(std::move(conv2d_doublue_add_act_fuse_pattern));
    // conv2d+add+act->conv2d_fusion
    ps.Add(std::move(conv2d_add_act_fuse_pattern));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dAddActPass() {
  return std::make_unique<Conv2dAddActPass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_add_act_pass, Conv2dAddActPass);
