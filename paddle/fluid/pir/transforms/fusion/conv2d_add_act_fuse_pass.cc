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

#include "paddle/fluid/pir/transforms/fusion/conv2d_add_act_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/common/ddim.h"

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

    pir::Value conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    pir::Value add_input = op.x();
    IR_ENFORCE(add_input == conv2d_out);

    if (!pir::ValueIsPersitable(op.y())) return false;

    pir::Value add_out = op.out();
    if (!add_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    if (next_op_list.size() != 1) return false;

    auto next_op = next_op_list[0].first;
    std::string act_name = "";
    if (next_op->isa<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->isa<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->isa<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") return false;

    auto op_attributes = conv2d_op->attributes();
    auto padding_algorithm = op_attributes.at("padding_algorithm")
                                 .dyn_cast<pir::StrAttribute>()
                                 .AsString();
    if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
        padding_algorithm != "VALID") {
      return false;
    }
    auto data_format = op_attributes.at("data_format")
                           .dyn_cast<pir::StrAttribute>()
                           .AsString();
    if (data_format != "NCHW" && data_format != "AnyLayout" &&
        data_format != "NHWC") {
      return false;
    }
    auto groups =
        op_attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();
    if (groups < 1) {
      return false;
    }
    op_attributes["activation"] = rewriter.str_attr(act_name);
    op_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    op_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    op_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    op_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op =
        rewriter.Build<paddle::dialect::FusedConv2dAddActOp>(conv2d_op.input(),
                                                             conv2d_op.filter(),
                                                             op.y(),
                                                             pir::Value{},
                                                             op_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});
    rewriter.EraseOp(op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dAdd2ActFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::AddOp add2_op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    paddle::dialect::AddOp add1_op = pir::GetDefiningOpForInput(add2_op, 1)
                                         ->dyn_cast<paddle::dialect::AddOp>();
    if (!add1_op) return false;

    if (!pir::ValueIsPersitable(add1_op.y())) return false;

    pir::Value add1_out = add1_op.out();
    if (!add1_out.HasOneUse()) return false;

    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(add1_op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::Value conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    auto next_op_list = pir::GetUseOpsForOutput(add2_op, 0);
    if (next_op_list.size() != 1) return false;

    auto next_op = next_op_list[0].first;
    std::string act_name = "";
    if (next_op->isa<paddle::dialect::ReluOp>()) {
      act_name = "relu";
    }
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->isa<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
    } else if (next_op->isa<paddle::dialect::SigmoidOp>()) {
      act_name = "sigmoid";
    }
#endif
    if (act_name == "") {
      return false;
    }

    auto op_attributes = conv2d_op->attributes();
    auto padding_algorithm = op_attributes.at("padding_algorithm")
                                 .dyn_cast<pir::StrAttribute>()
                                 .AsString();
    if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
        padding_algorithm != "VALID") {
      return false;
    }
    auto data_format = op_attributes.at("data_format")
                           .dyn_cast<pir::StrAttribute>()
                           .AsString();
    if (data_format != "NCHW" && data_format != "AnyLayout" &&
        data_format != "NHWC") {
      return false;
    }
    auto groups =
        op_attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();
    if (groups < 1) {
      return false;
    }
    op_attributes["activation"] = rewriter.str_attr(act_name);
    op_attributes["split_channels"] =
        rewriter.array_attr(std::vector<pir::Attribute>{});
    op_attributes["exhaustive_search"] = rewriter.bool_attr(false);
    op_attributes["workspace_size_MB"] = rewriter.int32_attr(32);
    op_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    auto conv2d_fuse_op =
        rewriter.Build<paddle::dialect::FusedConv2dAddActOp>(conv2d_op.input(),
                                                             conv2d_op.filter(),
                                                             add1_op.y(),
                                                             add2_op.x(),
                                                             op_attributes);
    rewriter.ReplaceOp(next_op,
                       std::vector<pir::Value>{conv2d_fuse_op.output()});

    rewriter.EraseOp(add2_op);
    rewriter.EraseOp(add1_op);
    rewriter.EraseOp(conv2d_op);
    return true;
  }
};

class Conv2dAddActFusePass : public pir::PatternRewritePass {
 public:
  Conv2dAddActFusePass()
      : pir::PatternRewritePass("conv2d_add_act_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    auto conv2d_add_act_fuse_pattern =
        std::make_unique<Conv2dAddActFusePattern>(
            context,
            1,
            std::vector<std::string>{
                paddle::dialect::FusedConv2dAddActOp::name()});
    auto conv2d_doublue_add_act_fuse_pattern =
        std::make_unique<Conv2dAdd2ActFusePattern>(
            context,
            1,
            std::vector<std::string>{
                paddle::dialect::FusedConv2dAddActOp::name()});

    // conv2d+add+add+act->fused_conv2d_add_act
    ps.Add(std::move(conv2d_doublue_add_act_fuse_pattern));
    // conv2d+add+act->fused_conv2d_add_act
    ps.Add(std::move(conv2d_add_act_fuse_pattern));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dAddActFusePass() {
  return std::make_unique<Conv2dAddActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_add_act_fuse_pass, Conv2dAddActFusePass);
