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

#include "paddle/fluid/pir/transforms/gpu/fused_bn_add_act_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/pass/pass.h"

namespace {

// std::vector<int> GetBnOpIndex(paddle::dialect::AddOp op) {
//   std::vector<int> index;
//   for (auto &i : {0, 1}) {
//     if (pir::GetDefiningOpForInput(op, i)
//             ->isa<paddle::dialect::BatchNormOp>()) {
//       index.push_back(i);
//     }
//   }
//   return index;
// }

// class FusedBnAddActPattern
//     : public pir::OpRewritePattern<paddle::dialect::AddOp> {
//  public:
//   using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

//   bool MatchAndRewrite(
//       paddle::dialect::AddOp op,
//       pir::PatternRewriter &rewriter) const override {  // NOLINT
//     auto index = GetBnOpIndex(op);
//     if (index.size() == 0) {
//       return false;
//     }

//     auto bn_op = pir::GetDefiningOpForInput(op, index[0])
//                      ->dyn_cast<paddle::dialect::BatchNormOp>();
//     if (!bn_op) {
//       return false;
//     }

//     auto bn_out = bn_op.out();
//     if (!bn_out.HasOneUse()) {
//       return false;
//     }

//     auto add_input = op.result(index[0]);
//     PADDLE_ENFORCE_EQ(
//         add_input,
//         bn_out,
//         phi::errors::PreconditionNotMet("The type of add input should be the
//         "
//                                         "same as the type of bn's out."));

//     auto add_out = op.out();
//     if (!add_out.HasOneUse()) {
//       return false;
//     }

//     auto next_op_list = pir::GetUseOpsForOutput(op, 0);
//     if (next_op_list.size() != 1) {
//       return false;
//     }

//     auto next_op = next_op_list[0].first;
//     if (!next_op->isa<paddle::dialect::ReluOp>()) {
//         return false;
//     }

//     auto bn_op_attributes = bn_op.attributes();
//     float momentum =
//     bn_op_attributes.at("momentum").dyn_cast<pir::FloatAttribute>().data();
//     float epsilon =
//     bn_op_attributes.at("epsilon").dyn_cast<pir::FloatAttribute>().data();
//     std::string act_type = "relu";
//     auto fused_bn_add_act_op =
//     rewriter.Build<paddle::dialect::FusedBnAddActivationOp>(
//       bn_op.x(),
//       index[0]==0 ? op.y(): op.x(),
//       bn_op.scale(),
//       bn_op.bias(),
//       bn_op.mean(),
//       bn_op.variance(),
//       momentum,
//       epsilon,
//       act_type
//     );
//     rewriter.ReplaceOp(next_op,
//                        std::vector<pir::Value>{fused_bn_add_act_op.out()});
//     rewriter.EraseOp(op);
//     rewriter.EraseOp(bn_op);
//     return true;
//   }
// };

class FusedBnAddActPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedBnAddActPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    VLOG(1) << "FusedBnAddActPattern.";
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &bn =
        pat.Op(paddle::dialect::BatchNorm_Op::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());

    bn({&pat.Tensor("x"),
        &pat.Tensor("mean"),
        &pat.Tensor("variance"),
        &pat.Tensor("scale"),
        &pat.Tensor("bias")},
       {&pat.Tensor("out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("variance_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    pat.Tensor("add_out") = add(pat.Tensor("out"), pat.Tensor("z"));
    pat.Tensor("relu_out") = relu(pat.Tensor("add_out"));

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto x = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
      if (!x.isa<pir::Float16Type>()) {
        VLOG(1) << "FusedBnAddActPattern !x.isa<pir::Float16Type>(): "
                << "false";
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      VLOG(1) << "data_format: " << data_format;
      if (data_format != "NHWC") {
        return false;
      }
      VLOG(1) << "FusedBnAddActPattern !x.isa<pir::Float16Type>(): "
              << "true";
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_bn_add_act =
        res.Op(paddle::dialect::FusedBnAddActivationOp::name(),
               {
                   {"momentum", pat.Attr("momentum")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"act_type", res.StrAttr("relu")},
               });
    fused_bn_add_act({&res.Tensor("x"),
                      &res.Tensor("z"),
                      &res.Tensor("scale"),
                      &res.Tensor("bias"),
                      &res.Tensor("mean"),
                      &res.Tensor("variance")},
                     {&res.Tensor("relu_out"),
                      &res.Tensor("mean_out"),
                      &res.Tensor("variance_out"),
                      &res.Tensor("saved_mean"),
                      &res.Tensor("saved_variance"),
                      &res.Tensor("reserve_space")});
  }
};

class FusedBnAddActGradPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedBnAddActGradPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    VLOG(1) << "FusedBnAddActGradPattern.";
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &bn =
        pat.Op(paddle::dialect::BatchNorm_Op::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());

    const auto &relu_grad = pat.Op(paddle::dialect::ReluGradOp::name());
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());
    const auto &bn_grad =
        pat.Op(paddle::dialect::BatchNormGradOp::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});
    bn({&pat.Tensor("x"),
        &pat.Tensor("mean"),
        &pat.Tensor("variance"),
        &pat.Tensor("scale"),
        &pat.Tensor("bias")},
       {&pat.Tensor("out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("variance_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    pat.Tensor("add_out") = add(pat.Tensor("out"), pat.Tensor("z"));
    pat.Tensor("relu_out1") = relu(pat.Tensor("add_out"));

    relu_grad(
        {
            &pat.Tensor("relu_out2"),
            &pat.Tensor("relu_out_grad"),
        },
        {&pat.Tensor("add_out_grad")});
    add_grad(
        {
            &pat.Tensor("out"),
            &pat.Tensor("z"),
            &pat.Tensor("add_out_grad"),
        },
        {
            &pat.Tensor("out_grad"),
            &pat.Tensor("z_grad"),
        });
    bn_grad(
        {
            &pat.Tensor("x"),
            &pat.Tensor("scale"),
            &pat.Tensor("bias"),
            &pat.Tensor("mean_out"),
            &pat.Tensor("variance_out"),
            &pat.Tensor("saved_mean"),
            &pat.Tensor("saved_variance"),
            &pat.Tensor("reserve_space"),
            &pat.Tensor("out_grad"),
        },
        {
            &pat.Tensor("x_grad"),
            &pat.Tensor("scale_grad"),
            &pat.Tensor("bias_grad"),
        });

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto x = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
      if (!x.isa<pir::Float16Type>()) {
        VLOG(1) << "FusedBnAddActGradPattern !x.isa<pir::Float16Type>(): "
                << "false";
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      VLOG(1) << "data_format: " << data_format;
      if (data_format != "NHWC") {
        return false;
      }
      VLOG(1) << "FusedBnAddActGradPattern !x.isa<pir::Float16Type>(): "
              << "true";
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_bn_add_act =
        res.Op(paddle::dialect::FusedBnAddActivationOp::name(),
               {
                   {"momentum", pat.Attr("momentum")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"act_type", res.StrAttr("relu")},
               });

    const auto &fused_bn_add_act_grad =
        res.Op(paddle::dialect::FusedBnAddActivationGradOp::name(),
               {
                   {"momentum", pat.Attr("momentum")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"act_type", res.StrAttr("relu")},
               });

    fused_bn_add_act({&res.Tensor("x"),
                      &res.Tensor("z"),
                      &res.Tensor("scale"),
                      &res.Tensor("bias"),
                      &res.Tensor("mean"),
                      &res.Tensor("variance")},
                     {&res.Tensor("relu_out1"),
                      &res.Tensor("mean_out"),
                      &res.Tensor("variance_out"),
                      &res.Tensor("saved_mean"),
                      &res.Tensor("saved_variance"),
                      &res.Tensor("reserve_space")});

    fused_bn_add_act_grad({&res.Tensor("x"),
                           &res.Tensor("scale"),
                           &res.Tensor("bias"),
                           &res.Tensor("relu_out2"),
                           &res.Tensor("saved_mean"),
                           &res.Tensor("saved_variance"),
                           &res.Tensor("reserve_space"),
                           &res.Tensor("relu_out_grad")},
                          {&res.Tensor("x_grad"),
                           &res.Tensor("z_grad"),
                           &res.Tensor("scale_grad"),
                           &res.Tensor("bias_grad")});
  }
};

class FusedBnAddActPass : public pir::PatternRewritePass {
 public:
  FusedBnAddActPass() : pir::PatternRewritePass("fused_bn_add_act_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedBnAddActPattern>(context));
    ps.Add(paddle::drr::Create<FusedBnAddActGradPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedBnAddActPass() {
  return std::make_unique<FusedBnAddActPass>();
}

}  // namespace pir

// REGISTER_IR_PASS(fused_bn_add_act_pass, FusedBnAddActPass);
