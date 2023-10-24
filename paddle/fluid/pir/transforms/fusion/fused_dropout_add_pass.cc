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

#include "paddle/fluid/pir/transforms/fusion/fused_dropout_add_pass.h"

#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class FusedDropoutAddPattern
    : public pir::drr::DrrPatternBase<FusedDropoutAddPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &dropout = pat.Op("pd_op.dropout",
                                 {{"p", pat.Attr("p")},
                                  {"is_test", pat.Attr("is_test")},
                                  {"mode", pat.Attr("mod")},
                                  {"seed", pat.Attr("seed")},
                                  {"fix_seed", pat.Attr("fix_seed")}});
    const auto &add = pat.Op("pd_op.add");

    dropout({&pat.Tensor("x"), &pat.Tensor("seed_tensor")},
            {&pat.Tensor("dropout_out"), &pat.Tensor("mask")});
    pat.Tensor("add_out") = add(pat.Tensor("dropout_out"), pat.Tensor("y"));

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_dropout_add =
        res.Op("pd_op.fused_dropout_add",
               {{{"p", pat.Attr("p")},
                 {"is_test", pat.Attr("is_test")},
                 {"mode", pat.Attr("mod")},
                 {"seed", pat.Attr("seed")},
                 {"fix_seed", pat.Attr("fix_seed")}}});
    fused_dropout_add(
        {&res.Tensor("x"), &res.Tensor("y"), &res.Tensor("seed_tensor")},
        {&res.Tensor("add_out"), &res.Tensor("mask")});
  }
};

class FusedDropoutGradAddGradPattern
    : public pir::drr::DrrPatternBase<FusedDropoutAddPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &dropout = pat.Op("pd_op.dropout",
                                 {{"p", pat.Attr("p")},
                                  {"is_test", pat.Attr("is_test")},
                                  {"mode", pat.Attr("mod")},
                                  {"seed", pat.Attr("seed")},
                                  {"fix_seed", pat.Attr("fix_seed")}});
    const auto &add = pat.Op("pd_op.add");

    const auto &add_grad = pat.Op("pd_op.add_grad");
    const auto &dropout_grad = pat.Op("pd_op.dropout_grad",
                                      {{"p", pat.Attr("p")},
                                       {"is_test", pat.Attr("is_test")},
                                       {"mode", pat.Attr("mod")}});

    dropout({&pat.Tensor("x"), &pat.Tensor("seed_tensor")},
            {&pat.Tensor("dropout_out"), &pat.Tensor("mask")});
    pat.Tensor("add_out") = add(pat.Tensor("dropout_out"), pat.Tensor("y"));
    add_grad({&pat.Tensor("dropout_out"),
              &pat.Tensor("y"),
              &pat.Tensor("add_out_grad")},
             {&pat.Tensor("dropout_out_grad"), &pat.Tensor("y_grad")});
    dropout_grad({&pat.Tensor("mask"), &pat.Tensor("dropout_out_grad")},
                 {&pat.Tensor("x_grad")});

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_dropout_add =
        res.Op("pd_op.fused_dropout_add",
               {{{"p", pat.Attr("p")},
                 {"is_test", pat.Attr("is_test")},
                 {"mode", pat.Attr("mod")},
                 {"seed", pat.Attr("seed")},
                 {"fix_seed", pat.Attr("fix_seed")}}});

    const auto &fused_dropout_add_grad =
        res.Op("pd_op.fused_dropout_add_grad",
               {{{"p", pat.Attr("p")},
                 {"is_test", pat.Attr("is_test")},
                 {"mode", pat.Attr("mod")},
                 {"fix_seed", pat.Attr("fix_seed")}}});

    fused_dropout_add(
        {&res.Tensor("x"), &res.Tensor("y"), &res.Tensor("seed_tensor")},
        {&res.Tensor("add_out"), &res.Tensor("mask")});
    fused_dropout_add_grad({&res.Tensor("mask"), &res.Tensor("add_out_grad")},
                           {&res.Tensor("x_grad"), &res.Tensor("y_grad")});
  }
};

class FusedDropoutAddPass : public pir::Pass {
 public:
  FusedDropoutAddPass() : pir::Pass("fused_dropout_add_pass", 1) {}

  bool Initialize(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    ps.Add(FusedDropoutAddPattern().Build(context));
    ps.Add(FusedDropoutGradAddGradPattern().Build(context));
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation *op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(pir::Operation *op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedDropoutAddPass() {
  return std::make_unique<FusedDropoutAddPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fused_dropout_add_pass, FusedDropoutAddPass);
