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

#include "paddle/fluid/pir/dialect/distributed/transforms/fuse_c_reduce_scatter_add_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class FusedCReducescatterAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedCReducescatterAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &c_reduce_scatter =
        pat.Op(paddle::dialect::CReducescatterOp::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"use_calc_stream", pat.Attr("use_calc_stream")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("out") = c_reduce_scatter(pat.Tensor("x"));

    pat.Tensor("add_out") = add(pat.Tensor("out"), pat.Tensor("bias"));

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &c_reducescatter_add =
        res.Op(paddle::dialect::CReducescatterAddOp::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"nranks", pat.Attr("num")},
                {"use_calc_stream", pat.Attr("use_calc_stream")}});

    c_reducescatter_add({&res.Tensor("x")}, {&res.Tensor("bias")});
  }
};

class FuseCReducescatterAddPass : public pir::PatternRewritePass {
 public:
  FuseCReducescatterAddPass()
      : pir::PatternRewritePass("fuse_c_reducescatter_add_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedCReducescatterAddPattern>(context));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFuseCReducescatterAddPass() {
  return std::make_unique<FuseCReducescatterAddPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fuse_c_reducescatter_add_pass, FuseCReducescatterAddPass);
