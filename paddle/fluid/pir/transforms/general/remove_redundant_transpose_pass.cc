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

#include "paddle/fluid/pir/transforms/general/remove_redundant_transpose_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class RemoveRedundantTransposePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "RemoveRedundantTransposePattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &transpose1 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_1")}});
    const auto &transpose2 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_2")}});

    pat.Tensor("ret") = transpose2(transpose1(pat.Tensor("arg_transpose")));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &new_perm_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          const auto &perm1 = match_ctx.Attr<std::vector<int>>("perm_1");
          const auto &perm2 = match_ctx.Attr<std::vector<int>>("perm_2");
          std::vector<int> new_perm;
          for (int v : perm2) {
            new_perm.emplace_back(perm1[v]);
          }
          return new_perm;
        });
    const auto &transpose_continuous =
        res.Op("pd_op.transpose", {{"perm", new_perm_attr}});

    res.Tensor("ret") = transpose_continuous(res.Tensor("arg_transpose"));
  }
};

class RemoveInvalidTransposePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveInvalidTransposePattern"; }
  uint32_t benefit() const override { return 1; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &transpose =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm")}});
    pat.Tensor("ret") = transpose(pat.Tensor("arg_transpose"));
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      const auto &perm = match_ctx.Attr<std::vector<int>>("perm");
      std::vector<int> dst_vector(perm.size());
      std::iota(dst_vector.begin(), dst_vector.end(), 0);
      for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] != dst_vector[i]) {
          return false;
        }
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("ret").Assign(res.Tensor("arg_transpose"));
  }
};

class RemoveRedundantTransposePass : public pir::PatternRewritePass {
 public:
  RemoveRedundantTransposePass()
      : pir::PatternRewritePass("remove_redundant_transpose_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<RemoveRedundantTransposePattern>(context));
    ps.Add(paddle::drr::Create<RemoveInvalidTransposePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateRemoveRedundantTransposePass() {
  return std::make_unique<RemoveRedundantTransposePass>();
}
}  // namespace pir

REGISTER_IR_PASS(remove_redundant_transpose_pass, RemoveRedundantTransposePass);
