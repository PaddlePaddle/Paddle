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

#include "paddle/fluid/pir/transforms/fusion/fused_weight_only_linear_pass.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

inline int getSMVersion() {
  int sm_version = 80;
#if defined(PADDLE_WITH_CUDA)
  sm_version = paddle::platform::GetGPUComputeCapability(
      paddle::platform::GetCurrentDeviceId());
#endif
  return sm_version;
}

class FusedWeightOnlyLinearPattern
    : public pir::drr::DrrPatternBase<FusedWeightOnlyLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    pir::drr::SourcePattern src = ctx->SourcePattern();
    const auto &matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_transpose_x")},
                {"transpose_y", src.Attr("matmul_transpose_y")}});
    src.Tensor("matmul_out") = matmul(src.Tensor("x"), src.Tensor("w"));

    const auto &add = src.Op("pd_op.add");
    src.Tensor("add_out") = add(src.Tensor("matmul_out"), src.Tensor("bias"));

    //
    // Constraints.
    //
    src.RequireNativeCall([](const pir::drr::MatchContext &match_ctx) -> bool {
      bool matmul_trans_x = match_ctx.Attr<bool>("matmul_transpose_x");
      bool matmul_trans_y = match_ctx.Attr<bool>("matmul_transpose_y");
      if (matmul_trans_x || matmul_trans_y) return false;

      if (!(match_ctx.Tensor("w").Shape().size() == 2 &&
            match_ctx.Tensor("x").Shape().size() >= 2 &&
            match_ctx.Tensor("bias").Shape().size() == 1)) {
        return false;
      }

      return true;
    });
    //
    // Result Pattern.
    //
    pir::drr::ResultPattern res = src.ResultPattern();

    // quantize weight
    const auto &weight_only_int8_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "weight_only_int8";
        });
    // int arch = getSMVersion();
    const auto &weight_quantize_arch_attr =
        res.Attr([&](const pir::drr::MatchContext &match_ctx) -> std::any {
          return 80;
        });

    const auto &weight_quantize = res.Op(
        "pd_op.weight_quantize",
        {{"algo", weight_only_int8_attr}, {"arch", weight_quantize_arch_attr}});
    weight_quantize({&res.Tensor("w")},
                    {&res.Tensor("quanted_weight_tensor"),
                     &res.Tensor("weight_scale_tensor")});

    const auto &weight_dtype_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "int8";
        });

    const auto &weight_only_linear_arch_attr = res.Attr(
        [&](const pir::drr::MatchContext &match_ctx) -> int { return 80; });
    const auto &weight_only_linear =
        res.Op("pd_op.weight_only_linear",
               {{"weight_dtype", weight_dtype_attr},
                {"arch", weight_only_linear_arch_attr}});
    weight_only_linear({&res.Tensor("x"),
                        &res.Tensor("quanted_weight_tensor"),
                        &res.Tensor("bias"),
                        &res.Tensor("weight_scale_tensor")},
                       {&res.Tensor("add_out")});
  }
};

class FusedWeightOnlyLinearPass : public pir::PatternRewritePass {
 public:
  FusedWeightOnlyLinearPass()
      : pir::PatternRewritePass("fused_weight_only_linear_pass", 4) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(FusedWeightOnlyLinearPattern().Build(context));
    return ps;
  }

  bool CanApplyOn(pir::Operation *op) const override {
    int sm_vesion = getSMVersion();
    if (sm_vesion != 70 && sm_vesion != 80 && sm_vesion != 86 &&
        sm_vesion != 75) {
      return false;
    }
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateFusedWeightOnlyLinearPass() {
  return std::make_unique<FusedWeightOnlyLinearPass>();
}
}  // namespace pir

REGISTER_IR_PASS(fused_weight_only_linear_pass, FusedWeightOnlyLinearPass);
