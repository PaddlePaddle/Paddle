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

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

int getSMVersion() {
  int sm_version = -1;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_CUTLASS)
  sm_version = paddle::platform::GetGPUComputeCapability(
      paddle::platform::GetCurrentDeviceId());
#else
  PADDLE_THROW(paddle::platform::errors::Unavailable(
      "fused_weight_only_linear_pass needs paddle compiled with CUDA."));
#endif
  return sm_version;
}

class FusedWeightOnlyLinearPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    const auto &matmul =
        src.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", src.Attr("matmul_transpose_x")},
                {"transpose_y", src.Attr("matmul_transpose_y")}});
    const auto &parameter = src.Op(
        pir::ParameterOp::name(), {{"parameter_name", src.Attr("param_name")}});
    src.Tensor("w") = parameter();
    src.Tensor("matmul_out") = matmul(src.Tensor("x"), src.Tensor("w"));
    const auto &add = src.Op(paddle::dialect::AddOp::name());
    src.Tensor("add_out") = add(src.Tensor("matmul_out"), src.Tensor("bias"));

    //
    // Constraints.
    //
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          bool matmul_trans_x = match_ctx.Attr<bool>("matmul_transpose_x");
          bool matmul_trans_y = match_ctx.Attr<bool>("matmul_transpose_y");
          if (matmul_trans_x || matmul_trans_y) return false;

          auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          auto bias_dims = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
          if (!(w_dims.size() == 2 && x_dims.size() >= 2 &&
                bias_dims.size() == 1)) {
            return false;
          }

          if (w_dims.at(0) % 64 != 0 || w_dims.at(1) % 16 != 0) return false;

          auto w_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
          if (!w_dtype.isa<pir::Float16Type>() &&
              !w_dtype.isa<pir::BFloat16Type>())
            return false;

          if (x_dims.at(x_dims.size() - 1) != w_dims.at(1)) return false;

          return true;
        });
    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();

    const auto &weight_quantize =
        res.Op(paddle::dialect::WeightQuantizeOp::name(),
               {{"algo", res.StrAttr("weight_only_int8")},
                {"arch", res.Int32Attr(getSMVersion())},
                {"group_size", res.Int32Attr(-1)}});
    weight_quantize({&res.Tensor("w")},
                    {&res.Tensor("quanted_weight_tensor"),
                     &res.Tensor("weight_scale_tensor")});

    const auto &weight_only_linear =
        res.Op(paddle::dialect::WeightOnlyLinearOp::name(),
               {{"weight_dtype", res.StrAttr("int8")},
                {"arch", res.Int32Attr(getSMVersion())},
                {"group_size", res.Int32Attr(-1)}});
    weight_only_linear({&res.Tensor("x"),
                        &res.Tensor("quanted_weight_tensor"),
                        &res.Tensor("bias"),
                        &res.Tensor("weight_scale_tensor")},
                       {&res.Tensor("add_out")});
  }

  std::string name() const override { return "FusedWeightOnlyLinearPattern"; }
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
    if (sm_vesion != 70 && sm_vesion != 75 && sm_vesion != 80 &&
        sm_vesion != 86) {
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
