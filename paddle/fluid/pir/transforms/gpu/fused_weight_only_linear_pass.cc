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

#include "paddle/fluid/pir/transforms/gpu/fused_weight_only_linear_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

int getSMVersion() {
  int sm_version = -1;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_CUTLASS)
  sm_version = paddle::platform::GetGPUComputeCapability(
      paddle::platform::GetCurrentDeviceId());
#endif
  return sm_version;
}

class FusedWeightOnlyLinearWithBiasPattern
    : public paddle::drr::DrrPatternBase {
 private:
  bool reverse_add_;
  std::string algo_;
  int sm_version_;

 public:
  FusedWeightOnlyLinearWithBiasPattern(bool reverse_add,
                                       std::string algo,
                                       int sm_version)
      : reverse_add_(reverse_add),
        algo_(std::move(algo)),
        sm_version_(sm_version) {}

  std::string name() const override {
    return "FusedWeightOnlyLinearWithBiasPattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    const auto &matmul =
        src.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", src.Attr("matmul_transpose_x")},
                {"transpose_y", src.Attr("matmul_transpose_y")}});
    src.Tensor("matmul_out") = matmul(src.Tensor("x"), src.Tensor("w"));
    const auto &add = src.Op(paddle::dialect::AddOp::name());

    src.Tensor("add_out") =
        reverse_add_ ? add(src.Tensor("matmul_out"), src.Tensor("bias"))
                     : add(src.Tensor("bias"), src.Tensor("matmul_out"));

    //
    // Constraints.
    //
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("w"))) {
        return false;
      }
      bool matmul_trans_x = match_ctx.Attr<bool>("matmul_transpose_x");
      bool matmul_trans_y = match_ctx.Attr<bool>("matmul_transpose_y");
      if (matmul_trans_x || matmul_trans_y) return false;

      auto w_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
      if (!w_dtype.isa<pir::Float16Type>() &&
          !w_dtype.isa<pir::BFloat16Type>()) {
        return false;
      }

      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto bias_dims = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      if (!(w_dims.size() == 2 && x_dims.size() >= 2 &&
            bias_dims.size() == 1)) {
        return false;
      }

      if (w_dims.at(0) % 64 != 0 || w_dims.at(1) % 16 != 0) return false;
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0)) return false;

      return true;
    });
    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();

    if (algo_ == "weight_only_int4") {
      // TODO(liuyuanle): When the operator weight_quantize supports
      // weight_only_int4 on gpu version, delete the memory copy.
      const auto &memcpy_d2h =
          res.Op(paddle::dialect::MemcpyD2hOp::name(),
                 {{"dst_place_type", res.Int32Attr(0 /*cpu*/)}});
      res.Tensor("w_cpu") = memcpy_d2h(res.Tensor("w"));
      const auto &weight_quantize =
          res.Op(paddle::dialect::WeightQuantizeOp::name(),
                 {{"algo", res.StrAttr(algo_)},
                  {"arch", res.Int32Attr(sm_version_)},
                  {"group_size", res.Int32Attr(-1)}});
      weight_quantize({&res.Tensor("w_cpu")},
                      {&res.Tensor("quanted_weight_tensor_cpu"),
                       &res.Tensor("weight_scale_tensor_cpu")});

      const auto &memcpy_h2d_1 =
          res.Op(paddle::dialect::MemcpyH2dOp::name(),
                 {{"dst_place_type", res.Int32Attr(1 /*gpu*/)}});
      res.Tensor("quanted_weight_tensor") =
          memcpy_h2d_1(res.Tensor("quanted_weight_tensor_cpu"));
      const auto &memcpy_h2d_2 =
          res.Op(paddle::dialect::MemcpyH2dOp::name(),
                 {{"dst_place_type", res.Int32Attr(1 /*gpu*/)}});
      res.Tensor("weight_scale_tensor") =
          memcpy_h2d_2(res.Tensor("weight_scale_tensor_cpu"));
    } else {
      const auto &weight_quantize =
          res.Op(paddle::dialect::WeightQuantizeOp::name(),
                 {{"algo", res.StrAttr(algo_)},
                  {"arch", res.Int32Attr(sm_version_)},
                  {"group_size", res.Int32Attr(-1)}});

      weight_quantize({&res.Tensor("w")},
                      {&res.Tensor("quanted_weight_tensor"),
                       &res.Tensor("weight_scale_tensor")});
    }

    const auto &weight_only_linear =
        res.Op(paddle::dialect::WeightOnlyLinearOp::name(),
               {{"weight_dtype",
                 res.StrAttr(algo_ == "weight_only_int8" ? "int8" : "int4")},
                {"arch", res.Int32Attr(sm_version_)},
                {"group_size", res.Int32Attr(-1)}});
    weight_only_linear({&res.Tensor("x"),
                        &res.Tensor("quanted_weight_tensor"),
                        &res.Tensor("bias"),
                        &res.Tensor("weight_scale_tensor")},
                       {&res.Tensor("add_out")});
  }
};

class FusedWeightOnlyLinearNoBiasPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string algo_;
  int sm_version_;

 public:
  FusedWeightOnlyLinearNoBiasPattern(std::string algo, int sm_version)
      : algo_(std::move(algo)), sm_version_(sm_version) {}

 public:
  std::string name() const override {
    return "FusedWeightOnlyLinearNoBiasPattern";
  }

  uint32_t benefit() const override { return 1; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    const auto &matmul =
        src.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", src.Attr("matmul_transpose_x")},
                {"transpose_y", src.Attr("matmul_transpose_y")}});
    src.Tensor("matmul_out") = matmul(src.Tensor("x"), src.Tensor("w"));

    //
    // Constraints.
    //
    src.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("w"))) {
        return false;
      }
      bool matmul_trans_x = match_ctx.Attr<bool>("matmul_transpose_x");
      bool matmul_trans_y = match_ctx.Attr<bool>("matmul_transpose_y");
      if (matmul_trans_x || matmul_trans_y) return false;

      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      if (!(w_dims.size() == 2 && x_dims.size() >= 2)) {
        return false;
      }

      if (w_dims.at(0) % 64 != 0 || w_dims.at(1) % 16 != 0) return false;

      auto w_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
      if (!w_dtype.isa<pir::Float16Type>() && !w_dtype.isa<pir::BFloat16Type>())
        return false;

      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0)) return false;

      return true;
    });
    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();

    if (algo_ == "weight_only_int4") {
      // TODO(liuyuanle): When the operator weight_quantize supports
      // weight_only_int4 on gpu version, delete the memory copy.
      const auto &memcpy_d2h =
          res.Op(paddle::dialect::MemcpyD2hOp::name(),
                 {{"dst_place_type", res.Int32Attr(0 /*cpu*/)}});
      res.Tensor("w_cpu") = memcpy_d2h(res.Tensor("w"));
      const auto &weight_quantize =
          res.Op(paddle::dialect::WeightQuantizeOp::name(),
                 {{"algo", res.StrAttr(algo_)},
                  {"arch", res.Int32Attr(sm_version_)},
                  {"group_size", res.Int32Attr(-1)}});
      weight_quantize({&res.Tensor("w_cpu")},
                      {&res.Tensor("quanted_weight_tensor_cpu"),
                       &res.Tensor("weight_scale_tensor_cpu")});

      const auto &memcpy_h2d_1 =
          res.Op(paddle::dialect::MemcpyH2dOp::name(),
                 {{"dst_place_type", res.Int32Attr(1 /*gpu*/)}});
      res.Tensor("quanted_weight_tensor") =
          memcpy_h2d_1(res.Tensor("quanted_weight_tensor_cpu"));
      const auto &memcpy_h2d_2 =
          res.Op(paddle::dialect::MemcpyH2dOp::name(),
                 {{"dst_place_type", res.Int32Attr(1 /*gpu*/)}});
      res.Tensor("weight_scale_tensor") =
          memcpy_h2d_2(res.Tensor("weight_scale_tensor_cpu"));
    } else {
      const auto &weight_quantize =
          res.Op(paddle::dialect::WeightQuantizeOp::name(),
                 {{"algo", res.StrAttr(algo_)},
                  {"arch", res.Int32Attr(sm_version_)},
                  {"group_size", res.Int32Attr(-1)}});

      weight_quantize({&res.Tensor("w")},
                      {&res.Tensor("quanted_weight_tensor"),
                       &res.Tensor("weight_scale_tensor")});
    }
    const auto &weight_only_linear =
        res.Op(paddle::dialect::WeightOnlyLinearOp::name(),
               {{"weight_dtype",
                 res.StrAttr(algo_ == "weight_only_int8" ? "int8" : "int4")},
                {"arch", res.Int32Attr(sm_version_)},
                {"group_size", res.Int32Attr(-1)}});
    weight_only_linear({&res.Tensor("x"),
                        &res.Tensor("quanted_weight_tensor"),
                        &res.InputNoneTensor(),
                        &res.Tensor("weight_scale_tensor")},
                       {&res.Tensor("matmul_out")});
  }
};

class FusedWeightOnlyLinearPass : public pir::PatternRewritePass {
 public:
  FusedWeightOnlyLinearPass()
      : pir::PatternRewritePass("fused_weight_only_linear_pass", 4),
        sm_version_(getSMVersion()) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    std::string algo = "weight_only_int8";
    if (Has("weight_only_algo")) {
      algo = Get<std::string>("weight_only_algo");
    }
    PADDLE_ENFORCE_EQ(algo == "weight_only_int8" || algo == "weight_only_int4",
                      true,
                      common::errors::InvalidArgument(
                          "fused_weight_only_linear_pass only support "
                          "weight_only_int8 or weight_only_int4, but get %s.",
                          algo));

    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedWeightOnlyLinearWithBiasPattern>(
        context, true, algo, sm_version_));
    ps.Add(paddle::drr::Create<FusedWeightOnlyLinearWithBiasPattern>(
        context, false, algo, sm_version_));
    ps.Add(paddle::drr::Create<FusedWeightOnlyLinearNoBiasPattern>(
        context, algo, sm_version_));
    return ps;
  }

  pir::GreedyRewriteConfig InitializeConfig() override {
    pir::GreedyRewriteConfig config;

    // NOTE(liuyuanle): Ensure that WithBiasPattern is executed before
    // NoBiasPattern.
    config.use_top_down_traversal = false;

    config.max_iterations = 10;
    return config;
  }

  bool CanApplyOn(pir::Operation *op) const override {
    if (sm_version_ != 70 && sm_version_ != 75 && sm_version_ != 80 &&
        sm_version_ != 86 && sm_version_ != 89 && sm_version_ != 90) {
      return false;
    }
    return op->num_regions() > 0;
  }

 private:
  int sm_version_;
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateFusedWeightOnlyLinearPass() {
  return std::make_unique<FusedWeightOnlyLinearPass>();
}
}  // namespace pir

REGISTER_IR_PASS(fused_weight_only_linear_pass, FusedWeightOnlyLinearPass);
