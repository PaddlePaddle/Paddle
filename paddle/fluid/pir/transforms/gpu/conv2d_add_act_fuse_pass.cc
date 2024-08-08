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

#include "paddle/fluid/pir/transforms/gpu/conv2d_add_act_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/common/ddim.h"

namespace {

class Conv2dAddActFusePassDrrPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string act_name_;
  bool cutlass_pattern_;
  int sm_version_;
  const std::unordered_set<std::string> conv2d_depthwise_act_set_ = {"relu",
                                                                     "swish"};

 public:
  static const int CUTLASS_NHWC_ALIGNMENT = 8;
  Conv2dAddActFusePassDrrPattern(const std::string &act_name,
                                 bool cutlass_pattern,
                                 int sm_version)
      : act_name_(act_name),
        cutlass_pattern_(cutlass_pattern),
        sm_version_(sm_version) {}
  std::string name() const override { return "Conv2dAddActFusePassDrrPattern"; }
  uint32_t benefit() const override { return cutlass_pattern_ ? 3 : 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &act_op = pat.Op("pd_op." + act_name_);
    pat.Tensor("conv2d_out") =
        conv2d(pat.Tensor("input"), pat.Tensor("filter"));
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("bias"));
    pat.Tensor("act_out") = act_op(pat.Tensor("add_out"));
    pat.AddConstraint([this](
                          const paddle::drr::MatchContext &match_ctx) -> bool {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      auto add_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto bias_ = 1;
      auto add_input_ = 1;
      for (auto &bias_dim : bias_shape) {
        bias_ *= bias_dim;
      }
      for (auto &add_input_dim : add_input_shape) {
        add_input_ *= add_input_dim;
      }
      if (bias_ == add_input_) {
        return false;
      }
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      auto groups = match_ctx.Attr<int>("groups");
      auto filter_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("filter"));
      if (!cutlass_pattern_) {
        auto data_format = match_ctx.Attr<std::string>("data_format");
        if (!(filter_dtype.isa<pir::Float16Type>() ||
              filter_dtype.isa<pir::Float32Type>() ||
              filter_dtype.isa<pir::Float64Type>())) {
          return false;
        }
        if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
            padding_algorithm != "VALID") {
          return false;
        }
        if (groups < 1) {
          return false;
        }
        if (data_format != "NCHW" && data_format != "AnyLayout" &&
            data_format != "NHWC") {
          return false;
        }
      } else {
        auto filter_shape = pir::GetShapeFromValue(match_ctx.Tensor("filter"));
        auto strides_shape = match_ctx.Attr<std::vector<int>>("strides");
        auto dilations_shape = match_ctx.Attr<std::vector<int>>("dilations");
        int stride_h = strides_shape[0];
        int stride_w = strides_shape[1];
        int dilation_h = dilations_shape[0];
        int dilation_w = dilations_shape[1];
        int oc = filter_shape[0];
        int kc = filter_shape[1];
        int kh = filter_shape[2];
        int kw = filter_shape[3];
        if (sm_version_ == 75 && !filter_dtype.isa<pir::Float16Type>()) {
          return false;
        }
        if (!(filter_dtype.isa<pir::Float16Type>() ||
              filter_dtype.isa<pir::Float32Type>() ||
              filter_dtype.isa<pir::BFloat16Type>())) {
          return false;
        }
        if (padding_algorithm != "EXPLICIT") {
          return false;
        }
        int ic = kc * groups;
        if (groups == 1) {
          if (oc % CUTLASS_NHWC_ALIGNMENT != 0 ||
              ic % CUTLASS_NHWC_ALIGNMENT != 0) {
            return false;
          }
        } else if (groups == ic && ic == oc) {
          if (!(kh == 3 && kw == 3) || (kh == 5 && kw == 5)) {
            return false;
          }
          if (!(stride_h == 1 || stride_h == 2)) {
            return false;
          }
          if (stride_h != stride_w) {
            return false;
          }
          if (dilation_h != 1) {
            return false;
          }
          if (dilation_w != 1) {
            return false;
          }
          if (ic % 8 != 0) {
            return false;
          }
          if (!conv2d_depthwise_act_set_.count(act_name_)) {
            return false;
          }
        } else {
          return false;
        }
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &force_backend_runtime_attr = res.ComputeAttr(
        [this](const paddle::drr::MatchContext &match_ctx) -> std::string {
          return cutlass_pattern_ ? "gpu" : "gpudnn";
        });
    const auto &fused_conv2d_add_act = res.Op(
        paddle::dialect::FusedConv2dAddActOp::name(),
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"activation", res.StrAttr(act_name_)},
            {"split_channels", res.VectorInt32Attr({})},
            {"exhaustive_search", res.BoolAttr(false)},
            {"workspace_size_MB", res.Int32Attr(32)},
            {"fuse_alpha", res.Float32Attr(0.0f)},
        }},
        {{{paddle::dialect::kForceBackendAttr, force_backend_runtime_attr}}});
    fused_conv2d_add_act({&res.Tensor("input"),
                          &res.Tensor("filter"),
                          &res.Tensor("bias"),
                          &res.InputNoneTensor()},
                         {&res.Tensor("act_out")});
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

    if (!pir::ValueIsPersistable(add1_op.y())) return false;

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
#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    if (next_op->isa<paddle::dialect::TanhOp>()) {
      act_name = "tanh";
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
    auto conv2d_double_add_act_fuse_pattern =
        std::make_unique<Conv2dAdd2ActFusePattern>(
            context,
            1,
            std::vector<std::string>{
                paddle::dialect::FusedConv2dAddActOp::name()});

// NOTE(liuyuanle): cudnn [8.7, 8.9 now) version has bug when act is
// tanh. Ref to issue
// https://github.com/PaddlePaddle/Paddle/issues/50853
#if CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8700
    const std::unordered_set<std::string> cudnn_act_set({"relu", "tanh"});
#else
    const std::unordered_set<std::string> cudnn_act_set({"relu"});
#endif
    // cutlass related
    const std::unordered_set<int> cutlass_sm = {
        75,
        80,
        85,
        86,
    };
    const std::unordered_set<std::string> cutlass_act_set = {
        "relu", "swish", "leaky_relu", "sigmoid"};
    // get pass attr
    bool use_cutlass = false;
    if (Has(std::string("use_cutlass"))) {
      use_cutlass = Get<bool>(std::string("use_cutlass"));
    }
    int sm_version = 0;
#ifdef PADDLE_WITH_CUDA
    sm_version = paddle::platform::GetGPUComputeCapability(
        paddle::platform::GetCurrentDeviceId());
#endif
    bool cutlass_pattern = false;
    if (use_cutlass && cutlass_sm.count(sm_version)) {
#if defined(PADDLE_WITH_CUTLASS)
      cutlass_pattern = true;
#endif
    }
    for (auto act_name : cudnn_act_set) {
      // conv2d+add+act->fused_conv2d_add_act
      ps.Add(paddle::drr::Create<Conv2dAddActFusePassDrrPattern>(
          context, act_name, false, sm_version));
    }
    if (cutlass_pattern) {
      for (auto act_name : cutlass_act_set) {
        // conv2d+add+act->fused_conv2d_add_act
        ps.Add(paddle::drr::Create<Conv2dAddActFusePassDrrPattern>(
            context, act_name, true, sm_version));
      }
    }

    // conv2d+add+add+act->fused_conv2d_add_act
    ps.Add(std::move(conv2d_double_add_act_fuse_pattern));
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
