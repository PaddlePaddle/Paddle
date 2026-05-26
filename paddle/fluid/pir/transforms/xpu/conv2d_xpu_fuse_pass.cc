// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Unified PIR fusion pass that absorbs:
//   * conv2d_bn_xpu_fuse_pass        (conv + bn)
//   * conv2d_bn_act_xpu_fuse_pass    (conv + bn + act)
//   * conv2d_bn_add_act_xpu_fuse_pass(conv + bn + add + act)
//   * depthwise_conv2d_xpu_fuse_pass (bare depthwise_conv2d)
//
// Design follows fc_xpu_fuse_pass.cc: a single PatternRewritePass with
// multiple DRR Patterns, each matching a different op-combination and
// emitting conv2d_xpu directly. Pattern priority is controlled via
// benefit() so that the longest matching subgraph wins.
//
// Does NOT absorb conv2d_add_fuse_pass: that pass has independent
// int8/int16 weight quantization logic (cast + scale tensor) which is
// orthogonal to BN folding and would pollute the unified pattern set.

#include "paddle/fluid/pir/transforms/xpu/conv2d_xpu_fuse_pass.h"

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/backends/xpu/xpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// ============================================================================
// Shared ComputeAttr helpers
// ============================================================================

// 4-element paddings: pad-2 -> {h, h, w, w}; pad-4 stays as is.
inline auto MakePaddingsAttr(paddle::drr::ResultPattern* res) {
  return res->ComputeAttr(
      [](const paddle::drr::MatchContext& match_ctx) -> std::vector<int> {
        auto paddings = match_ctx.Attr<std::vector<int>>("paddings");
        if (paddings.size() == 2) {
          return {paddings[0], paddings[0], paddings[1], paddings[1]};
        }
        return paddings;
      });
}

// Output dtype propagation (only fp32 is supported by XDNN currently).
inline auto MakeOutDtypeAttr(paddle::drr::ResultPattern* res) {
  return res->ComputeAttr(
      [](const paddle::drr::MatchContext& match_ctx) -> phi::DataType {
        auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("input"));
        if (x_dtype.isa<pir::Float32Type>()) {
          return phi::DataType::FLOAT32;
        }
        return phi::DataType::UNDEFINED;
      });
}

// {xpu_max_ptr_size} -- shape used to expand a scalar filter_max to the
// per-cluster array consumed by XDNN.
inline auto MakeExpand1ShapeAttr(paddle::drr::ResultPattern* res) {
  return res->ComputeAttr(
      [](const paddle::drr::MatchContext& match_ctx) -> std::vector<int64_t> {
        return {
            static_cast<int64_t>(phi::backends::xpu::get_xpu_max_ptr_size(-1))};
      });
}

// Shape of bn_var tensor (1-D, length = num_channels).
inline auto MakeBnVarShapeAttr(paddle::drr::ResultPattern* res) {
  return res->ComputeAttr(
      [](const paddle::drr::MatchContext& match_ctx) -> std::vector<int64_t> {
        return pir::GetShapeFromValue(match_ctx.Tensor("bn_var"));
      });
}

// {C, 1, 1, 1} -- shape used to broadcast the per-channel BN scale onto
// a 4-D filter tensor.
inline auto MakeScaleShapeAttr(paddle::drr::ResultPattern* res) {
  return res->ComputeAttr(
      [](const paddle::drr::MatchContext& match_ctx) -> std::vector<int64_t> {
        auto bn_scale_shape =
            pir::GetShapeFromValue(match_ctx.Tensor("bn_scale"));
        return {bn_scale_shape[0], 1, 1, 1};
      });
}

// ============================================================================
// Shared subgraph emitters
// ============================================================================

// Build the BN-fold subgraph in `res` and produce three named tensors:
//   "res_filter"     = filter * (bn_scale / sqrt(bn_var + eps))
//   "res_bias"       = bn_bias - bn_mean * (bn_scale / sqrt(bn_var + eps))
//   "res_filter_max" = expand( max( abs( res_filter ) ) )
// The corresponding source-pattern tensors must already be declared:
//   filter, bn_mean, bn_var, bn_scale, bn_bias  (and pat.Attr("epsilon"))
inline void BuildBnFoldSubgraph(paddle::drr::SourcePattern* pat,
                                paddle::drr::ResultPattern* res) {
  const auto bn_var_shape_attr = MakeBnVarShapeAttr(res);
  const auto scale_shape_attr = MakeScaleShapeAttr(res);
  const auto expand_1_shape = MakeExpand1ShapeAttr(res);

  // new_scale = bn_scale / sqrt(bn_var + epsilon)
  const auto& full_eps = res->Op(paddle::dialect::FullOp::name(),
                                 {{"shape", bn_var_shape_attr},
                                  {"value", pat->Attr("epsilon")},
                                  {"dtype", res->DataTypeAttr("float32")},
                                  {"place", res->PlaceAttr("cpu")}});
  const auto& add = res->Op(paddle::dialect::AddOp::name());
  res->Tensor("var_add_out") = add(res->Tensor("bn_var"), full_eps());
  const auto& sqrt = res->Op(paddle::dialect::SqrtOp::name());
  res->Tensor("sqrt_out") = sqrt(res->Tensor("var_add_out"));
  const auto& div = res->Op(paddle::dialect::DivideOp::name());
  res->Tensor("new_scale") =
      div(res->Tensor("bn_scale"), res->Tensor("sqrt_out"));
  const auto& reshape_scale = res->Op(paddle::dialect::ReshapeOp::name(),
                                      {{"shape", scale_shape_attr}});
  res->Tensor("res_scale") = reshape_scale(res->Tensor("new_scale"));

  // res_filter = filter * res_scale
  const auto& mul_filter = res->Op(paddle::dialect::MultiplyOp::name());
  res->Tensor("res_filter") =
      mul_filter(res->Tensor("filter"), res->Tensor("res_scale"));

  // res_bias = bn_bias - bn_mean * new_scale
  const auto& mul_mean = res->Op(paddle::dialect::MultiplyOp::name());
  res->Tensor("bn_mean_mul_out") =
      mul_mean(res->Tensor("bn_mean"), res->Tensor("new_scale"));
  const auto& sub_bias = res->Op(paddle::dialect::SubtractOp::name());
  res->Tensor("res_bias") =
      sub_bias(res->Tensor("bn_bias"), res->Tensor("bn_mean_mul_out"));

  // filter_max = expand( max( abs( res_filter ) ) )
  const auto& abs_op = res->Op(paddle::dialect::AbsOp::name());
  const auto& max_op =
      res->Op(paddle::dialect::MaxOp::name(),
              {{"axis", res->VectorInt64Attr(std::vector<int64_t>{})},
               {"keepdim", res->BoolAttr(false)}});
  const auto& expand_op =
      res->Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
  res->Tensor("res_filter_abs") = abs_op(res->Tensor("res_filter"));
  res->Tensor("filter_max") = max_op(res->Tensor("res_filter_abs"));
  res->Tensor("res_filter_max") = expand_op(res->Tensor("filter_max"));
}

// Build filter_max subgraph for the no-BN case:
//   "filter_max_expanded" = expand( max( abs( filter ) ) )
inline void BuildFilterMaxSubgraph(paddle::drr::ResultPattern* res) {
  const auto expand_1_shape = MakeExpand1ShapeAttr(res);
  const auto& abs_op = res->Op(paddle::dialect::AbsOp::name());
  const auto& max_op =
      res->Op(paddle::dialect::MaxOp::name(),
              {{"axis", res->VectorInt64Attr(std::vector<int64_t>{})},
               {"keepdim", res->BoolAttr(false)}});
  const auto& expand_op =
      res->Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
  res->Tensor("filter_abs") = abs_op(res->Tensor("filter"));
  res->Tensor("filter_max_scalar") = max_op(res->Tensor("filter_abs"));
  res->Tensor("filter_max_expanded") =
      expand_op(res->Tensor("filter_max_scalar"));
}

// ============================================================================
// Constraint helper for conv-bn patterns
// ============================================================================
inline bool CheckConvBnConstraints(const paddle::drr::MatchContext& m) {
  std::vector<int64_t> conv_input_shape =
      pir::GetShapeFromValue(m.Tensor("input"));
  if (conv_input_shape.size() != 4) return false;

  if (!pir::ValueIsPersistable(m.Tensor("bn_mean")) ||
      !pir::ValueIsPersistable(m.Tensor("bn_var")) ||
      !pir::ValueIsPersistable(m.Tensor("bn_scale")) ||
      !pir::ValueIsPersistable(m.Tensor("bn_bias"))) {
    return false;
  }

  auto paddings_size = m.Attr<std::vector<int>>("paddings");
  if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
    return false;
  }

  auto bn_bias_shape = pir::GetShapeFromValue(m.Tensor("bn_bias"));
  auto filter_shape = pir::GetShapeFromValue(m.Tensor("filter"));
  if (bn_bias_shape.at(0) != filter_shape.at(0)) return false;

  return true;
}

// ============================================================================
// Pattern 1: bare DepthwiseConv2d -> conv2d_xpu(LINEAR)
// ============================================================================
class Conv2dDepthwiseOnlyXpuFusePattern : public paddle::drr::DrrPatternBase {
 public:
  Conv2dDepthwiseOnlyXpuFusePattern() = default;

  std::string name() const override {
    return "Conv2dDepthwiseOnlyXpuFusePattern";
  }
  uint32_t benefit() const override { return 1; }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& dw_conv =
        pat.Op(paddle::dialect::DepthwiseConv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    dw_conv({&pat.Tensor("input"), &pat.Tensor("filter")},
            {&pat.Tensor("dw_out")});

    pat.AddConstraint([](const paddle::drr::MatchContext& m) {
      auto conv_input_shape = pir::GetShapeFromValue(m.Tensor("input"));
      auto paddings_size = m.Attr<std::vector<int>>("paddings");
      if (conv_input_shape.size() != 4) return false;
      if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
        return false;
      }
      if (!pir::ValueIsPersistable(m.Tensor("filter"))) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    BuildFilterMaxSubgraph(&res);

    const auto paddings_attr = MakePaddingsAttr(&res);
    const auto out_dtype_attr = MakeOutDtypeAttr(&res);

    const auto& conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type",
                    res.Int32Attr(static_cast<int>(xpu::Activation_t::LINEAR))},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("filter"),
            &res.Tensor("filter_max_expanded"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("dw_out"), &res.Tensor("out_max")});
  }
};

// ============================================================================
// Pattern 2: Conv2d/Depthwise + BN -> conv2d_xpu(LINEAR, with_bias)
// ============================================================================
class Conv2dBnXpuFusePattern : public paddle::drr::DrrPatternBase {
  bool is_depthwise_;
  bool bn_inplace_;

 public:
  Conv2dBnXpuFusePattern(bool is_depthwise, bool bn_inplace)
      : is_depthwise_(is_depthwise), bn_inplace_(bn_inplace) {}

  std::string name() const override {
    std::string s = "Conv2dBnXpuFusePattern_";
    s += (is_depthwise_ ? "depthwise_" : "conv_");
    s += (bn_inplace_ ? "bn_inplace" : "bn");
    return s;
  }
  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& conv =
        pat.Op(is_depthwise_ ? paddle::dialect::DepthwiseConv2dOp::name()
                             : paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto& bn = pat.Op(bn_inplace_ ? paddle::dialect::BatchNorm_Op::name()
                                        : paddle::dialect::BatchNormOp::name(),
                            {{"epsilon", pat.Attr("epsilon")}});

    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv2d_out")});
    bn({&pat.Tensor("conv2d_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});

    pat.AddConstraint(CheckConvBnConstraints);

    paddle::drr::ResultPattern res = pat.ResultPattern();
    BuildBnFoldSubgraph(&pat, &res);

    const auto paddings_attr = MakePaddingsAttr(&res);
    const auto out_dtype_attr = MakeOutDtypeAttr(&res);

    const auto& conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type",
                    res.Int32Attr(static_cast<int>(xpu::Activation_t::LINEAR))},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("res_filter"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("res_bias"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("bn_out"), &res.Tensor("out_max")});
  }
};

// ============================================================================
// Pattern 3: Conv2d/Depthwise + BN + Act -> conv2d_xpu(act, with_bias)
// ============================================================================
class Conv2dBnActXpuFusePattern : public paddle::drr::DrrPatternBase {
  bool is_depthwise_;
  bool bn_inplace_;
  std::string act_op_name_;
  int act_type_;

 public:
  Conv2dBnActXpuFusePattern(bool is_depthwise,
                            bool bn_inplace,
                            std::string act_op_name,
                            int act_type)
      : is_depthwise_(is_depthwise),
        bn_inplace_(bn_inplace),
        act_op_name_(std::move(act_op_name)),
        act_type_(act_type) {}

  std::string name() const override {
    std::string s = "Conv2dBnActXpuFusePattern_";
    s += (is_depthwise_ ? "depthwise_" : "conv_");
    s += (bn_inplace_ ? "bn_inplace_" : "bn_");
    auto pos = act_op_name_.rfind('.');
    s += (pos == std::string::npos) ? act_op_name_
                                    : act_op_name_.substr(pos + 1);
    return s;
  }
  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& conv =
        pat.Op(is_depthwise_ ? paddle::dialect::DepthwiseConv2dOp::name()
                             : paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto& bn = pat.Op(bn_inplace_ ? paddle::dialect::BatchNorm_Op::name()
                                        : paddle::dialect::BatchNormOp::name(),
                            {{"epsilon", pat.Attr("epsilon")}});
    const auto& act = pat.Op(act_op_name_);

    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv2d_out")});
    bn({&pat.Tensor("conv2d_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    act({&pat.Tensor("bn_out")}, {&pat.Tensor("act_out")});

    pat.AddConstraint(CheckConvBnConstraints);

    paddle::drr::ResultPattern res = pat.ResultPattern();
    BuildBnFoldSubgraph(&pat, &res);

    const auto paddings_attr = MakePaddingsAttr(&res);
    const auto out_dtype_attr = MakeOutDtypeAttr(&res);

    const auto& conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type", res.Int32Attr(act_type_)},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("res_filter"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("res_bias"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("act_out"), &res.Tensor("out_max")});
  }
};

// ============================================================================
// Pattern 4: Conv2d/Depthwise + BN + Add(branch) + Act
//             -> conv2d_xpu(act, with_bias, with_branch)
// ============================================================================
class Conv2dBnAddActXpuFusePattern : public paddle::drr::DrrPatternBase {
  bool is_depthwise_;
  bool bn_inplace_;
  bool residual_first_;
  std::string act_op_name_;
  int act_type_;

 public:
  Conv2dBnAddActXpuFusePattern(bool is_depthwise,
                               bool bn_inplace,
                               bool residual_first,
                               std::string act_op_name,
                               int act_type)
      : is_depthwise_(is_depthwise),
        bn_inplace_(bn_inplace),
        residual_first_(residual_first),
        act_op_name_(std::move(act_op_name)),
        act_type_(act_type) {}

  std::string name() const override {
    std::string s = "Conv2dBnAddActXpuFusePattern_";
    s += (is_depthwise_ ? "depthwise_" : "conv_");
    s += (bn_inplace_ ? "bn_inplace_" : "bn_");
    s += (residual_first_ ? "resfirst_" : "resafter_");
    auto pos = act_op_name_.rfind('.');
    s += (pos == std::string::npos) ? act_op_name_
                                    : act_op_name_.substr(pos + 1);
    return s;
  }
  uint32_t benefit() const override { return 4; }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& conv =
        pat.Op(is_depthwise_ ? paddle::dialect::DepthwiseConv2dOp::name()
                             : paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto& bn = pat.Op(bn_inplace_ ? paddle::dialect::BatchNorm_Op::name()
                                        : paddle::dialect::BatchNormOp::name(),
                            {{"epsilon", pat.Attr("epsilon")}});
    const auto& add = pat.Op(paddle::dialect::AddOp::name());
    const auto& act = pat.Op(act_op_name_);

    conv({&pat.Tensor("input"), &pat.Tensor("filter")},
         {&pat.Tensor("conv2d_out")});
    bn({&pat.Tensor("conv2d_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});
    if (residual_first_) {
      add({&pat.Tensor("residual"), &pat.Tensor("bn_out")},
          {&pat.Tensor("add_out")});
    } else {
      add({&pat.Tensor("bn_out"), &pat.Tensor("residual")},
          {&pat.Tensor("add_out")});
    }
    act({&pat.Tensor("add_out")}, {&pat.Tensor("act_out")});

    pat.AddConstraint([](const paddle::drr::MatchContext& m) {
      if (!CheckConvBnConstraints(m)) return false;
      auto bn_out_shape = pir::GetShapeFromValue(m.Tensor("bn_out"));
      auto residual_shape = pir::GetShapeFromValue(m.Tensor("residual"));
      if (bn_out_shape.size() != residual_shape.size()) return false;
      for (size_t i = 0; i < bn_out_shape.size(); ++i) {
        if (bn_out_shape[i] != residual_shape[i]) return false;
      }
      if (m.Tensor("residual") == m.Tensor("conv2d_out")) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    BuildBnFoldSubgraph(&pat, &res);

    const auto paddings_attr = MakePaddingsAttr(&res);
    const auto out_dtype_attr = MakeOutDtypeAttr(&res);

    const auto& conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type", res.Int32Attr(act_type_)},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("res_filter"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("res_bias"),
            &res.Tensor("residual"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("act_out"), &res.Tensor("out_max")});
  }
};

// ============================================================================
// Pass class
// ============================================================================
class Conv2dXpuFusePass : public pir::PatternRewritePass {
 public:
  Conv2dXpuFusePass() : pir::PatternRewritePass("conv2d_xpu_fuse_pass", 2) {}

  pir::GreedyRewriteConfig InitializeConfig() override {
    // Use bottom-up traversal so that patterns anchored on later ops (e.g.
    // `relu` for Conv+BN+Act and Conv+BN+Add+Act) are tried before patterns
    // anchored on earlier ops (`bn` for Conv+BN). Each DRR pattern's anchor
    // is its sole source-pattern output op; without bottom-up traversal, the
    // smaller Conv+BN pattern would always match at `bn` first and prevent
    // the longer activation patterns from ever firing on the same subgraph.
    pir::GreedyRewriteConfig config;
    config.use_top_down_traversal = false;
    config.max_iterations = 10;
    return config;
  }

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    // Bit mask to selectively enable patterns; useful for debugging.
    //   bit 0 (0x01): bare DepthwiseConv2d -> conv2d_xpu(LINEAR)
    //   bit 1 (0x02): conv + bn
    //   bit 2 (0x04): conv + bn + act
    //   bit 3 (0x08): conv + bn + add(branch) + act
    int mask = 0xff;
    if (const char* v = std::getenv("XPU_PADDLE_CONV2D_PATTERN")) {
      char* endptr = nullptr;
      auto val = std::strtol(v, &endptr, 16);
      if (endptr != v && *endptr == '\0') {
        mask = static_cast<int>(val);
      } else {
        LOG(WARNING) << "Invalid XPU_PADDLE_CONV2D_PATTERN: " << v;
      }
    }

    const std::vector<std::pair<std::string, int>> acts = {
        {paddle::dialect::ReluOp::name(),
         static_cast<int>(xpu::Activation_t::RELU)},
        {paddle::dialect::SwishOp::name(),
         static_cast<int>(xpu::Activation_t::SWISH)},
        {paddle::dialect::HardswishOp::name(),
         static_cast<int>(xpu::Activation_t::HARD_SWISH)},
    };

    if (mask & 0x08) {
      for (bool is_depthwise : {false, true}) {
        for (bool bn_inplace : {true, false}) {
          for (bool residual_first : {false, true}) {
            for (const auto& act : acts) {
              ps.Add(paddle::drr::Create<Conv2dBnAddActXpuFusePattern>(
                  context,
                  is_depthwise,
                  bn_inplace,
                  residual_first,
                  act.first,
                  act.second));
            }
          }
        }
      }
    }

    if (mask & 0x04) {
      for (bool is_depthwise : {false, true}) {
        for (bool bn_inplace : {true, false}) {
          for (const auto& act : acts) {
            ps.Add(paddle::drr::Create<Conv2dBnActXpuFusePattern>(
                context, is_depthwise, bn_inplace, act.first, act.second));
          }
        }
      }
    }

    if (mask & 0x02) {
      for (bool is_depthwise : {false, true}) {
        for (bool bn_inplace : {true, false}) {
          ps.Add(paddle::drr::Create<Conv2dBnXpuFusePattern>(
              context, is_depthwise, bn_inplace));
        }
      }
    }

    if (mask & 0x01) {
      ps.Add(paddle::drr::Create<Conv2dDepthwiseOnlyXpuFusePattern>(context));
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dXpuFusePass() {
  return std::make_unique<Conv2dXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_xpu_fuse_pass, Conv2dXpuFusePass);
