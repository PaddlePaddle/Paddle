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

// Eliminate the channel-axis concat that feeds a conv2d_xpu, WITHOUT
// introducing a new fused op: rewrite
//
//   combine(x0..xn) -> concat(axis=1) -> conv2d_xpu(W[F,sumC,1,1], act, bias)
//
// into N existing conv2d_xpu ops chained via the `branch` accumulator:
//
//   x0 -> conv2d_xpu(W0, branch=null, act=LINEAR)             -> (o0, m0)
//   x1 -> conv2d_xpu(W1, branch=o0, branch_max=m0, LINEAR)    -> (o1, m1)
//   ...
//   xn -> conv2d_xpu(Wn, branch=o{n-1}, act=real_act, bias=real_bias) -> (on,
//   mn)
//
// where Wi = split(W, sections)[i] (constant-folded at build time, zero runtime
// cost). Math is exactly equivalent to conv on the concatenated tensor:
//   conv(concat[X0..Xn]) = sum_i conv(Xi, Wi).
// The chain of branch-accumulating conv2d_xpu ops realizes that sum without an
// explicit concat, and without any new kernel/op.
//
// This is an imperative OpRewritePattern (not DRR), because building
// pd_op.split with a dynamically-computed sections IntArray is not supported in
// the DRR ResultPattern.
//
// Constraints: concat axis == 1, conv2d_xpu groups == 1, 1x1 conv
// (ksize=[1,1], pad=0, stride=1, dilation=1), filter persistable, N in [2,7].
// Must run AFTER conv2d_xpu_fuse_pass (so the downstream conv is already
// conv2d_xpu).

#include "paddle/fluid/pir/transforms/xpu/concat_conv_xpu_fuse_pass.h"

#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/backends/xpu/xpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

using paddle::dialect::DataTypeAttribute;
using paddle::dialect::IntArrayAttribute;

namespace {

// Read a scalar int value out of a tensor produced by pd_op.full.
// The `value` attribute may be stored as DoubleAttribute (float dtype) or
// IntAttribute (int dtype), depending on the FullOp's dtype.
bool ReadFullIntScalar(pir::Value v, int* out) {
  if (!v) return false;
  pir::Operation* def = v.defining_op();
  if (def == nullptr) return false;
  if (def->name() != paddle::dialect::FullOp::name()) return false;
  auto attr = def->attribute("value");
  if (!attr) return false;
  if (auto d = attr.dyn_cast<pir::DoubleAttribute>()) {
    *out = static_cast<int>(d.data());
    return true;
  }
  if (auto i = attr.dyn_cast<pir::Int32Attribute>()) {
    *out = static_cast<int>(i.data());
    return true;
  }
  if (auto i = attr.dyn_cast<pir::Int64Attribute>()) {
    *out = static_cast<int>(i.data());
    return true;
  }
  return false;
}

// Imperative rewrite anchored on pd_op.conv2d_xpu.
class ConcatConvXpuFusePattern
    : public pir::OpRewritePattern<paddle::dialect::Conv2dXpuOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Conv2dXpuOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::Conv2dXpuOp conv_op,
                       pir::PatternRewriter& rewriter) const override {
    // 1. The conv's input (operand 0) must come from concat.
    pir::Value concat_in_value = conv_op.operand_source(0);
    if (!concat_in_value) return false;
    pir::Operation* concat_def = concat_in_value.defining_op();
    if (concat_def == nullptr) return false;
    if (concat_def->name() != paddle::dialect::ConcatOp::name()) return false;
    paddle::dialect::ConcatOp concat_op =
        concat_def->dyn_cast<paddle::dialect::ConcatOp>();
    if (!concat_op) return false;

    // 2. concat axis (operand 1) must be the constant 1.
    int axis = 0;
    if (!ReadFullIntScalar(concat_op.operand_source(1), &axis) || axis != 1) {
      return false;
    }

    // 3. concat's input must come from builtin.combine.
    pir::Value combine_value = concat_op.operand_source(0);
    if (!combine_value) return false;
    pir::Operation* combine_def = combine_value.defining_op();
    if (combine_def == nullptr) return false;
    if (!combine_def->isa<pir::CombineOp>()) return false;
    pir::CombineOp combine_op = combine_def->dyn_cast<pir::CombineOp>();

    // Gather the N branch inputs.
    std::vector<pir::Value> branch_inputs;
    for (uint32_t i = 0; i < combine_op.num_operands(); ++i) {
      branch_inputs.push_back(combine_op.operand_source(i));
    }
    size_t n = branch_inputs.size();
    if (n < 2 || n > 7) return false;  // det/rec channel-concat have 2..7.

    // 4. Constraints on the conv2d_xpu.
    int groups = conv_op->attribute<pir::Int32Attribute>("groups").data();
    if (groups != 1) return false;

    std::vector<int> paddings = ReadIntVectorAttr(conv_op, "paddings");
    for (auto p : paddings) {
      if (p != 0) return false;
    }
    std::vector<int> strides = ReadIntVectorAttr(conv_op, "strides");
    for (auto s : strides) {
      if (s != 1) return false;
    }
    std::vector<int> dilations = ReadIntVectorAttr(conv_op, "dilations");
    for (auto d : dilations) {
      if (d != 1) return false;
    }

    // 5. filter must be 1x1 and persistable.
    pir::Value filter = conv_op.operand_source(2);
    if (!pir::ValueIsPersistable(filter)) return false;
    auto filter_shape = pir::GetShapeFromValue(filter);
    if (filter_shape.size() != 4) return false;
    if (filter_shape[2] != 1 || filter_shape[3] != 1) return false;

    // All branch inputs must be 4-D NCHW.
    std::vector<int64_t> sections;
    sections.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto s = pir::GetShapeFromValue(branch_inputs[i]);
      if (s.size() != 4) return false;
      sections.push_back(s[1]);
    }

    // ---- Rewrite ----
    rewriter.SetInsertionPointAfter(conv_op);

    // Build pd_op.split on the constant filter along axis=1 with `sections`.
    pir::Operation* split_op =
        rewriter.Build<paddle::dialect::SplitOp>(filter, sections, /*axis=*/1);
    // split output is a VectorType (Tensor[]); unwrap via builtin.split to get
    // the N individual slice values.
    pir::Operation* split_unwrap =
        rewriter.Build<pir::SplitOp>(split_op->result(0));
    std::vector<pir::Value> filter_slices;
    for (uint32_t i = 0; i < split_unwrap->num_results(); ++i) {
      filter_slices.push_back(split_unwrap->result(i));
    }

    // Propagate shared attrs from the original conv2d_xpu.
    std::vector<int> paddings_attr = ReadIntVectorAttr(conv_op, "paddings");
    std::vector<int> dilations_attr = ReadIntVectorAttr(conv_op, "dilations");
    std::vector<int> strides_attr = ReadIntVectorAttr(conv_op, "strides");
    std::string padding_algorithm =
        conv_op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
    int real_act_type =
        conv_op->attribute<pir::Int32Attribute>("act_type").data();
    float real_act_param =
        conv_op->attribute<pir::FloatAttribute>("act_param").data();
    phi::DataType out_dtype =
        conv_op->attribute<DataTypeAttribute>("out_dtype").data();

    // Shared operands.
    pir::Value filter_max =
        conv_op.operand_source(3);                     // filter_max (operand 3)
    pir::Value real_bias = conv_op.operand_source(4);  // bias (operand 4, opt)
    pir::Value null_v;                                 // null placeholder
    // xpu::Activation_t::LINEAR == 0 (no-op activation). Intermediate
    // segments must use LINEAR so only the final segment applies the real
    // activation; otherwise the activation would be applied N times.
    const int linear_act = 0;

    // Chain N conv2d_xpu ops: segment i accumulates via branch = prev out.
    pir::Value prev_out;
    pir::Value prev_out_max;
    for (size_t i = 0; i < n; ++i) {
      bool is_last = (i == n - 1);
      pir::Operation* seg = rewriter.Build<paddle::dialect::Conv2dXpuOp>(
          branch_inputs[i],  // x
          null_v,            // x_max
          filter_slices[i],  // filter (the i-th slice)
          filter_max,        // filter_max (reused: per-F, F dim unchanged)
          is_last ? real_bias : null_v,      // bias only on last segment
          (i == 0) ? null_v : prev_out,      // branch: first writes, rest add
          (i == 0) ? null_v : prev_out_max,  // branch_max
          null_v,                            // scale_max
          null_v,                            // out_max_in
          paddings_attr,
          dilations_attr,
          strides_attr,
          padding_algorithm,
          groups,
          is_last ? real_act_type : linear_act,  // act only on last segment
          is_last ? real_act_param : 0.0f,
          out_dtype);
      prev_out = seg->result(0);
      prev_out_max = seg->result(1);
    }

    // Replace conv2d_xpu's two results (out, out_max) with the chain's final.
    rewriter.ReplaceAllUsesWith(conv_op->result(0), prev_out);
    rewriter.ReplaceAllUsesWith(conv_op->result(1), prev_out_max);
    rewriter.EraseOp(conv_op);
    // concat/combine are now dead; leave to DCE pass if still referenced.
    if (concat_op->use_empty()) {
      rewriter.EraseOp(concat_def);
    }
    if (combine_op->use_empty()) {
      rewriter.EraseOp(combine_def);
    }
    return true;
  }

 private:
  static std::vector<int> ReadIntVectorAttr(pir::Operation* op,
                                            const std::string& name) {
    auto attr = op->attribute(name);
    if (!attr) return {};
    // conv2d_xpu stores paddings/strides/dilations as pir::ArrayAttribute of
    // Int32Attribute (see generated Build).
    if (auto arr = attr.dyn_cast<pir::ArrayAttribute>()) {
      std::vector<int> out;
      out.reserve(arr.size());
      for (size_t i = 0; i < arr.size(); ++i) {
        out.push_back(arr.at(i).dyn_cast<pir::Int32Attribute>().data());
      }
      return out;
    }
    if (auto ia = attr.dyn_cast<IntArrayAttribute>()) {
      std::vector<int> out;
      for (auto v : ia.data().GetData()) {
        out.push_back(static_cast<int>(v));
      }
      return out;
    }
    return {};
  }
  static std::vector<int> ReadIntVectorAttr(
      paddle::dialect::Conv2dXpuOp op,  // NOLINT runtime references
      const std::string& name) {
    return ReadIntVectorAttr(op.operation(), name);
  }
};

class ConcatConvXpuFusePass : public pir::PatternRewritePass {
 public:
  ConcatConvXpuFusePass()
      : pir::PatternRewritePass("concat_conv_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ConcatConvXpuFusePattern>(context);
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConcatConvXpuFusePass() {
  return std::make_unique<ConcatConvXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(concat_conv_xpu_fuse_pass, ConcatConvXpuFusePass);
