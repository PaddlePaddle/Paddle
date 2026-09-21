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

#include "paddle/fluid/pir/transforms/xpu/qkv_attention_xpu_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// Fuse the SVTR / EncoderWithSVTR self-attention sub-graph
//
//   reshape -> transpose(perm=[2,0,3,1,4]) -> { slice_q, slice_k, slice_v }
//   slice_q -> scale -> matmul1.lhs
//   slice_k -> transpose(perm=[0,1,3,2]) -> matmul1.rhs
//   matmul1 -> softmax -> matmul2.lhs
//   slice_v -> matmul2.rhs
//   matmul2 -> transpose(perm=[0,2,1,3]) -> reshape -> out
//
// into a single ``qkv_attention_xpu`` op with ``qkv_fc_fusion=true`` so that
// q == k == v == ``qkv_input``. The XDNN kernel handles the internal split.
//
// Op signature (from paddle/phi/ops/yaml/fused_ops.yaml):
//   qkv_attention_xpu(q, k, v, q_max, k_max, v_max, qk_max, qkv_max,
//                     alpha, head_num, head_dim, qkv_fc_fusion, out_dtype)
class QkvAttentionXpuFusePattern : public paddle::drr::DrrPatternBase {
 public:
  QkvAttentionXpuFusePattern() = default;

  std::string name() const override { return "QkvAttentionXpuFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    // ---- pre-attention reshape + transpose -------------------------------
    const auto &reshape_pre = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &transpose_pre = pat.Op(paddle::dialect::TransposeOp::name(),
                                       {{"perm", pat.Attr("perm_pre")}});

    pat.Tensor("qkv_5d") =
        reshape_pre(pat.Tensor("qkv_input"), pat.Tensor("shape_5d"));
    pat.Tensor("qkv_t") = transpose_pre(pat.Tensor("qkv_5d"));

    // ---- three slices to extract Q / K / V -------------------------------
    const auto &slice_q = pat.Op(paddle::dialect::SliceOp::name(),
                                 {{"axes", pat.Attr("axes_q")},
                                  {"infer_flags", pat.Attr("infer_flags_q")},
                                  {"decrease_axis", pat.Attr("decrease_q")}});
    const auto &slice_k = pat.Op(paddle::dialect::SliceOp::name(),
                                 {{"axes", pat.Attr("axes_k")},
                                  {"infer_flags", pat.Attr("infer_flags_k")},
                                  {"decrease_axis", pat.Attr("decrease_k")}});
    const auto &slice_v = pat.Op(paddle::dialect::SliceOp::name(),
                                 {{"axes", pat.Attr("axes_v")},
                                  {"infer_flags", pat.Attr("infer_flags_v")},
                                  {"decrease_axis", pat.Attr("decrease_v")}});

    pat.Tensor("Q") = slice_q(
        pat.Tensor("qkv_t"), pat.Tensor("q_starts"), pat.Tensor("q_ends"));
    pat.Tensor("K") = slice_k(
        pat.Tensor("qkv_t"), pat.Tensor("k_starts"), pat.Tensor("k_ends"));
    pat.Tensor("V") = slice_v(
        pat.Tensor("qkv_t"), pat.Tensor("v_starts"), pat.Tensor("v_ends"));

    // ---- Q * scale -------------------------------------------------------
    const auto &full_alpha =
        pat.Op(paddle::dialect::FullOp::name(), {{"value", pat.Attr("alpha")}});
    pat.Tensor("scale_val") = full_alpha();
    const auto &scale_op =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    pat.Tensor("Q_scaled") = scale_op(pat.Tensor("Q"), pat.Tensor("scale_val"));

    // ---- K^T -------------------------------------------------------------
    const auto &transpose_k = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_k")}});
    pat.Tensor("K_T") = transpose_k(pat.Tensor("K"));

    // ---- QK^T, softmax, attn @ V ----------------------------------------
    const auto &matmul_qk = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("qk_tx")},
                                    {"transpose_y", pat.Attr("qk_ty")}});
    pat.Tensor("QK") = matmul_qk(pat.Tensor("Q_scaled"), pat.Tensor("K_T"));

    const auto &softmax_op = pat.Op(paddle::dialect::SoftmaxOp::name(),
                                    {{"axis", pat.Attr("softmax_axis")}});
    pat.Tensor("attn") = softmax_op(pat.Tensor("QK"));

    const auto &matmul_av = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("av_tx")},
                                    {"transpose_y", pat.Attr("av_ty")}});
    pat.Tensor("attn_v") = matmul_av(pat.Tensor("attn"), pat.Tensor("V"));

    // ---- post-attention transpose + reshape ------------------------------
    const auto &transpose_post = pat.Op(paddle::dialect::TransposeOp::name(),
                                        {{"perm", pat.Attr("perm_post")}});
    pat.Tensor("attn_t") = transpose_post(pat.Tensor("attn_v"));

    const auto &reshape_post = pat.Op(paddle::dialect::ReshapeOp::name());
    pat.Tensor("attn_out") =
        reshape_post(pat.Tensor("attn_t"), pat.Tensor("shape_3d"));

    // ---------------- constraints ----------------------------------------
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      // perm checks
      auto perm_pre = match_ctx.Attr<std::vector<int>>("perm_pre");
      if (perm_pre != std::vector<int>{2, 0, 3, 1, 4}) return false;
      auto perm_k = match_ctx.Attr<std::vector<int>>("perm_k");
      if (perm_k != std::vector<int>{0, 1, 3, 2}) return false;
      auto perm_post = match_ctx.Attr<std::vector<int>>("perm_post");
      if (perm_post != std::vector<int>{0, 2, 1, 3}) return false;

      // slice axes must all be [0]
      auto axes_q = match_ctx.Attr<std::vector<int64_t>>("axes_q");
      auto axes_k = match_ctx.Attr<std::vector<int64_t>>("axes_k");
      auto axes_v = match_ctx.Attr<std::vector<int64_t>>("axes_v");
      if (axes_q != std::vector<int64_t>{0} ||
          axes_k != std::vector<int64_t>{0} ||
          axes_v != std::vector<int64_t>{0}) {
        return false;
      }

      // matmul transpose flags must all be false
      if (match_ctx.Attr<bool>("qk_tx") || match_ctx.Attr<bool>("qk_ty") ||
          match_ctx.Attr<bool>("av_tx") || match_ctx.Attr<bool>("av_ty")) {
        return false;
      }

      // softmax axis must be -1 (over key dim)
      if (match_ctx.Attr<int>("softmax_axis") != -1) return false;

      // scale bias must be 0
      if (std::abs(match_ctx.Attr<float>("scale_bias")) > 1e-6f) return false;

      // qkv_t shape must be 5D: (3, B, head_num, N, head_dim)
      auto qkv_t_shape = pir::GetShapeFromValue(match_ctx.Tensor("qkv_t"));
      if (qkv_t_shape.size() != 5) return false;
      if (qkv_t_shape[0] != 3) return false;

      // Q must be (B, head_num, N, head_dim) with static head dims
      auto q_shape = pir::GetShapeFromValue(match_ctx.Tensor("Q"));
      if (q_shape.size() != 4) return false;
      if (q_shape[1] <= 0 || q_shape[3] <= 0) return false;

      return true;
    });

    // ---------------- result pattern -------------------------------------
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &head_num_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto q_shape = pir::GetShapeFromValue(match_ctx.Tensor("Q"));
          return static_cast<int>(q_shape[1]);
        });
    const auto &head_dim_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto q_shape = pir::GetShapeFromValue(match_ctx.Tensor("Q"));
          return static_cast<int>(q_shape[3]);
        });
    const auto &alpha_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          // ``value`` of pd_op.full is a Scalar exposed as double in DRR
          // (matches every other DRR pass that reads pd_op.full's value).
          // Cast through double to avoid bad_any_cast.
          return static_cast<float>(match_ctx.Attr<double>("alpha"));
        });
    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype =
              pir::GetDataTypeFromValue(match_ctx.Tensor("qkv_input"));
          if (x_dtype.isa<pir::Float16Type>()) return phi::DataType::FLOAT16;
          return phi::DataType::FLOAT32;
        });

    const auto &qkv_attention =
        res.Op(paddle::dialect::QkvAttentionXpuOp::name(),
               {{
                   {"alpha", alpha_attr},
                   {"head_num", head_num_attr},
                   {"head_dim", head_dim_attr},
                   {"qkv_fc_fusion", res.BoolAttr(true)},
                   {"out_dtype", out_dtype_attr},
               }});

    qkv_attention(
        {
            &res.Tensor("qkv_input"),  // q
            &res.Tensor("qkv_input"),  // k
            &res.Tensor("qkv_input"),  // v
            &res.InputNoneTensor(),    // q_max
            &res.InputNoneTensor(),    // k_max
            &res.InputNoneTensor(),    // v_max
            &res.InputNoneTensor(),    // qk_max
            &res.InputNoneTensor(),    // qkv_max
        },
        {&res.Tensor("attn_out")});
  }
};

class QkvAttentionXpuFusePass : public pir::PatternRewritePass {
 public:
  QkvAttentionXpuFusePass()
      : pir::PatternRewritePass("qkv_attention_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<QkvAttentionXpuFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateQkvAttentionXpuFusePass() {
  return std::make_unique<QkvAttentionXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(qkv_attention_xpu_fuse_pass, QkvAttentionXpuFusePass);
