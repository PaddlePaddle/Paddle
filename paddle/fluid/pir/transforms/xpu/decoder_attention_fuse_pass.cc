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

#include "paddle/fluid/pir/transforms/xpu/decoder_attention_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DecoderAttentionPattern : public paddle::drr::DrrPatternBase {
  /*
  Origin subgraph:

                    v          q           k
                    |          |           |
                    |          |           |
                    |          |           |
                 reshape    reshape     reshape
                    |          |           |
                    |          |           |
                    |          |           |
                transpose  transpose   transpose
                    |          |           |
                    |           \         /
                    |             \     /
                    |            qk_matmul
                    |                |
                    |                |
                    |                |
                    |              scale
                    |                |
                    |                |
                    |                |
                     \            qk_softmax
                       \             |
                         \          /
                           \      /
                             qkv_matmul
                                |
                                |
                                |
                            transpose
                                |
                                |
                                |
                             reshape
                                |
                                |
                                |
                              output

  -------------------------------------------------------
  Fused subgraph:
                   q          k          v
                    \         |         /
                      \       |       /
                        \     |     /
                     qkv_attention_xpu
                             |
                             |
                             |
                           output
  */
 public:
  std::string name() const override { return "DecoderAttentionPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &reshape_q = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &reshape_k = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &reshape_v = pat.Op(paddle::dialect::ReshapeOp::name());

    const auto &full_int_array_q =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_int_array_k =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_int_array_v =
        pat.Op(paddle::dialect::FullIntArrayOp::name());

    pat.Tensor("reshape_q_out") =
        reshape_q(pat.Tensor("q"), full_int_array_q());
    pat.Tensor("reshape_k_out") =
        reshape_k(pat.Tensor("k"), full_int_array_k());
    pat.Tensor("reshape_v_out") =
        reshape_v(pat.Tensor("v"), full_int_array_v());

    const auto &transpose_q = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("transpose_q_perm")}});
    const auto &transpose_k = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("transpose_k_perm")}});
    const auto &transpose_v = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("transpose_v_perm")}});

    pat.Tensor("transpose_q_out") = transpose_q(pat.Tensor("reshape_q_out"));
    pat.Tensor("transpose_k_out") = transpose_k(pat.Tensor("reshape_k_out"));
    pat.Tensor("transpose_v_out") = transpose_v(pat.Tensor("reshape_v_out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &transpose_q_perm =
          match_ctx.Attr<std::vector<int32_t>>("transpose_q_perm");
      const auto &transpose_k_perm =
          match_ctx.Attr<std::vector<int32_t>>("transpose_k_perm");
      const auto &transpose_v_perm =
          match_ctx.Attr<std::vector<int32_t>>("transpose_v_perm");

      VLOG(0) << "calc constraints of transpose op";

      if (transpose_q_perm.size() != 4 || transpose_k_perm.size() != 4 ||
          transpose_v_perm.size() != 4) {
        return false;
      }
      if (transpose_q_perm[0] != 0 || transpose_q_perm[1] != 2 ||
          transpose_q_perm[2] != 1 || transpose_q_perm[3] != 3) {
        return false;
      }
      if (transpose_k_perm[0] != 0 || transpose_k_perm[1] != 2 ||
          transpose_k_perm[2] != 1 || transpose_k_perm[3] != 3) {
        return false;
      }
      if (transpose_v_perm[0] != 0 || transpose_v_perm[1] != 2 ||
          transpose_v_perm[2] != 1 || transpose_v_perm[3] != 3) {
        return false;
      }
      VLOG(0) << "constraints of transpose op are satisfied";
      return true;
    });

    const auto &qk_matmul =
        pat.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("qk_matmul_transpose_x")},
                {"transpose_y", pat.Attr("qk_matmul_transpose_y")}});

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"shape", pat.Attr("full_op_shape")},
                               {"value", pat.Attr("full_op_value")},
                               {"dtype", pat.Attr("full_op_dtype")},
                               {"place", pat.Attr("full_op_place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &softmax = pat.Op(paddle::dialect::SoftmaxOp::name());

    pat.Tensor("qk_matmul_out") =
        qk_matmul(pat.Tensor("transpose_q_out"), pat.Tensor("transpose_k_out"));
    pat.Tensor("scale_out") = scale(pat.Tensor("qk_matmul_out"), full());
    pat.Tensor("softmax_out") = softmax(pat.Tensor("scale_out"));

    const auto &qkv_softmax_matmul =
        pat.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("qkv_softmax_matmul_transpose_x")},
                {"transpose_y", pat.Attr("qkv_softmax_matmul_transpose_y")}});

    pat.Tensor("qkv_softmax_matmul_out") = qkv_softmax_matmul(
        pat.Tensor("softmax_out"), pat.Tensor("transpose_v_out"));

    const auto &transpose_qkv = pat.Op(paddle::dialect::TransposeOp::name());
    const auto &reshape_qkv = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &full_int_array_qkv =
        pat.Op(paddle::dialect::FullIntArrayOp::name());

    pat.Tensor("transpose_qkv_out") =
        transpose_qkv(pat.Tensor("qkv_softmax_matmul_out"));
    pat.Tensor("reshape_qkv_out") =
        reshape_qkv(pat.Tensor("transpose_qkv_out"), full_int_array_qkv());

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &qk_matmul_transpose_x =
          match_ctx.Attr<bool>("qk_matmul_transpose_x");
      const auto &qk_matmul_transpose_y =
          match_ctx.Attr<bool>("qk_matmul_transpose_y");

      const auto &qkv_softmax_matmul_transpose_x =
          match_ctx.Attr<bool>("qkv_softmax_matmul_transpose_x");
      const auto &qkv_softmax_matmul_transpose_y =
          match_ctx.Attr<bool>("qkv_softmax_matmul_transpose_y");

      if (qk_matmul_transpose_x != false || qk_matmul_transpose_y != true ||
          qkv_softmax_matmul_transpose_x != false ||
          qkv_softmax_matmul_transpose_y != false) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &head_num_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> int32_t {
          auto transpose_q_out_shape =
              ::pir::GetShapeFromValue(match_ctx.Tensor("transpose_q_out"));
          return static_cast<int32_t>(transpose_q_out_shape[1]);
        });
    const auto &head_dim_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> int32_t {
          auto transpose_q_out_shape =
              ::pir::GetShapeFromValue(match_ctx.Tensor("transpose_q_out"));
          return static_cast<int32_t>(transpose_q_out_shape[3]);
        });

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("q"));
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    const auto &alpha_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return static_cast<float>(match_ctx.Attr<double>("full_op_value"));
        });

    const auto &qkv_attention_xpu =
        res.Op(paddle::dialect::QkvAttentionXpuOp::name(),
               {{"alpha", alpha_attr},
                {"head_num", head_num_attr},
                {"head_dim", head_dim_attr},
                {"qkv_fc_fusion", res.BoolAttr(false)},
                {"out_dtype", out_dtype_attr}});

    qkv_attention_xpu(
        {
            &res.Tensor("q"),
            &res.Tensor("k"),
            &res.Tensor("v"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("reshape_qkv_out")});
  }
};

class DecoderAttentionXpuFusePass : public pir::PatternRewritePass {
 public:
  DecoderAttentionXpuFusePass()
      : pir::PatternRewritePass("decoder_attention_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DecoderAttentionPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateDecoderAttentionXpuFusePass() {
  return std::make_unique<DecoderAttentionXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(decoder_attention_xpu_fuse_pass, DecoderAttentionXpuFusePass);
