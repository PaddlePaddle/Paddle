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

#include "paddle/cinn/hlir/dialect/operator/transforms/convert_memory_effec_attn_to_flash_attn_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class ConvertMEA2FAPattern : public pir::OpRewritePattern<
                                 paddle::dialect::MemoryEfficientAttentionOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::MemoryEfficientAttentionOp>::OpRewritePattern;

  bool Match(paddle::dialect::MemoryEfficientAttentionOp op) const override {
#ifndef PADDLE_WITH_FLASHATTN
    return false;
#endif
    auto bias = op->operand_source(3);
    auto cu_seq_q = op->operand_source(4);
    auto cu_seq_k = op->operand_source(5);
    auto causal_diagonal = op->operand_source(6);
    auto seq_k = op->operand_source(7);

    // bias, cur_seq_q, cur_seq_k,  causal_diagonal, seq_k should be null
    if (bias || cu_seq_q || cu_seq_k || causal_diagonal || seq_k) {
      return false;
    }
    // flash attention does not support float32
    if (paddle::dialect::TransToPhiDataType(
            op->operand_source(0)
                .type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dtype()) == phi::DataType::FLOAT32) {
      return false;
    }

    bool is_test =
        op->attribute("is_test").dyn_cast<pir::BoolAttribute>().data();
    if (!is_test) {
      return false;
    }

    float scale = op->attribute("scale").dyn_cast<pir::FloatAttribute>().data();

    if (scale > 0) {
      auto hidden_size =
          phi::vectorize(op->operand_source(0)
                             .type()
                             .dyn_cast<paddle::dialect::DenseTensorType>()
                             .dims())
              .back();
      auto scale_t = 1.0 / std::sqrt(hidden_size);
      if ((std::abs(scale_t - scale) / scale_t) < 1e-5) {
        return true;
      }
      return false;
    }

    auto max_seqlen_q =
        op->attribute("max_seqlen_q").dyn_cast<pir::FloatAttribute>().data();
    auto max_seqlen_k =
        op->attribute("max_seqlen_k").dyn_cast<pir::FloatAttribute>().data();
    if (max_seqlen_q > 0 || max_seqlen_k > 0) {
      return false;
    }

    return true;
  }

  void Rewrite(paddle::dialect::MemoryEfficientAttentionOp op,
               pir::PatternRewriter& rewriter) const override {
    auto q = op->operand_source(0);
    auto k = op->operand_source(1);
    auto v = op->operand_source(2);

    auto dropout_p =
        op->attribute("dropout_p").dyn_cast<pir::DoubleAttribute>().data();

    auto causal = op->attribute("causal").dyn_cast<pir::BoolAttribute>().data();

    pir::Value fixed_seed;
    pir::Value attn_mask;
    auto fa = rewriter.Build<paddle::dialect::FlashAttnOp>(
        q, k, v, fixed_seed, attn_mask, dropout_p, causal, false, true, "");

    rewriter.ReplaceAllUsesWith(op->result(0), fa.result(0));
    rewriter.EraseOp(op);
  }
};

class ConvertMEA2FAPass : public pir::PatternRewritePass {
 public:
  ConvertMEA2FAPass() : pir::PatternRewritePass("convert_MEA_to_FA", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ConvertMEA2FAPattern>(context);
    return ps;
  }
};

std::unique_ptr<pir::Pass> CreateConvertMEA2FAPass() {
  return std::make_unique<ConvertMEA2FAPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

REGISTER_IR_PASS(convert_MEA_to_FA, ::cinn::dialect::ir::ConvertMEA2FAPass);
