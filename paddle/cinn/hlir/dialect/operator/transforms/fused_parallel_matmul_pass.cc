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

#include "paddle/cinn/hlir/dialect/operator/transforms/fused_parallel_matmul_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class MergeParallelMatmulPattern
    : public pir::OpRewritePattern<paddle::dialect::MatmulOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MatmulOp matmul_op,
                       pir::PatternRewriter& rewriter) const override {
    auto input_x = matmul_op.operand_source(0);
    size_t matmul_count = 0;

    auto GetTransposeAttr = [&](pir::Operation* op,
                                const std::string& attr_name) -> bool {
      return op->attribute(attr_name).dyn_cast<pir::BoolAttribute>().data();
    };

    auto CanMerge = [&](pir::Operation* op) -> bool {
      if (auto inner_matmul = op->dyn_cast<paddle::dialect::MatmulOp>()) {
        auto trans_x = GetTransposeAttr(op, "transpose_x");
        auto trans_y = GetTransposeAttr(op, "transpose_y");
        auto y_dim = inner_matmul->operand_source(1)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims();

        return (trans_x == false) && (trans_x == false) && (y_dim.size() == 2);
      }

      return false;
    };

    auto transpose_x = GetTransposeAttr(matmul_op, "transpose_x");
    auto transpose_y = GetTransposeAttr(matmul_op, "transpose_y");

    if (transpose_x == false && transpose_y == false) {
      std::vector<::pir::Operation*> merge_ops;
      for (auto it = input_x.use_begin(); it != input_x.use_end(); ++it) {
        if (CanMerge(it->owner())) {
          merge_ops.push_back(it->owner());
        }
      }

      if (merge_ops.size() > 1) {
        {
          std::vector<pir::Value> combine_ins;
          for (auto& op : merge_ops) {
            combine_ins.push_back(op->operand_source(1));
          }
          auto combine_out =
              rewriter.Build<pir::CombineOp>(combine_ins).result(0);
          auto concat_out =
              rewriter.Build<paddle::dialect::ConcatOp>(combine_out, -1)
                  .result(0);
          auto matmul_out =
              rewriter.Build<paddle::dialect::MatmulOp>(input_x, concat_out)
                  .result(0);

          auto split_out = rewriter
                               .Build<paddle::dialect::SplitWithNumOp>(
                                   matmul_out, merge_ops.size(), -1)
                               .result(0);

          auto split_vec = rewriter.Build<pir::SplitOp>(split_out).outputs();

          for (size_t i = 0; i < merge_ops.size(); ++i) {
            rewriter.ReplaceAllUsesWith(merge_ops[i]->result(0), split_vec[i]);
            rewriter.EraseOp(merge_ops[i]);
          }
        }

        return true;
      }
    }
    return false;
  }
};

class FusedParallelMatmulPass : public pir::PatternRewritePass {
 public:
  FusedParallelMatmulPass()
      : pir::PatternRewritePass("fused_parallel_matmul_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<MergeParallelMatmulPattern>(context);
    return ps;
  }
};

std::unique_ptr<pir::Pass> CreateFusedParallelMatmulPass() {
  return std::make_unique<FusedParallelMatmulPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
