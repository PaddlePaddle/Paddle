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

#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
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
    auto ValidMatmulTranspose = [&](pir::Operation* op) -> bool {
      if (!op->dyn_cast<paddle::dialect::MatmulOp>()) {
        return false;
      }

      bool trans_x =
          op->attribute("transpose_x").dyn_cast<pir::BoolAttribute>().data();
      bool trans_y =
          op->attribute("transpose_y").dyn_cast<pir::BoolAttribute>().data();
      return !trans_x && !trans_y;
    };
    if (!ValidMatmulTranspose(matmul_op)) {
      return false;
    }

    auto IsFirstInput = [&](pir::Operation* op, pir::Value in_x) -> bool {
      return in_x == op->operand_source(0);
    };

    auto VectorPrefixEqual = [](const std::vector<std::int64_t>& a,
                                const std::vector<std::int64_t>& b) {
      return std::vector<std::int64_t>(a.begin(), a.end() - 1) ==
             std::vector<std::int64_t>(b.begin(), b.end() - 1);
    };

    auto IsDynamicShape = [&](const std::vector<int64_t>& dims) {
      return std::any_of(
          dims.begin(), dims.end(), [](int64_t dim) { return dim < 0; });
    };

    auto input_x = matmul_op.operand_source(0);
    std::vector<pir::Operation*> merge_ops = [&]() {
      std::vector<pir::Operation*> ret;
      std::optional<std::vector<std::int64_t>> pre_dim;
      std::vector<std::int64_t> cur_dim;
      for (auto it = input_x.use_begin(); it != input_x.use_end(); ++it) {
        if (!ValidMatmulTranspose(it->owner())) {
          continue;
        }

        if (!IsFirstInput(it->owner(), input_x)) {
          continue;
        }
        if (!pre_dim.has_value()) {
          pre_dim = ::common::vectorize(
              it->owner()
                  ->operand_source(1)
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dims());
        }
        cur_dim = ::common::vectorize(
            it->owner()
                ->operand_source(1)
                .type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dims());
        if (IsDynamicShape(cur_dim)) {
          continue;
        }
        if (VectorPrefixEqual(pre_dim.value(), cur_dim)) {
          ret.push_back(it->owner());
        }
      }
      return ret;
    }();
    if (merge_ops.size() <= 1) {
      return false;
    }
    std::sort(
        merge_ops.begin(),
        merge_ops.end(),
        [&](pir::Operation* a, pir::Operation* b) {
          int a_distance = std::distance(a->GetParent()->begin(),
                                         a->operator pir::Block::Iterator());
          int b_distance = std::distance(b->GetParent()->begin(),
                                         b->operator pir::Block::Iterator());
          return a_distance < b_distance;
        });

    const std::vector<pir::Value> combine_ins = [&]() {
      std::vector<pir::Value> ret;
      for (pir::Operation* op : merge_ops) {
        ret.push_back(op->operand_source(1));
      }
      return ret;
    }();
    const std::vector<std::int64_t> combine_shapes = [&]() {
      std::vector<std::int64_t> ret{0};
      std::int64_t accumulate = 0;
      for (pir::Value input : combine_ins) {
        auto shape =
            input.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
        accumulate += shape[shape.size() - 1];
        ret.push_back(accumulate);
      }
      return ret;
    }();
    const std::vector<pir::Value> outputs = [&]() {
      std::vector<pir::Value> ret;
      for (pir::Operation* matmul_op : merge_ops) {
        ret.push_back(matmul_op->result(0));
      }
      return ret;
    }();

    auto* insert_point = FindInsertPoint(merge_ops, outputs);
    MoveUpstreamOpBeforeGroup(
        merge_ops, merge_ops.back()->GetParent(), insert_point);
    rewriter.set_insertion_point(insert_point);

    auto combine_out = rewriter.Build<pir::CombineOp>(combine_ins).result(0);
    auto concat_out =
        rewriter.Build<paddle::dialect::ConcatOp>(combine_out, -1).result(0);
    auto matmul_out =
        rewriter.Build<paddle::dialect::MatmulOp>(input_x, concat_out)
            .result(0);

    const auto& matmul_out_rank =
        matmul_out.type()
            .dyn_cast<paddle::dialect::DenseTensorType>()
            .dims()
            .size();

    for (size_t i = 0; i < merge_ops.size(); ++i) {
      auto split_out = rewriter
                           .Build<paddle::dialect::SliceOp>(
                               matmul_out,
                               std::vector<std::int64_t>{matmul_out_rank - 1},
                               std::vector<std::int64_t>{combine_shapes[i]},
                               std::vector<int64_t>{combine_shapes[i + 1]},
                               std::vector<std::int64_t>{},
                               std::vector<std::int64_t>{})
                           .result(0);

      rewriter.ReplaceAllUsesWith(merge_ops[i]->result(0), split_out);
      rewriter.EraseOp(merge_ops[i]);
    }

    return true;
  }
};

class FuseParallelMatmulPass : public pir::PatternRewritePass {
 public:
  FuseParallelMatmulPass()
      : pir::PatternRewritePass("fuse_parallel_matmul_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<MergeParallelMatmulPattern>(context);
    return ps;
  }
};

std::unique_ptr<pir::Pass> CreateFuseParallelMatmulPass() {
  return std::make_unique<FuseParallelMatmulPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
