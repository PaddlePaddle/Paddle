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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

int64_t GetDimByIndex(const phi::DDim& first,
                      const phi::DDim& second,
                      int short_align_axis,
                      int idx) {
  // rank of first less than rank of second
  if (idx < short_align_axis) {
    return second[idx];
  } else {
    return first[idx - short_align_axis] > second[idx]
               ? first[idx - short_align_axis]
               : second[idx];
  }
}

std::vector<int64_t> GetOutputShape(const phi::DDim& x, const phi::DDim& y) {
  std::vector<int64_t> vec_res;
  if (x.size() >= y.size()) {
    int short_align_axis = x.size() - y.size();
    int max_rank = x.size();
    vec_res.resize(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
      vec_res[i] = GetDimByIndex(y, x, short_align_axis, i);
    }
  } else {
    int short_align_axis = y.size() - x.size();
    int max_rank = y.size();

    vec_res.resize(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
      vec_res[i] = GetDimByIndex(x, y, short_align_axis, max_rank);
    }
  }

  return vec_res;
}

bool is_same_dim(const phi::DDim& first, const std::vector<int64_t>& second) {
  if (first.size() == second.size()) {
    bool same = true;

    for (size_t i = 0; i < first.size(); ++i) {
      if (first[i] != second[i]) {
        same = false;
        break;
      }
    }

    return same;
  }
  return false;
}

bool ProcessOp(pir::Operation* op, pir::PatternRewriter* rewriter) {
  auto x_dims = op->operand_source(0)
                    .type()
                    .dyn_cast<paddle::dialect::DenseTensorType>()
                    .dims();
  auto y_dims = op->operand_source(1)
                    .type()
                    .dyn_cast<paddle::dialect::DenseTensorType>()
                    .dims();

  if (x_dims != y_dims) {
    auto output_shape = GetOutputShape(x_dims, y_dims);
    if (!is_same_dim(x_dims, output_shape)) {
      // add broadcast to input 0
      auto new_transpose_op = rewriter->Build<cinn::dialect::BroadcastOp>(
          op->operand_source(0), std::vector<int64_t>({}), output_shape);

      op->operand(0).set_source(new_transpose_op->result(0));
    }

    if (!is_same_dim(y_dims, output_shape)) {
      auto new_transpose_op = rewriter->Build<cinn::dialect::BroadcastOp>(
          op->operand_source(1), std::vector<int64_t>({}), output_shape);

      op->operand(1).set_source(new_transpose_op->result(0));
    }

    return true;
  }

  return false;
}

class AddBrodcastToElementwiseAddPattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AddOp op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

class AddBrodcastToElementwiseSubPattern
    : public pir::OpRewritePattern<paddle::dialect::SubtractOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SubtractOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SubtractOp op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

class AddBrodcastToElementwiseDivPattern
    : public pir::OpRewritePattern<paddle::dialect::DivideOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::DivideOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::DivideOp op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

class AddBrodcastToElementwiseMulPattern
    : public pir::OpRewritePattern<paddle::dialect::MultiplyOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MultiplyOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MultiplyOp op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

AddBroadcastToElementwisePass::AddBroadcastToElementwisePass()
    : pir::Pass("add_broadcast_to_elementwise_pass", 1) {}

bool AddBroadcastToElementwisePass::Initialize(pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  ps.Add<AddBrodcastToElementwiseAddPattern>(context);
  ps.Add<AddBrodcastToElementwiseSubPattern>(context);
  ps.Add<AddBrodcastToElementwiseDivPattern>(context);
  ps.Add<AddBrodcastToElementwiseMulPattern>(context);

  patterns_ = ::pir::FrozenRewritePatternSet(std::move(ps));
  return true;
}

void AddBroadcastToElementwisePass::Run(pir::Operation* op) {
  pir::GreedyRewriteConfig cfg;
  cfg.use_top_down_traversal = true;
  cfg.max_iterations = 10;
  pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
}

bool AddBroadcastToElementwisePass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
