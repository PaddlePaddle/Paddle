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

#include "paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

bool CanMerge(pir::Operation* op) {
  auto& in_dims = op->operand_source(0)
                      .type()
                      .dyn_cast<paddle::dialect::DenseTensorType>()
                      .dims();
  auto& out_dims =
      op->result(0).type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
  std::vector<int64_t> vec_in_dim;
  std::vector<int64_t> vec_out_dim;

  for (size_t i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] != 1) {
      vec_in_dim.push_back(in_dims[i]);
    }
  }

  for (size_t i = 0; i < out_dims.size(); ++i) {
    if (out_dims[i] != 1) {
      vec_out_dim.push_back(out_dims[i]);
    }
  }

  return vec_in_dim == vec_out_dim;
}

std::vector<int64_t> GetBroadcastAxis(pir::Operation* reshape_op,
                                      pir::Operation* broadcast_op) {
  auto in_dims =
      phi::vectorize(reshape_op->operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims());
  auto out_dims =
      phi::vectorize(reshape_op->result(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims());

  auto op_broadcast_axes =
      cinn::dialect::ir::GetVectorAttr(broadcast_op, "broadcast_axes");

  std::vector<int64_t> new_broadcast_axes(in_dims.size(), 0);
  std::reverse(in_dims.begin(), in_dims.end());
  std::reverse(out_dims.begin(), out_dims.end());

  auto in_shape_size = in_dims.size();

  size_t out_index = 0;
  for (size_t i = 0; i < in_shape_size; ++i) {
    while (in_dims[i] != out_dims[out_index]) {
      out_index++;
    }

    size_t forward_out_index = out_dims.size() - 1 - out_index;

    new_broadcast_axes[in_shape_size - 1 - i] =
        op_broadcast_axes[forward_out_index];
  }

  return new_broadcast_axes;
}

class MergeReshapeWithBroadcastPattern
    : public pir::OpRewritePattern<cinn::dialect::BroadcastOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::BroadcastOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::BroadcastOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto reshape_op = op->operand_source(0)
                          .defining_op()
                          ->dyn_cast<cinn::dialect::ReshapeOp>();

    if (reshape_op && CanMerge(reshape_op)) {
      auto broadcast_axes = GetBroadcastAxis(reshape_op, op);

      auto output_shape =
          phi::vectorize(op->result(0)
                             .type()
                             .dyn_cast<paddle::dialect::DenseTensorType>()
                             .dims());
      auto new_broadcast_op = rewriter.Build<cinn::dialect::BroadcastOp>(
          reshape_op->operand_source(0), broadcast_axes, output_shape);

      rewriter.ReplaceAllUsesWith(op->result(0), new_broadcast_op.result(0));
      rewriter.EraseOp(op);
      return true;
    }

    return false;
  }
};

MergeReshapeWithBroadcastPass::MergeReshapeWithBroadcastPass()
    : pir::PatternRewritePass("merge_reshape_with_broadcast_pass", 1) {}

pir::RewritePatternSet MergeReshapeWithBroadcastPass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);

  // merge reshape with broadcast op
  ps.Add<MergeReshapeWithBroadcastPattern>(context);

  return ps;
}

bool MergeReshapeWithBroadcastPass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
