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
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/reduce_split_pass.h"
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

/*
 * Split a ReduceOp that has a large reduce size into two ReduceOps, in order
 * to improve parallelism.
 *
 * Example:
 *   x.shape = [16, 65536]
 *
 *   Program before pass:
 *     out = reduce(x, axis=1)  # [16]
 *
 *   Program after pass:
 *     tmp = reshape(x, [16, 8, 8192])
 *     tmp = reduce(tmp, axis=2)  # [16, 8]
 *     out = reduce(tmp, axis=1)  # [16]
 */
class ReduceSplitPattern
    : public pir::OpRewritePattern<cinn::dialect::ReduceSumOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::ReduceSumOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::ReduceSumOp sum_op,
                       pir::PatternRewriter& rewriter) const override {
    auto reduce_axis = cinn::dialect::ir::GetVectorAttr(sum_op, "axis");
    auto input_dim = sum_op->operand_source(0)
                         .type()
                         .dyn_cast<pir::DenseTensorType>()
                         .dims();

    size_t spatial_num = 1;
    size_t reduce_num = 1;

    for (int axis = 0; axis < input_dim.size(); axis++) {
      if (input_dim[axis] < 0) {
        return false;
      }
      spatial_num *= input_dim[axis];
    }
    for (auto axis : reduce_axis) {
      if (axis < 0) {
        axis += input_dim.size();
      }
      reduce_num *= input_dim[axis];
    }
    spatial_num /= reduce_num;

    if (spatial_num >= 128 || reduce_num < 4096) {
      return false;
    }

    // split the first reduce axis into [fac, -1], where fac is its minimal
    // factor that satisfies `fac * spatial_num >= 128`
    int split_axis = reduce_axis[0];
    if (split_axis < 0) {
      split_axis += input_dim.size();
    }
    int first_split_factor = input_dim[split_axis];
    for (int fac = 2; fac < sqrt(input_dim[split_axis]) + 1; fac++) {
      if (input_dim[split_axis] % fac == 0) {
        if ((input_dim[split_axis] / fac) * spatial_num >= 128) {
          first_split_factor = input_dim[split_axis] / fac;
        }
        if (fac * spatial_num >= 128) {
          first_split_factor = fac;
          break;
        }
      }
    }
    std::vector<int64_t> split_factor;
    if (first_split_factor < input_dim[split_axis]) {
      split_factor.push_back(first_split_factor);
      split_factor.push_back(input_dim[split_axis] / first_split_factor);
    }
    VLOG(4) << "split_axis: " << split_axis;
    VLOG(4) << "split_factor: " << utils::Join(split_factor, ", ");

    // add reshape if we have splitted the first reduce axis
    auto input_x = sum_op.operand_source(0);
    if (split_factor.size() == 2) {
      std::vector<int> output_shape;
      for (int i = 0; i < input_dim.size(); ++i) {
        if (i != split_axis) {
          output_shape.push_back(input_dim[i]);
        } else {
          output_shape.push_back(split_factor.front());
          output_shape.push_back(split_factor.back());
        }
      }
      input_x = rewriter
                    .Build<cinn::dialect::ReshapeOp>(sum_op.operand_source(0),
                                                     output_shape)
                    .result(0);
    }

    std::vector<int64_t> first_reduce_axis;
    std::vector<int64_t> second_reduce_axis;

    for (auto axis : reduce_axis) {
      if (axis < 0) {
        axis += input_dim.size();
      }

      if (axis < split_axis) {
        second_reduce_axis.push_back(axis);
      } else if (axis == split_axis) {
        second_reduce_axis.push_back(axis);
        if (split_factor.size() == 2) {
          first_reduce_axis.push_back(axis + 1);
        }
      } else {
        if (split_factor.size() == 2) {
          first_reduce_axis.push_back(axis + 1);
        } else {
          first_reduce_axis.push_back(axis);
        }
      }
    }

    bool orig_keepdim =
        sum_op.attribute("keepdim").dyn_cast<pir::BoolAttribute>().data();

    auto first_reduce_out = rewriter
                                .Build<cinn::dialect::ReduceSumOp>(
                                    input_x, first_reduce_axis, orig_keepdim)
                                .result(0);

    bool second_keepdim = orig_keepdim && split_factor.empty();

    auto second_reduce_out =
        rewriter
            .Build<cinn::dialect::ReduceSumOp>(
                first_reduce_out, second_reduce_axis, second_keepdim)
            .result(0);

    rewriter.ReplaceAllUsesWith(sum_op.result(0), second_reduce_out);
    rewriter.EraseOp(sum_op);
    return true;
  }
};

class ReduceSplitPass : public pir::PatternRewritePass {
 public:
  ReduceSplitPass() : pir::PatternRewritePass("reduce_split_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ReduceSplitPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateReduceSplitPass() {
  return std::make_unique<ReduceSplitPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
