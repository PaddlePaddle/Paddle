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
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
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
      vec_res[i] = GetDimByIndex(x, y, short_align_axis, i);
    }
  }

  return vec_res;
}

bool IsSameDim(const phi::DDim& first, const std::vector<int64_t>& second) {
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
    if (!IsSameDim(x_dims, output_shape)) {
      // add broadcast to input 0
      if (auto full_op = op->operand_source(0)
                             .dyn_cast<pir::OpResult>()
                             .owner()
                             ->dyn_cast<paddle::dialect::FullOp>()) {
        auto new_full = rewriter->Build<paddle::dialect::FullOp>(
            output_shape,
            full_op->attribute("value").dyn_cast<pir::FloatAttribute>().data(),
            full_op->attribute("dtype")
                .dyn_cast<paddle::dialect::DataTypeAttribute>()
                .data(),
            full_op->attribute("place")
                .dyn_cast<paddle::dialect::PlaceAttribute>()
                .data());
        op->operand(0).set_source(new_full->result(0));
      } else {
        auto new_transpose_op = rewriter->Build<cinn::dialect::BroadcastOp>(
            op->operand_source(0),
            cinn::hlir::framework::pir::GetBroadcastAxis(x_dims, output_shape),
            output_shape);

        op->operand(0).set_source(new_transpose_op->result(0));
      }
    }

    if (!IsSameDim(y_dims, output_shape)) {
      if (auto full_op = op->operand_source(1)
                             .dyn_cast<pir::OpResult>()
                             .owner()
                             ->dyn_cast<paddle::dialect::FullOp>()) {
        auto new_full = rewriter->Build<paddle::dialect::FullOp>(
            output_shape,
            full_op->attribute("value").dyn_cast<pir::FloatAttribute>().data(),
            full_op->attribute("dtype")
                .dyn_cast<paddle::dialect::DataTypeAttribute>()
                .data(),
            full_op->attribute("place")
                .dyn_cast<paddle::dialect::PlaceAttribute>()
                .data());

        op->operand(1).set_source(new_full->result(0));
      } else {
        auto new_transpose_op = rewriter->Build<cinn::dialect::BroadcastOp>(
            op->operand_source(1),
            cinn::hlir::framework::pir::GetBroadcastAxis(y_dims, output_shape),
            output_shape);

        op->operand(1).set_source(new_transpose_op->result(0));
      }
    }

    return true;
  }

  return false;
}

template <typename OPTYPE>
class AddBrodcastToElementwisePattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter);
  }
};

AddBroadcastToElementwisePass::AddBroadcastToElementwisePass()
    : pir::PatternRewritePass("add_broadcast_to_elementwise_pass", 1) {}

pir::RewritePatternSet AddBroadcastToElementwisePass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  // elementwise ops
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::AddOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::SubtractOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::MultiplyOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::DivideOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::ElementwisePowOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::RemainderOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::FloorDivideOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::MaximumOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::MinimumOp>>(context);

  // compare ops
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::LessThanOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::LessEqualOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::EqualOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::NotEqualOp>>(context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::GreaterThanOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::GreaterEqualOp>>(
      context);

  // bitwise ops
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::BitwiseOrOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::BitwiseXorOp>>(
      context);
  ps.Add<AddBrodcastToElementwisePattern<paddle::dialect::BitwiseNotOp>>(
      context);

  return ps;
}

bool AddBroadcastToElementwisePass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
