// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_cast_to_elementwise_add_pass.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace cinn {
namespace dialect {
namespace ir {

pir::Type GetOutputDtype(const pir::Type& x, const pir::Type& y) {
  pir::IrContext* context = pir::IrContext::Instance();
  // type promotion
  if (x.isa<pir::Complex128Type>() || y.isa<pir::Complex128Type>()) {
    return pir::Complex128Type::get(context);
  }
  if (x.isa<pir::Complex64Type>() || y.isa<pir::Complex64Type>()) {
    return pir::Complex64Type::get(context);
  }

  auto is_integer_or_bool = [](const pir::Type& x) {
    return x.isa<pir::IndexType>() || x.isa<pir::Int64Type>() ||
           x.isa<pir::Int32Type>() || x.isa<pir::Int16Type>() ||
           x.isa<pir::Int8Type>() || x.isa<pir::UInt8Type>() ||
           x.isa<pir::BoolType>();
  };

  if (is_integer_or_bool(x) || is_integer_or_bool(y)) {
    PADDLE_THROW(::common::errors::InvalidType(
        "Type promotion only support calculations between floating-point "
        "numbers and between complex and real numbers. But got different "
        "data type x: %s, y: %s.",
        ::paddle::dialect::TransToPhiDataType(x),
        ::paddle::dialect::TransToPhiDataType(y)));
  }

  if (x.isa<pir::Float64Type>() || y.isa<pir::Float64Type>()) {
    return pir::Float64Type::get(context);
  }
  return pir::Float32Type::get(context);
}

template <typename OPTYPE>
class AddCastToElementwiseAddPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    const pir::Type& x_dtype = op->operand_source(0)
                                   .type()
                                   .template dyn_cast<pir::DenseTensorType>()
                                   .dtype();
    const pir::Type& y_dtype = op->operand_source(1)
                                   .type()
                                   .template dyn_cast<pir::DenseTensorType>()
                                   .dtype();

    if (x_dtype != y_dtype) {
      pir::Type output_dtype = GetOutputDtype(x_dtype, y_dtype);

      auto cast_op0 = rewriter.Build<paddle::dialect::CastOp>(
          op->operand_source(0),
          ::paddle::dialect::TransToPhiDataType(output_dtype));

      auto cast_op1 = rewriter.Build<paddle::dialect::CastOp>(
          op->operand_source(1),
          ::paddle::dialect::TransToPhiDataType(output_dtype));

      op->operand(0).set_source(cast_op0->result(0));
      op->operand(1).set_source(cast_op1->result(0));

      return true;
    }
    return false;
  }
};

class AddCastToElementwiseAddPass : public pir::PatternRewritePass {
 public:
  AddCastToElementwiseAddPass()
      : pir::PatternRewritePass("add_cast_to_elementwise_add_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddCastToElementwiseAddPattern<paddle::dialect::AddOp>>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0 && op->isa<cinn::dialect::GroupOp>();
  }
};

// NOTE: This is a temporary type promotion pass in the CINN frontend.
// It is necessary because the `NeedTypePromotion` function in
// `paddle/phi/common/type_promotion.h` explicitly disables automatic promotion
// for fp16 and bf16 data types. This pass ensures type promotion for 'add'
// operations with mixed-type operands, which the `NeedTypePromotion` function
// blocks.
//
// This pass becomes obsolete and should be removed once the restriction on
// fp16/bf16 promotion is lifted from the common type promotion logic.
std::unique_ptr<pir::Pass> CreateAddCastToElementwiseAddPass() {
  return std::make_unique<AddCastToElementwiseAddPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
