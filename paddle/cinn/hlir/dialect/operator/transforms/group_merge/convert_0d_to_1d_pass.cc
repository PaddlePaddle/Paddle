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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_0d_to_1d_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class FullOpPattern : public pir::OpRewritePattern<paddle::dialect::FullOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FullOp>::OpRewritePattern;

  bool Match(paddle::dialect::FullOp op) const override {
    return op.attribute("shape")
               .dyn_cast<paddle::dialect::IntArrayAttribute>()
               .data()
               .size() == 0;
  }

  void Rewrite(paddle::dialect::FullOp op,
               pir::PatternRewriter &rewriter) const override {
    float factor =
        op->attribute("value").dyn_cast<::pir::FloatAttribute>().data();
    phi::DataType dtype = op->attribute("dtype")
                              .dyn_cast<paddle::dialect::DataTypeAttribute>()
                              .data();
    phi::Place place = op->attribute("place")
                           .dyn_cast<paddle::dialect::PlaceAttribute>()
                           .data();

    auto full_op = rewriter.Build<paddle::dialect::FullOp>(
        std::vector<int64_t>({1}), factor, dtype, place);
    rewriter.ReplaceAllUsesWith(op.result(0), full_op.result(0));
    rewriter.EraseOp(op);
  }
};

class CombineOpPattern : public pir::OpRewritePattern<pir::CombineOp> {
 public:
  using pir::OpRewritePattern<pir::CombineOp>::OpRewritePattern;

  bool Match(pir::CombineOp op) const override {
    auto out_type = op.result(0).type().dyn_cast<pir::VectorType>();
    for (auto type : out_type.data()) {
      if (HasZeroDim(type)) return true;
    }
    return false;
  }

  void Rewrite(pir::CombineOp op,
               pir::PatternRewriter &rewriter) const override {
    pir::Builder builder(rewriter.ir_context());

    const std::vector<pir::Type> inputs_type = [&]() {
      std::vector<pir::Type> types;
      for (auto value : op->operands_source()) {
        types.push_back(value.type());
      }
      return types;
    }();
    op.result(0).set_type(builder.vec_type(inputs_type));
  }

 private:
  bool HasZeroDim(pir::Type type) const {
    if (!type) return false;
    const auto dense_tensor_type = type.dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type && (dense_tensor_type.dims().size() == 0U);
  }
};

class Convert0DTo1DPass : public pir::PatternRewritePass {
 public:
  Convert0DTo1DPass() : pir::PatternRewritePass("convert_0D_to_1D", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FullOpPattern>(context);
    ps.Add<CombineOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation *op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateConvert0DTo1DPass() {
  return std::make_unique<Convert0DTo1DPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
