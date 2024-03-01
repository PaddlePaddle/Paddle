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
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

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
               pir::PatternRewriter& rewriter) const override {
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

class SumOpPattern : public pir::OpRewritePattern<paddle::dialect::SumOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SumOp>::OpRewritePattern;

  bool Match(paddle::dialect::SumOp op) const override {
    const auto& tensor_type =
        op.result(0).type().dyn_cast<pir::DenseTensorType>();
    return tensor_type.dims().size() == 0;
  }

  void Rewrite(paddle::dialect::SumOp op,
               pir::PatternRewriter& rewriter) const override {
    std::vector<int64_t> axis{};
    const auto& dtype = op->attribute("dtype")
                            .dyn_cast<paddle::dialect::DataTypeAttribute>()
                            .data();
    auto new_reduce_op = rewriter.Build<paddle::dialect::SumOp>(
        op.operand_source(0), axis, dtype, /*keepdim=*/true);
    auto reshape_op = rewriter.Build<paddle::dialect::ReshapeOp>(
        new_reduce_op.result(0), /*shape=*/std::vector<int64_t>({1}));
    rewriter.ReplaceAllUsesWith(op.result(0), reshape_op.result(0));
    rewriter.EraseOp(op);
  }
};

pir::DenseTensorType Make1DTensorType(const pir::DenseTensorType& tensor_type) {
  return pir::DenseTensorType::get(pir::IrContext::Instance(),
                                   tensor_type.dtype(),
                                   {1},
                                   tensor_type.data_layout(),
                                   tensor_type.lod(),
                                   tensor_type.offset());
}

void ConvertValue0DTo1D(pir::Value operand) {
  auto ConvertVectorType0DTo1D =
      [](const pir::VectorType& vector_tensor_type) -> std::vector<pir::Type> {
    std::vector<pir::Type> types;
    for (std::size_t i = 0; i < vector_tensor_type.size(); ++i) {
      CHECK(vector_tensor_type[i].isa<pir::DenseTensorType>());
      const auto& dense_type =
          vector_tensor_type[i].dyn_cast<pir::DenseTensorType>();
      types.push_back(dense_type.dims().size() == 0
                          ? Make1DTensorType(dense_type)
                          : vector_tensor_type[i]);
    }
    return types;
  };

  if (const auto& tensor_type =
          operand.type().dyn_cast<pir::DenseTensorType>()) {
    if (tensor_type.dims().size() == 0) {
      operand.set_type(Make1DTensorType(tensor_type));
    }
  } else if (const auto& vector_tensor_type =
                 operand.type().dyn_cast<pir::VectorType>()) {
    pir::Builder builder(pir::IrContext::Instance());
    std::vector<pir::Type> inputs_type =
        ConvertVectorType0DTo1D(vector_tensor_type);
    operand.set_type(builder.vec_type(inputs_type));
  } else {
    VLOG(4) << "Unsupported operand type: " << operand.type();
  }
}

class WhileOpPattern : public pir::OpRewritePattern<paddle::dialect::WhileOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::WhileOp>::OpRewritePattern;

  bool Match(paddle::dialect::WhileOp op) const override {
    for (const auto& value : op.block_args()) {
      if (const auto& tensor_type =
              value.type().template dyn_cast<pir::DenseTensorType>()) {
        if (tensor_type.dims().size() == 0) {
          return true;
        }
      }
    }
    return false;
  }

  void Rewrite(paddle::dialect::WhileOp op,
               pir::PatternRewriter& rewriter) const override {
    for (pir::Value value : op.block_args()) {
      ConvertValue0DTo1D(value);
    }
  }
};

class CombineOpPattern : public pir::OpRewritePattern<pir::CombineOp> {
 public:
  using pir::OpRewritePattern<pir::CombineOp>::OpRewritePattern;

  bool Match(pir::CombineOp op) const override {
    for (std::size_t i = 1; i < op->operands().size(); ++i) {
      if (op.operand_source(i).type() != op.operand_source(0).type()) {
        return true;
      }
    }
    return false;
  }

  void Rewrite(pir::CombineOp op,
               pir::PatternRewriter& rewriter) const override {
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
};

class Convert0DTo1DPass : public pir::Pass {
 public:
  Convert0DTo1DPass() : pir::Pass("convert_0D_to_1D", 1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FullOpPattern>(context);
    ps.Add<CombineOpPattern>(context);
    ps.Add<SumOpPattern>(context);
    ps.Add<WhileOpPattern>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      ApplyPatternOnOperation(op->region(i));
      for (const auto& block : op->region(i)) {
        ConvertBlock0DTo1D(block);
      }
    }
  }

  void ApplyPatternOnOperation(pir::Region& region) {  // NOLINT
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    const auto& [_, num_rewrites] =
        pir::ApplyPatternsGreedily(region, patterns_, cfg);
    AddStatistics(num_rewrites);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

  void ConvertOperation0DTo1D(const pir::Operation& op) {  // NOLINT
    for (std::size_t i = 0; i < op.num_operands(); ++i) {
      ConvertValue0DTo1D(op.operand_source(i));
    }
    for (std::size_t i = 0; i < op.num_results(); ++i) {
      ConvertValue0DTo1D(op.result(i));
    }
  }

  void ConvertBlock0DTo1D(const pir::Block& block) {
    for (auto& op : block) {
      ConvertOperation0DTo1D(op);
      for (std::size_t i = 0; i < op.num_regions(); ++i) {
        ApplyPatternOnOperation(op.region(i));
        for (auto& inner_block : op.region(i)) {
          ConvertBlock0DTo1D(inner_block);
        }
      }
    }
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateConvert0DTo1DPass() {
  return std::make_unique<Convert0DTo1DPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
