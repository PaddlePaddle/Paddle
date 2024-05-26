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

#include <memory>
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/pass/pass.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace details {

struct TensorDimInShape {
  pir::Value value;
  int axis;
};

struct TensorDimInData {
  pir::Value value;
  int axis;
};

using TensorDim = std::variant<TensorDimInShape, TensorDimInData>;

using TensorDim4SymbolNameT =
    std::function<std::optional<TensorDim>(const std::string& symbol_name)>;

using SymbolName2CachedValue = std::unordered_map<symbol::DimExpr, pir::Value>;

struct CachedDimExprToValueConverter {
  CachedDimExprToValueConverter(
      const TensorDim4SymbolNameT& TensorDim4SymbolNameVal,
      pir::PatternRewriter* rewriter_val)
      : TensorDim4SymbolName(TensorDim4SymbolNameVal), rewriter(rewriter_val) {}

  TensorDim4SymbolNameT TensorDim4SymbolName;
  pir::PatternRewriter* rewriter;

  pir::Value ConvertToValue(const symbol::DimExpr& dim_expr) {
    pir::Value value =
        std::visit([&](const auto& impl) { return ConvertToValueImpl(impl); },
                   dim_expr.variant());
    return value;
  }

  pir::Value GetInputShapeByInputTensor(pir::Value input_tensor) {
    auto iter = tensor2shape_.find(input_tensor);
    if (iter == tensor2shape_.end()) {
      pir::Value shape =
          rewriter->Build<paddle::dialect::ShapeOp>(input_tensor).out();
      pir::Value cast_shape =
          rewriter->Build<paddle::dialect::CastOp>(shape, phi::DataType::INT64)
              .out();
      iter = tensor2shape_.emplace(input_tensor, cast_shape).first;
    }
    return iter->second;
  }

 private:
  std::unordered_map<pir::Value /*input tensor*/,
                     pir::Value /*input shape tensor*/>
      tensor2shape_;

  pir::Value ConvertToValueImpl(int64_t dim_expr) {
    return rewriter
        ->Build<paddle::dialect::FullIntArrayOp>(std::vector{dim_expr},
                                                 phi::DataType::INT64)
        .out();
  }

  pir::Value ConvertToValueImpl(const std::string& symbol_name) {
    const auto& tensor_dim = TensorDim4SymbolName(symbol_name);
    PADDLE_ENFORCE(
        tensor_dim.has_value(),
        phi::errors::PreconditionNotMet(
            "symbol [%s] are not bound to any input of generate_shape op",
            symbol_name));
    return std::visit(
        [&](const auto& impl) { return ConvertTensorDimToValue(impl); },
        tensor_dim.value());
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInShape& tensor_dim) {
    pir::Value input_shape = GetInputShapeByInputTensor(tensor_dim.value);
    return ConvertTensorDimToValue(
        TensorDimInData{.value = input_shape, .axis = tensor_dim.axis});
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInData& tensor_dim) {
    auto CastToInt64IfNeed = [&](pir::Value value) {
      if (value.type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dtype()
              .isa<pir::Int64Type>()) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::CastOp>(value, phi::DataType::INT64)
          .out();
    };
    auto FlattenValueIfNeed = [&](pir::Value value) {
      const auto& dims =
          value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      if (dims.size() <= 1) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::FlattenOp>(value, 0, dims.size() - 1)
          .out();
    };
    if (tensor_dim.value.type()
            .dyn_cast<paddle::dialect::DenseTensorType>()
            .dims()
            .size() == 0) {
      return CastToInt64IfNeed(
          rewriter
              ->Build<paddle::dialect::ReshapeOp>(tensor_dim.value,
                                                  std::vector<int64_t>{1})
              .out());
    }
    return CastToInt64IfNeed(rewriter
                                 ->Build<paddle::dialect::SliceOp>(
                                     FlattenValueIfNeed(tensor_dim.value),
                                     std::vector<int64_t>{0LL},
                                     std::vector<int64_t>{tensor_dim.axis},
                                     std::vector<int64_t>{tensor_dim.axis + 1},
                                     std::vector<int64_t>{},
                                     std::vector<int64_t>{})
                                 .out());
  }

  pir::Value ConvertToValueImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr) {
    PADDLE_THROW(
        phi::errors::Fatal("Dead code. This logical should handled by "
                           "ConvertToValueImpl(symbol::Add<symbol::DimExpr>)"));
  }

  pir::Value ConvertToValueImpl(
      const symbol::Reciprocal<symbol::DimExpr>& dim_expr) {
    PADDLE_THROW(
        phi::errors::Fatal("Dead code. This logical should handled by "
                           "ConvertToValueImpl(symbol::Mul<symbol::DimExpr>)"));
  }

  pir::Value ConvertToValueImpl(const symbol::Add<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value acc = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Negative<symbol::DimExpr>>()) {
        const auto& [operand] =
            *operands->at(i).dyn_cast<symbol::Negative<symbol::DimExpr>>();
        pir::Value operand_value = ConvertToValue(operand);
        acc = rewriter->Build<paddle::dialect::SubtractOp>(acc, operand_value)
                  .out();
      } else {
        pir::Value operand_value = ConvertToValue(operands->at(i));
        acc = rewriter->Build<paddle::dialect::AddOp>(acc, operand_value).out();
      }
    }
    return acc;
  }

  pir::Value ConvertToValueImpl(const symbol::Mul<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value prod = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Reciprocal<symbol::DimExpr>>()) {
        const auto& operand =
            operands->at(i)
                .dyn_cast<symbol::Reciprocal<symbol::DimExpr>>()
                ->data;
        pir::Value operand_value = ConvertToValue(operand);
        prod = rewriter->Build<paddle::dialect::DivideOp>(prod, operand_value)
                   .out();
      } else {
        pir::Value operand_value = ConvertToValue(operands->at(i));
        prod = rewriter->Build<paddle::dialect::MultiplyOp>(prod, operand_value)
                   .out();
      }
    }
    return prod;
  }

  pir::Value ConvertToValueImpl(const symbol::Max<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value max = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      max =
          rewriter->Build<paddle::dialect::MaximumOp>(max, operand_value).out();
    }
    return max;
  }

  pir::Value ConvertToValueImpl(const symbol::Min<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value min = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      min =
          rewriter->Build<paddle::dialect::MinimumOp>(min, operand_value).out();
    }
    return min;
  }

  pir::Value ConvertToValueImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value broadcasted = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      broadcasted = rewriter
                        ->Build<paddle::dialect::ShapeBroadcastOp>(
                            broadcasted, operand_value)
                        .out();
    }
    return broadcasted;
  }
};

std::optional<pir::Value> GetOutReplacement(cinn::dialect::GenerateShapeOp op,
                                            pir::PatternRewriter* rewriter);

TensorDim4SymbolNameT MakeGetterTensorDim4SymbolName(
    cinn::dialect::GenerateShapeOp op);

void InsertSymbolBinding(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::SymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim);

void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::DataSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim);
void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::ShapeSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim);

std::vector<symbol::DimExpr> GetOutDimExprs(cinn::dialect::GenerateShapeOp op);

pir::Value GetValueOfRewrittenOps(const std::vector<symbol::DimExpr>& dim_exprs,
                                  CachedDimExprToValueConverter* converter);

std::vector<pir::Value> GetValuesOfRewrittenOps(
    const std::vector<symbol::DimExpr>& dim_exprs,
    CachedDimExprToValueConverter* converter);
}  // namespace details
class SplitGenerateShapeIntoShapeOps
    : public pir::OpRewritePattern<cinn::dialect::GenerateShapeOp> {
 public:
  using OpRewritePattern<cinn::dialect::GenerateShapeOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::GenerateShapeOp op,
                       pir::PatternRewriter& rewriter) const override;
};

class SplitGenerateShapeIntoShapeOpsPass : public pir::PatternRewritePass {
 public:
  SplitGenerateShapeIntoShapeOpsPass()
      : pir::PatternRewritePass("split_generate_shape_into_shape_ops_pass", 1) {
  }

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<SplitGenerateShapeIntoShapeOps>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};
std::unique_ptr<pir::Pass> CreateSplitGenerateShapeIntoShapeOpsPass();

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
