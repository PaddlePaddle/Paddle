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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/single_op_fallback_to_phi.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    // Fallback only when FusionOp has two operators inside: AnySingleOp +
    // cf.yield
    if (fusion_op.GetOperators().size() > 2) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        fusion_op.GetOperators().size(),
        2,
        ::common::errors::InvalidArgument(
            "fusion_op should have two operators inside, but got %d",
            fusion_op.GetOperators().size()));
    PADDLE_ENFORCE(
        fusion_op.GetOperators()[1]->isa<::pir::YieldOp>(),
        ::common::errors::InvalidArgument(
            "The last operator of fusion_op must be YieldOp, but got %s",
            fusion_op.GetOperators()[1]->name()));

    auto* program = fusion_op->GetParentProgram();
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
        fusion_op->GetParentProgram());
    std::optional<pir::Operation*> paddle_op =
        FallBackOp(fusion_op.GetOperators()[0], rewriter);
    if (!paddle_op.has_value()) {
      return false;
    }

    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(fusion_op.result(i),
                                  paddle_op.value()->result(i));
    }

    rewriter.EraseOp(fusion_op);
    return true;
  }

 private:
  typedef pir::Operation* (FusionOpPattern::*CinnOpHandler)(
      pir::Operation*, pir::PatternRewriter&) const;

  pir::Operation* ReshapeOpPattern(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    PADDLE_ENFORCE(op->isa<cinn::dialect::ReshapeOp>(),
                   ::common::errors::InvalidArgument(
                       "Input should be cinn::dialect::ReshapeOp, but got %s",
                       op->name()));
    auto reshape_op = op->dyn_cast<cinn::dialect::ReshapeOp>();

    const std::vector<int64_t> vec_out_shape = [&]() {
      auto out_shape_attr = reshape_op.attribute("shape")
                                .dyn_cast<pir::ArrayAttribute>()
                                .AsVector();
      PADDLE_ENFORCE_GT(out_shape_attr.size(),
                        0,
                        ::common::errors::InvalidArgument(
                            "The shape attribute should not be empty"));

      std::vector<int64_t> ret;
      std::transform(
          out_shape_attr.begin(),
          out_shape_attr.end(),
          std::back_inserter(ret),
          [](const auto& attr) {
            return attr.template dyn_cast<::pir::Int32Attribute>().data();
          });
      return ret;
    }();

    auto paddle_reshape = rewriter.Build<paddle::dialect::ReshapeOp>(
        reshape_op->operand_source(0), vec_out_shape);
    return paddle_reshape;
  }

  const std::unordered_map<std::string, CinnOpHandler>& op_handler_map() const {
    static std::unordered_map<std::string, CinnOpHandler> handler_map = {
        {cinn::dialect::ReshapeOp::name(), &FusionOpPattern::ReshapeOpPattern},
    };
    return handler_map;
  }

  std::optional<pir::Operation*> FallBackOp(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    auto it = op_handler_map().find(op->name());
    if (it == op_handler_map().end()) {
      VLOG(4) << "No fallback handler for op: " << op->name();
      return std::nullopt;
    }
    return (this->*(it->second))(op, rewriter);
  }
};

class SingleOpFallbackToPhiPass : public pir::PatternRewritePass {
 public:
  SingleOpFallbackToPhiPass()
      : pir::PatternRewritePass("single_op_fallback_to_phi", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSingleOpFallbackToPhiPass() {
  return std::make_unique<SingleOpFallbackToPhiPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
