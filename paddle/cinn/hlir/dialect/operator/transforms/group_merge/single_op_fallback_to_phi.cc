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
    // Fallback only when FusionOp has two operators inside: AnySingleOp + yiled
    // store cf.yield

    if (fusion_op.GetOperators().size() != 3) {
      return false;
    }

    if (!fusion_op.GetOperators()[1]->isa<cinn::dialect::YieldStoreOp>()) {
      return false;
    }

    PADDLE_ENFORCE_EQ(
        fusion_op.GetOperators().size(),
        3,
        ::common::errors::InvalidArgument(
            "fusion_op should have two operators inside, but got %d",
            fusion_op.GetOperators().size()));
    PADDLE_ENFORCE(
        fusion_op.GetOperators()[2]->isa<::pir::YieldOp>(),
        ::common::errors::InvalidArgument(
            "The last operator of fusion_op must be YieldOp, but got %s",
            fusion_op.GetOperators()[2]->name()));

    std::optional<pir::Operation*> paddle_op =
        FallBackOp(fusion_op.GetOperators()[0], rewriter);
    if (!paddle_op.has_value()) {
      return false;
    }

    // TODO(phlrain): support multi output
    PADDLE_ENFORCE_EQ(
        paddle_op.value()->num_results(),
        1u,
        ::common::errors::PreconditionNotMet("Only support ONE output op"));

    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(fusion_op.result(i),
                                  paddle_op.value()->result(0));
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

  pir::Operation* AssignOutOpPattern(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    PADDLE_ENFORCE(
        op->isa<paddle::dialect::AssignOut_Op>(),
        ::common::errors::InvalidArgument(
            "Input should be paddle::dialect::AssignOut_Op, but got %s",
            op->name()));
    auto assign_out_op = op->dyn_cast<paddle::dialect::AssignOut_Op>();

    auto paddle_assign_out_ = rewriter.Build<paddle::dialect::AssignOut_Op>(
        assign_out_op->operand_source(0), assign_out_op->operand_source(1));
    return paddle_assign_out_;
  }

  pir::Operation* CastOpPattern(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    PADDLE_ENFORCE(
        op->isa<paddle::dialect::CastOp>(),
        ::common::errors::InvalidArgument(
            "Input should be paddle::dialect::CastOp, but got %s", op->name()));
    auto cast_op = op->dyn_cast<paddle::dialect::CastOp>();

    auto paddle_cast_op = rewriter.Build<paddle::dialect::CastOp>(
        cast_op->operand_source(0), cast_op->attributes());
    return paddle_cast_op;
  }

  const std::unordered_map<std::string, CinnOpHandler>& op_handler_map() const {
    static std::unordered_map<std::string, CinnOpHandler> handler_map = {
        {cinn::dialect::ReshapeOp::name(), &FusionOpPattern::ReshapeOpPattern},
        {paddle::dialect::AssignOut_Op::name(),
         &FusionOpPattern::AssignOutOpPattern},
        {paddle::dialect::CastOp::name(), &FusionOpPattern::CastOpPattern},
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

// Fallback reshape pattern like this:
// (%1) = cinn_op.generate_shape (%0)
// (%2) = pd_op.reshape (%0, %1)
// (%3) = cinn_op.yield_store (%2)
// () = cf.yield (%3)
class FusionOpSingleReshapePattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpSingleReshapePattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto& ops = fusion_op.GetOperators();
    if (ops.size() != 4) return false;
    if (!ops[0]->isa<cinn::dialect::GenerateShapeOp>() ||
        !ops[1]->isa<paddle::dialect::ReshapeOp>() ||
        !ops[2]->isa<cinn::dialect::YieldStoreOp>()) {
      return false;
    }

    // Input of generate_shape op and reshape op should be same, so
    // generate_shape has no new symbol dim
    if (ops[0]->num_operands() == 1 &&
        ops[0]->operand_source(0) != ops[1]->operand_source(0)) {
      return false;
    }

    // generate_shape op should only be used by reshape op
    // reshape op should only be used by yield_store op
    if (ops[0]->result(0).use_count() != 1 ||
        ops[1]->result(0).use_count() != 1) {
      return false;
    }
    if (ops[0]->result(0).first_use().owner() != ops[1] ||
        ops[1]->result(0).first_use().owner() != ops[2]) {
      return false;
    }

    const std::vector<int64_t> shape = [&] {
      auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
          fusion_op->GetParentProgram());
      const auto& reshape_x_shape =
          shape_analysis.GetShapeOrDataForValue(ops[1]->operand_source(0))
              .shape();
      const auto& reshape_out_shape =
          shape_analysis.GetShapeOrDataForValue(ops[1]->result(0)).shape();
      std::vector<int64_t> shape(reshape_out_shape.size(), -1);
      for (size_t i = 0; i < reshape_out_shape.size(); ++i) {
        if (reshape_out_shape[i].isa<int64_t>()) {
          shape[i] = reshape_out_shape[i].dyn_cast<int64_t>();
          continue;
        }
        if (reshape_out_shape[i].isa<std::string>()) {
          if (i < reshape_x_shape.size() &&
              reshape_x_shape[i] == reshape_out_shape[i]) {
            shape[i] = 0;
            continue;
          }
        }
        shape[i] = -1;
      }
      return shape;
    }();

    const int dynamic_dim_cnt = std::count(shape.begin(), shape.end(), -1);
    if (dynamic_dim_cnt > 1) {
      return false;
    }

    // create new reshape out of fusion op
    auto new_reshape = rewriter.Build<paddle::dialect::ReshapeOp>(
        ops[1]->operand_source(0), shape);

    rewriter.ReplaceAllUsesWith(fusion_op.result(0), new_reshape.result(0));
    rewriter.EraseOp(fusion_op);
    return true;
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
    ps.Add<FusionOpSingleReshapePattern>(context);

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
