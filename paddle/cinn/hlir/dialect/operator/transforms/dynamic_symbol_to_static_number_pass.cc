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

#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_symbol_to_static_number_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/common/dim_expr_simplify.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

PD_DEFINE_string(cinn_convert_dynamic_symbol_to_static_number);

namespace {

class DynamicToStaticConverter {
 public:
  using DimExpr4SymbolName = std::function<std::optional<symbol::DimExpr>(
      const std::string& symbol_name)>;
  explicit DynamicToStaticConverter(cinn::dialect::FusionOp fusion_op)
      : fusion_op_(fusion_op) {
    shape_analysis_ = &pir::ShapeAnalysisManager::Instance().Get(
        fusion_op_->GetParentProgram());
    DimExpr4SymbolName_ = InitDimExpr4SymbolName();
  }

  bool Convert() {
    if (!IsSymbolFullyInfered()) {
      return false;
    }
    bool updated = false;
    VisitEachValue(fusion_op_, [&](pir::Value value) {
      updated = updated || UpdateValueShape(value);
    });
    shape_analysis_->Init();
    return updated;
  }

 private:
  bool IsSymbolFullyInfered() {
    VisitEachValue(fusion_op_, [&](pir::Value value) {
      if (!shape_analysis_->HasShapeOrDataForValue(value)) {
        return false;
      }
    });
    return true;
  }

  DimExpr4SymbolName InitDimExpr4SymbolName() {
    // TODO(Hongyu Jia)
  }

  template <typename DoEachT>
  void VisitEachValue(cinn::dialect::FusionOp fusion_op,
                      const DoEachT& DoEach) {
    for (pir::Operation* op : fusion_op.GetOperators()) {
      for (pir::Value value : op->operands_source()) {
        DoEach(value);
      }
      for (pir::Value value : op->results()) {
        DoEach(value);
      }
    }
  }

  template <typename DoEachT>
  void VisitEachDimExpr(const std::vector<symbol::DimExpr>& dim_exprs,
                        const DoEachT& DoEach) {
    for (const auto& dim_expr : dim_exprs) {
      DoEach(dim_expr);
    }
  }

  std::vector<int> GetOriginValueShape(pir::Value value) {
    auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
    return ::common::vectorize<int>(dim);
  }

  std::vector<int> GetTargetValueShape(pir::Value value) {
    const auto& dynamic_shapes =
        shape_analysis_->GetShapeOrDataForValue(value).shape();
    std::vector<int> static_shapes{};
    VisitEachDimExpr(dynamic_shapes, [&](const symbol::DimExpr& dim_expr) {
      const auto& static_shape = cinn::common::SimplifyDimExpr(
          cinn::dialect::SubstituteDimExpr(dim_expr, DimExpr4SymbolName_));
      CHECK(static_shape.Has<std::int64_t>());
      static_shapes.push_back(
          static_cast<int>(static_shape.Get<std::int64_t>()));
    });
    return static_shapes;
  }

  bool UpdateValueShape(pir::Value value) {
    bool update = false;
    CHECK(shape_analysis_->HasShapeOrDataForValue(value));
    const auto& origin_shape = GetOriginValueShape(value);
    const auto& target_shape = GetTargetValueShape(value);
    CHECK_EQ(origin_shape.size(), target_shape.size());
    for (std::size_t i = 0; i < origin_shape.size(); ++i) {
      if (origin_shape.at(i) == -1) {
        CHECK_GT(target_shape.at(i), 0);
        update = true;
      } else {
        CHECK(origin_shape.at(i) == target_shape.at(i));
      }
    }
    // TODO(Hongyu Jia): update value's shape, origin_shape->target_shape
    return update;
  }

  cinn::dialect::FusionOp fusion_op_;
  pir::ShapeConstraintIRAnalysis* shape_analysis_;
  DimExpr4SymbolName DimExpr4SymbolName_{};
};

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    return DynamicToStaticConverter(fusion_op).Convert();
  }
};

class DynamicSymbolToStaticNumberPass : public pir::PatternRewritePass {
 public:
  DynamicSymbolToStaticNumberPass()
      : pir::PatternRewritePass("dynamic_symbol_to_static_number", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (FLAGS_cinn_convert_dynamic_symbol_to_static_number.empty()) {
      return false;
    }
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateDynamicSymbolToStaticNumberPass() {
  return std::make_unique<DynamicSymbolToStaticNumberPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, DivideGroupOpToFusionOpPass);
