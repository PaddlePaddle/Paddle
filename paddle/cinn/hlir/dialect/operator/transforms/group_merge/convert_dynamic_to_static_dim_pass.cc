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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_dynamic_to_static_dim_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

PD_DECLARE_string(cinn_convert_dynamic_dim_to_static_dim);

namespace {

template <typename DoEachT>
void ForEachRawDynamicToStaticDimPair(const DoEachT& DoEach) {
  const std::string& env_var = FLAGS_cinn_convert_dynamic_dim_to_static_dim;
  size_t start = 0;
  while (true) {
    size_t end = env_var.find(",", start);
    DoEach(env_var.substr(start, end));
    if (end == std::string::npos) return;
    start = end + 1;
  }
}

std::optional<std::pair<std::string, int64_t>> ParseRawDynamicToStaticDimPair(
    const std::string& raw_pair) {
  size_t pos = raw_pair.find(":", 0);
  if (pos == std::string::npos) return std::nullopt;
  std::int64_t constant = 0;
  try {
    constant = std::stoll(raw_pair.substr(pos + 1, -1), nullptr);
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
  if (constant <= 0) return std::nullopt;
  std::string symbol = raw_pair.substr(0, pos);
  if (symbol == "") return std::nullopt;
  const auto& IsWordOrUnderLine = [&](const char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_');
  };
  const auto& IsDigit = [&](const char ch) { return ch >= '0' && ch <= '9'; };
  if (!IsWordOrUnderLine(symbol[0])) return std::nullopt;
  for (int i = 1; i < symbol.size(); ++i) {
    if (!(IsWordOrUnderLine(symbol[i]) || IsDigit(symbol[i])))
      return std::nullopt;
  }
  return std::pair{symbol, int64_t{constant}};
}

std::unordered_map<std::string, int64_t> GetDynamicToStaticDimFlag() {
  std::unordered_map<std::string, int64_t> map;
  ForEachRawDynamicToStaticDimPair([&](const std::string& raw_pair) {
    if (auto pair = ParseRawDynamicToStaticDimPair(raw_pair)) {
      map.insert(pair.value());
    }
  });
  return map;
}

using GlobalDynamicToStaticDimMapT = std::unordered_map<std::string, int64_t>;

std::optional<GlobalDynamicToStaticDimMapT> CalcGlobalDynamicToStaticDimMap() {
  GlobalDynamicToStaticDimMapT map = GetDynamicToStaticDimFlag();
  if (map.empty()) return std::nullopt;
  return map;
}

const std::optional<GlobalDynamicToStaticDimMapT>*
GetGlobalDynamicToStaticDimMap() {
  static std::optional<GlobalDynamicToStaticDimMapT> map(
      CalcGlobalDynamicToStaticDimMap());
  return &map;
}

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
    bool updated = false;
    VisitEachValue(fusion_op_, [&](pir::Value value) {
      updated |= UpdateValueShape(value);
    });
    shape_analysis_->InitInferContext();
    return updated;
  }

 private:
  DimExpr4SymbolName InitDimExpr4SymbolName() {
    const auto* map = GetGlobalDynamicToStaticDimMap();
    PADDLE_ENFORCE_EQ(
        map->has_value(),
        true,
        ::common::errors::InvalidArgument("The map must have a value."));
    return
        [map](
            const std::string& symbol_name) -> std::optional<symbol::DimExpr> {
          PADDLE_ENFORCE_NE(map->value().find(symbol_name),
                            map->value().end(),
                            ::common::errors::InvalidArgument(
                                "The symbol '%s' must be present in the map.",
                                symbol_name.c_str()));
          return map->value().at(symbol_name);
        };
  }

  template <typename DoEachT>
  void VisitEachValue(cinn::dialect::FusionOp fusion_op,
                      const DoEachT& DoEach) {
    for (pir::Operation* op : fusion_op.GetOperators()) {
      for (std::size_t i = 0; i < op->num_operands(); ++i) {
        DoEach(op->operand_source(i));
      }
      for (std::size_t i = 0; i < op->num_results(); ++i) {
        DoEach(op->result(i));
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

  std::vector<std::int64_t> GetOriginValueShape(pir::Value value) {
    auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
    return ::common::vectorize(dim);
  }

  std::vector<std::int64_t> GetTargetValueShape(pir::Value value) {
    const auto& dynamic_shapes =
        shape_analysis_->GetShapeOrDataForValue(value).shape();
    std::vector<std::int64_t> static_shapes{};
    VisitEachDimExpr(dynamic_shapes, [&](const symbol::DimExpr& dim_expr) {
      const auto& static_shape = symbol::SimplifyDimExpr(
          cinn::dialect::SubstituteDimExpr(dim_expr, DimExpr4SymbolName_));
      PADDLE_ENFORCE_EQ(static_shape.Has<std::int64_t>(),
                        true,
                        ::common::errors::InvalidArgument(
                            "The static_shape must have an int64_t type."));
      static_shapes.push_back(static_shape.Get<std::int64_t>());
    });
    return static_shapes;
  }

  bool UpdateValueShape(pir::Value value) {
    bool update = false;
    const auto& origin_shape = GetOriginValueShape(value);
    const auto& target_shape = GetTargetValueShape(value);
    PADDLE_ENFORCE_EQ(
        origin_shape.size(),
        target_shape.size(),
        ::common::errors::InvalidArgument(
            "The size of origin shape and target shape is not equal,"
            "where the size of origin shape:%d but the size of target "
            "shape:%d.",
            origin_shape.size(),
            target_shape.size()));
    for (std::size_t i = 0; i < origin_shape.size(); ++i) {
      if (origin_shape.at(i) == -1) {
        PADDLE_ENFORCE_GT(target_shape.at(i),
                          0,
                          ::common::errors::InvalidArgument(
                              "The size of target shape is incorrect."
                              "Expected size is larger than 0, but receive %d.",
                              target_shape.at(i)));
        update = true;
      } else {
        PADDLE_ENFORCE_EQ(
            origin_shape.at(i),
            target_shape.at(i),
            ::common::errors::InvalidArgument(
                "The shape at index %d must be equal in both origin_shape and "
                "target_shape, but got origin_shape[%d] = %d and "
                "target_shape[%d] = %d.",
                i,
                i,
                origin_shape.at(i),
                i,
                target_shape.at(i)));
      }
    }
    if (update) {
      const auto& origin_type = value.type().dyn_cast<::pir::DenseTensorType>();
      pir::DenseTensorType target_type =
          pir::DenseTensorType::get(pir::IrContext::Instance(),
                                    origin_type.dtype(),
                                    ::common::make_ddim(target_shape),
                                    origin_type.data_layout(),
                                    origin_type.lod(),
                                    origin_type.offset());
      value.set_type(target_type);
      VLOG(4) << "DynamicToStaticConverter update Value: "
              << std::hash<pir::Value>()(value);
    }
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

class ConvertDynamicToStaticDimPass : public pir::PatternRewritePass {
 public:
  ConvertDynamicToStaticDimPass()
      : pir::PatternRewritePass("convert_dynamic_to_static_dim_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::optional<std::unique_ptr<::pir::Pass>>
CreateConvertDynamicToStaticDimPass() {
  if (!GetGlobalDynamicToStaticDimMap()->has_value()) return std::nullopt;
  return std::make_unique<ConvertDynamicToStaticDimPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
