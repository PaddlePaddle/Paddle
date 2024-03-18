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

#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include <glog/logging.h>
#include <algorithm>
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

using ShapeOrDataDimExprs4ValueT =
    std::function<symbol::ShapeOrDataDimExprs(pir::Value)>;

std::vector<pir::Value> FindSourceDenseTensorOfDimTensor(
    pir::Value shape,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> ret{};
  const auto& Emplace = [&](pir::Value value) {
    if (std::find(ret.begin(), ret.end(), value) != ret.end()) return;
    ret.emplace_back(value);
  };
  const auto& ForEachInputValue =
      [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
        // find input dimension tensor;
        pir::Operation* owner = value.defining_op();
        if (owner == nullptr) return;
        for (auto input_value : pir::GetUsedExternalValue(*owner)) {
          Visit(input_value);
        }
      };
  const auto& IsDimTensorOrListDimExpr = symbol::Overloaded{
      [](const symbol::TensorShapeOrDataDimExprs& dim_expr) {
        return dim_expr.data().has_value();
      },
      [](const symbol::TensorListShapeOrDataDimExprs& dim_expr) {
        return true;
      }};
  // For TensorListShapeOrDataDimExprs case, we should recursivly visit its
  // each dim_expr, which is automatically in next step.
  const auto& NeedTrackUpstream = [&](pir::Value value) -> bool {
    const auto& sym_shape = ShapeOrDataDimExprs4Value(value);
    return std::visit(IsDimTensorOrListDimExpr, sym_shape.variant());
  };
  const auto& ForEachInputDimTensor =
      [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
        // find input dimension tensor;
        ForEachInputValue(value, [&](pir::Value input) {
          if (NeedTrackUpstream(input)) {
            Visit(input);
          }
        });
      };
  common::BfsWalker<pir::Value> walker(ForEachInputDimTensor);
  walker(shape, [&](pir::Value value) {
    size_t input_cnt = 0;
    ForEachInputValue(value, [&](pir::Value input) {
      ++input_cnt;
      if (NeedTrackUpstream(input)) return;
      Emplace(input);
    });
    if (input_cnt == 0) {
      // `value` is a result of a source op.
      Emplace(value);
    }
  });
  return ret;
}

bool MakeGenerateShapeOpAttribute(
    pir::IrContext* ir_context,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    pir::Value output_shape,
    const std::vector<pir::Value>& origin_inputs,
    std::vector<pir::Value>* minimal_inputs,
    std::vector<pir::Attribute>* output_dim_expr_attrs,
    GenerateShapeOp::SymbolBindings* symbol_bindings) {
  const auto& shape_or_data_dim_exprs = ShapeOrDataDimExprs4Value(output_shape);
  ExprVec data_vec =
      paddle::dialect::details::GetExprVecFromData(shape_or_data_dim_exprs);
  // CHECK(shape_or_data_dim_exprs.data().has_value());
  CHECK(data_vec.size());
  // const auto& out_dim_exprs = shape_or_data_dim_exprs.data().value();
  const auto& out_dim_exprs = data_vec;
  return MakeGenerateShapeOpAttribute(ir_context,
                                      ShapeOrDataDimExprs4Value,
                                      out_dim_exprs,
                                      origin_inputs,
                                      minimal_inputs,
                                      output_dim_expr_attrs,
                                      symbol_bindings);
}

std::unordered_set<pir::Operation*> GetOpSetFromOutputToInputsValue(
    const std::vector<pir::Value>& input_values, pir::Value output_value) {
  std::unordered_set<pir::Operation*> op_set;
  std::queue<pir::Operation*> op_queue;
  op_queue.push(output_value.defining_op());
  op_set.insert(output_value.defining_op());
  std::unordered_set<pir::Value> visited_input_value;
  const std::unordered_set<pir::Value> input_value_set(input_values.begin(),
                                                       input_values.end());
  while (!op_queue.empty()) {
    auto* op = op_queue.front();
    op_queue.pop();
    for (uint32_t i = 0; i < op->num_operands(); ++i) {
      pir::Value value = op->operand_source(i);
      if (input_value_set.count(value)) {
        visited_input_value.insert(value);
        continue;
      }
      if (!value || !value.type() || op_set.count(value.defining_op())) {
        continue;
      }
      op_queue.push(value.defining_op());
      op_set.insert(value.defining_op());
    }
  }
  return op_set;
}

std::vector<pir::Operation*> GetSubGraphFromOutputToInputsValue(
    const std::vector<pir::Value>& input_values, pir::Value output_value) {
  std::unordered_set<pir::Value> visited_value(input_values.begin(),
                                               input_values.end());
  auto HasVisitAllInputs = [&](pir::Operation* op) {
    for (uint32_t i = 0; i < op->num_operands(); ++i) {
      if (!visited_value.count(op->operand_source(i))) return false;
    }
    return true;
  };
  const std::unordered_set<pir::Operation*>& op_set =
      GetOpSetFromOutputToInputsValue(input_values, output_value);
  std::queue<pir::Operation*> op_queue;
  for (auto* op : op_set) {
    if (HasVisitAllInputs(op)) {
      op_queue.push(op);
    }
  }

  std::vector<pir::Operation*> ops;
  while (!op_queue.empty()) {
    auto* op = op_queue.front();
    op_queue.pop();
    ops.push_back(op);
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      visited_value.insert(op->result(i));
      for (auto iter = op->result(i).use_begin();
           iter != op->result(i).use_end();
           ++iter) {
        auto* use_op = iter->owner();
        if (op_set.count(use_op) && HasVisitAllInputs(use_op)) {
          op_queue.push(use_op);
        }
      }
    }
  }
  return ops;
}

void InferSymbolicShapeForSubgraph(
    const std::vector<pir::Operation*>& ops,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  for (auto* op : ops) {
    auto infer_symbolic_shape_interface =
        op->dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
    if (infer_symbolic_shape_interface) {
      infer_symbolic_shape_interface.InferSymbolicShape(shape_analysis);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
    }
  }
}

void UpdateLocalShapeAnalysis(
    const std::vector<pir::Value>& input_tensors,
    pir::Value shape,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& dim_expr_map,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  // init inputs value's dim expr
  auto CreateExprsByExprMap =
      [&](const std::vector<symbol::DimExpr>& dim_exprs) {
        std::vector<symbol::DimExpr> new_shape;
        new_shape.reserve(dim_exprs.size());
        for (const auto& dim_expr : dim_exprs) {
          auto iter = dim_expr_map.find(dim_expr);
          if (iter == dim_expr_map.end()) {
            new_shape.push_back(dim_expr);
          } else {
            new_shape.push_back(iter->second);
          }
        }
        return new_shape;
      };

  for (const auto& input_tensor : input_tensors) {
    const auto& shape_or_data = ShapeOrDataDimExprs4Value(input_tensor);
    std::vector<symbol::DimExpr> new_shape =
        CreateExprsByExprMap(shape_or_data.shape());
    if (shape_or_data.data()) {
      std::vector<symbol::DimExpr> new_data =
          CreateExprsByExprMap(shape_or_data.data().value());
      shape_analysis->SetShapeOrDataForValue(
          input_tensor, symbol::TensorShapeOrDataDimExprs(new_shape, new_data));
    } else {
      shape_analysis->SetShapeOrDataForValue(
          input_tensor, symbol::TensorShapeOrDataDimExprs(new_shape));
    }
  }
  // infer new symbol shape for shape value
  std::vector<pir::Operation*> sub_graph_ops =
      GetSubGraphFromOutputToInputsValue(input_tensors, shape);
  InferSymbolicShapeForSubgraph(sub_graph_ops, shape_analysis);
}

std::optional<pir::Value> GetOutOfRewrittenGenerateShapeOp(
    pir::Value shape,
    pir::PatternRewriter* rewriter,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> input_tensors =
      FindSourceDenseTensorOfDimTensor(shape, ShapeOrDataDimExprs4Value);
  if (input_tensors.empty()) return std::nullopt;
  const std::unordered_map<symbol::DimExpr, symbol::DimExpr> dim_expr_map =
      [&] {
        std::unordered_map<symbol::DimExpr, symbol::DimExpr> dim_expr_map;
        int64_t local_dim_expr_id = 0;
        for (auto input_tensor : input_tensors) {
          const auto& shape_or_data = ShapeOrDataDimExprs4Value(input_tensor);
          for (const auto& dim_expr : shape_or_data.shape()) {
            if (!dim_expr.isa<int64_t>() && dim_expr_map.count(dim_expr) == 0) {
              dim_expr_map[dim_expr] =
                  symbol::DimExpr("SS" + std::to_string(local_dim_expr_id++));
            }
          }
          if (shape_or_data.data()) {
            for (const auto& dim_expr : shape_or_data.data().value()) {
              if (!dim_expr.isa<int64_t>() &&
                  dim_expr_map.count(dim_expr) == 0) {
                dim_expr_map[dim_expr] =
                    symbol::DimExpr("SS" + std::to_string(local_dim_expr_id++));
              }
            }
          }
        }
        return dim_expr_map;
      }();

  const bool has_complex_dim_expr = [&]() {
    bool has_complex_dim_expr = false;
    for (const auto& kv : dim_expr_map) {
      if (!kv.first.isa<int64_t>() && !kv.first.isa<std::string>()) {
        has_complex_dim_expr = true;
        break;
      }
    }
    return has_complex_dim_expr;
  }();
  pir::ShapeConstraintIRAnalysis shape_analysis;
  if (has_complex_dim_expr) {
    UpdateLocalShapeAnalysis(input_tensors,
                             shape,
                             dim_expr_map,
                             ShapeOrDataDimExprs4Value,
                             &shape_analysis);
  }

  auto LocalDimExprs4Value = [&](pir::Value value) {
    if (has_complex_dim_expr) {
      return shape_analysis.GetShapeOrDataForValue(value);
    }
    return ShapeOrDataDimExprs4Value(value);
  };

  std::vector<pir::Attribute> output_dim_expr_attrs{};
  GenerateShapeOp::SymbolBindings symbol_bindings{};
  bool success = MakeGenerateShapeOpAttribute(rewriter->ir_context(),
                                              LocalDimExprs4Value,
                                              shape,
                                              /*origin inputs*/ input_tensors,
                                              /*minimal inputs*/ &input_tensors,
                                              &output_dim_expr_attrs,
                                              &symbol_bindings);
  if (!success) return std::nullopt;
  return rewriter
      ->Build<cinn::dialect::GenerateShapeOp>(
          input_tensors, output_dim_expr_attrs, symbol_bindings)
      .out();
}

bool ReplaceShapeOpsToGenerateShape(
    pir::Value shape_operand,
    pir::PatternRewriter* rewriter,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  if (shape_operand.defining_op()->isa<cinn::dialect::GenerateShapeOp>()) {
    return false;
  }
  auto ShapeOrDataDimExprs4Value =
      [&shape_analysis](
          pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    CHECK(shape_analysis->HasShapeOrDataForValue(value));
    return shape_analysis->GetShapeOrDataForValue(value);
  };
  std::optional<pir::Value> opt_generated_shape =
      GetOutOfRewrittenGenerateShapeOp(
          shape_operand, rewriter, ShapeOrDataDimExprs4Value);
  if (!opt_generated_shape.has_value()) return false;
  shape_analysis->SetShapeOrDataForValue(
      opt_generated_shape.value(), ShapeOrDataDimExprs4Value(shape_operand));
  rewriter->ReplaceAllUsesWith(shape_operand, opt_generated_shape.value());
  return true;
}

template <typename OP_TYPE>
bool ProcessOp(OP_TYPE op,
               pir::PatternRewriter* rewriter,
               pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return ReplaceShapeOpsToGenerateShape(
      op->operand_source(1), rewriter, shape_analysis);
}

}  // namespace

template <typename OPTYPE>
class FuseShapeOpsIntoGenerateShapeOpPattern
    : public pir::OpRewritePattern<OPTYPE> {
 public:
  explicit FuseShapeOpsIntoGenerateShapeOpPattern(pir::IrContext* context)
      : pir::OpRewritePattern<OPTYPE>(context) {}

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
    return ProcessOp(op, &rewriter, &shape_analysis);
  }
};

class FuseShapeOpsIntoGenerateShapeOpPass : public pir::PatternRewritePass {
 public:
  FuseShapeOpsIntoGenerateShapeOpPass()
      : pir::PatternRewritePass("fuse_shape_ops_into_generate_shape_op_pass",
                                1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::ExpandOp>>(
        context);
    ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::ReshapeOp>>(
        context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateFuseShapeOpsIntoGenerateShapeOpPass() {
  return std::make_unique<FuseShapeOpsIntoGenerateShapeOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
