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
#include "paddle/cinn/common/topo_walker.h"
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
  const auto& MayContainDimData = ::common::Overloaded{
      [](const symbol::TensorShapeOrDataDimExprs& dim_expr) {
        return dim_expr.data().has_value();
      },
      [](const symbol::TensorListShapeOrDataDimExprs& dim_expr) {
        return true;
      },
      [](const symbol::RankedTensorArrayShapeOrDataDimExprs& dim_expr) {
        return false;
      },
      [](const symbol::NullShapeOrDataDimExpr& null_shape_or_data) {
        return false;
      }};
  // For TensorListShapeOrDataDimExprs case, we should recursively visit its
  // each dim_expr, which is automatically in next step.
  const auto& NeedTrackUpstream = [&](pir::Value value) -> bool {
    const auto& sym_shape = ShapeOrDataDimExprs4Value(value);
    return std::visit(MayContainDimData, sym_shape.variant());
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
  if (!paddle::dialect::details::HasCompleteData(shape_or_data_dim_exprs)) {
    LOG(WARNING) << "The output_shape has no data.";
    return false;
  }
  ExprVec data_vec =
      paddle::dialect::details::GetExprVecFromData(shape_or_data_dim_exprs);
  // CHECK(shape_or_data_dim_exprs.data().has_value());
  PADDLE_ENFORCE_GT(
      data_vec.size(),
      0,
      ::common::errors::PreconditionNotMet("The data_vec must not be empty."));
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
  const std::unordered_set<pir::Value> input_value_set(input_values.begin(),
                                                       input_values.end());
  auto VisitNextOp = [&](pir::Operation* node,
                         const std::function<void(pir::Operation*)>& Visit) {
    for (uint32_t i = 0; i < node->num_operands(); ++i) {
      pir::Value in_value = node->operand_source(i);
      if (!in_value || !in_value.type()) continue;
      if (input_value_set.count(in_value)) continue;
      if (op_set.count(in_value.defining_op())) continue;

      Visit(in_value.defining_op());
    }
  };
  common::BfsWalker<pir::Operation*> walker(VisitNextOp);
  walker(output_value.defining_op(), [&](pir::Operation* op) {
    if (!op) return;
    op_set.insert(op);
  });
  return op_set;
}

std::vector<pir::Operation*> GetSubGraphFromOutputToInputsValue(
    const std::vector<pir::Value>& input_values, pir::Value output_value) {
  const std::unordered_set<pir::Operation*>& op_set =
      GetOpSetFromOutputToInputsValue(input_values, output_value);
  auto VisitUpstreamOp =
      [&](pir::Operation* node,
          const std::function<void(pir::Operation*)>& Visit) {
        for (uint32_t i = 0; i < node->num_operands(); ++i) {
          pir::Value in_value = node->operand_source(i);
          if (!in_value || !in_value.type()) continue;
          if (in_value.defining_op() == nullptr) continue;
          if (op_set.count(in_value.defining_op()) == 0) continue;
          Visit(in_value.defining_op());
        }
      };
  auto VisitDownstreamOp =
      [&](pir::Operation* node,
          const std::function<void(pir::Operation * node)>& Visit) {
        for (uint32_t i = 0; i < node->num_results(); ++i) {
          for (auto iter = node->result(i).use_begin();
               iter != node->result(i).use_end();
               ++iter) {
            if (op_set.count(iter->owner())) {
              Visit(iter->owner());
            }
          }
        }
      };
  common::TopoWalker<pir::Operation*> walker(VisitUpstreamOp,
                                             VisitDownstreamOp);

  const std::vector<pir::Operation*> input_ops = [&] {
    const std::unordered_set<pir::Value> input_value_set(input_values.begin(),
                                                         input_values.end());
    auto IsInputOp = [&](pir::Operation* op) {
      for (uint32_t i = 0; i < op->num_operands(); ++i) {
        if (input_value_set.count(op->operand_source(i)) == 0) {
          return false;
        }
      }
      return true;
    };
    std::vector<pir::Operation*> input_ops;
    for (auto* op : op_set) {
      if (IsInputOp(op)) {
        input_ops.push_back(op);
      }
    }
    return input_ops;
  }();
  std::vector<pir::Operation*> ops;
  walker(input_ops.begin(), input_ops.end(), [&](pir::Operation* node) {
    if (!node) return;
    ops.push_back(node);
  });
  return ops;
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
  auto out_type = [&]() -> pir::Type {
    if (shape.type().isa<paddle::dialect::DenseTensorType>()) {
      return shape.type();
    }
    return paddle::dialect::DenseTensorType::get(
        rewriter->ir_context(),
        pir::Int64Type::get(rewriter->ir_context()),
        ::common::make_ddim({output_dim_expr_attrs.size()}));
  }();
  return rewriter
      ->Build<cinn::dialect::GenerateShapeOp>(
          input_tensors, output_dim_expr_attrs, symbol_bindings, out_type)
      .out();
}

bool ReplaceShapeOpsToGenerateShape(
    pir::OpOperand shape_operand,
    pir::PatternRewriter* rewriter,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  auto* shape_def_op = shape_operand.source().defining_op();
  if (!shape_def_op || shape_def_op->isa<cinn::dialect::GenerateShapeOp>()) {
    return false;
  }
  auto ShapeOrDataDimExprs4Value =
      [&shape_analysis](
          pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    return shape_analysis->GetShapeOrDataForValue(value);
  };
  std::optional<pir::Value> opt_generated_shape =
      GetOutOfRewrittenGenerateShapeOp(
          shape_operand.source(), rewriter, ShapeOrDataDimExprs4Value);
  if (!opt_generated_shape.has_value()) return false;
  // Replace the shape op input only, don't replace other users of the shape
  // operand.
  shape_operand.set_source(opt_generated_shape.value());
  return true;
}

template <typename OP_TYPE>
bool ProcessOp(OP_TYPE op,
               pir::PatternRewriter* rewriter,
               pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return ReplaceShapeOpsToGenerateShape(
      op->operand(1), rewriter, shape_analysis);
}

template <>
bool ProcessOp<paddle::dialect::SliceOp>(
    paddle::dialect::SliceOp op,
    pir::PatternRewriter* rewriter,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return ReplaceShapeOpsToGenerateShape(
             op->operand(1), rewriter, shape_analysis) &&
         ReplaceShapeOpsToGenerateShape(
             op->operand(2), rewriter, shape_analysis);
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

class FuseSingleElementShapeOpsIntoGenerateShapeOpPattern
    : public pir::RewritePattern {
 public:
  explicit FuseSingleElementShapeOpsIntoGenerateShapeOpPattern(
      pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  bool Match(pir::Operation* op) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
    if (!IsSingleElementShapeOp(op, &shape_analysis)) return false;
    if (op->isa<cinn::dialect::GenerateShapeOp>()) return false;

    // all user op's output should has no data of shape expr
    pir::Value output = op->result(0);
    if (output.use_empty()) return false;
    for (auto iter = output.use_begin(); iter != output.use_end(); ++iter) {
      auto* user = iter->owner();
      if (IsSingleElementShapeOp(user, &shape_analysis)) return false;
      if (user->isa<cinn::dialect::GenerateShapeOp>()) return false;
      if (user->isa<pir::ShadowOutputOp>()) return false;
    }

    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    auto ShapeOrDataDimExprs4Value =
        [&shape_analysis](
            pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
      return shape_analysis.GetShapeOrDataForValue(value);
    };
    std::optional<pir::Value> opt_generated_shape =
        GetOutOfRewrittenGenerateShapeOp(
            op->result(0), &rewriter, ShapeOrDataDimExprs4Value);
    if (!opt_generated_shape.has_value()) {
      LOG(WARNING) << "Create GenerateShapeOp Failed.";
      return;
    }

    rewriter.ReplaceAllUsesWith(op->result(0), opt_generated_shape.value());

    if (op->use_empty()) {
      rewriter.EraseOp(op);
    }
  }

 private:
  bool IsSingleElementShapeOp(
      pir::Operation* op,
      pir::ShapeConstraintIRAnalysis* shape_analysis) const {
    if (op->num_operands() == 0) return false;
    if (op->num_results() != 1) return false;

    pir::Value output = op->result(0);
    const auto& out_shape = shape_analysis->GetShapeOrDataForValue(output);
    if (!out_shape.isa<symbol::TensorShapeOrDataDimExprs>()) return false;
    if (!out_shape.data().has_value()) return false;

    auto dtype =
        output.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
    if (!dtype.isa<pir::Int32Type>() && !dtype.isa<pir::Int64Type>()) {
      return false;
    }

    // Only process the op which output is a single element
    return out_shape.data()->size() == 1;
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
    ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::Reshape_Op>>(
        context);
    ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::SliceOp>>(
        context);
    ps.Add<FuseSingleElementShapeOpsIntoGenerateShapeOpPattern>(context);
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
