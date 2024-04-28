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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include <string>
#include "paddle/common/bfs_walker.h"
#include "paddle/common/topo_walker.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/src/core/value_impl.h"

namespace pir {

static std::string GetValueId(Value val) {
  auto op_id = val.defining_op()->id();
  auto val_idx = val.dyn_cast<OpResult>().index();

  return val.defining_op()->name() + "_" + std::to_string(op_id) + "_rst_" +
         std::to_string(val_idx);
}

void InferSymbolicShapeContext::Init() {
  value_id_to_shape_or_data_.clear();
  next_sym_idx_ = 0;
  constraints_manager_.SetEqualCallbackFunc(
      [&](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
        return SubstituteDimExpr(lhs, rhs);
      });
}

const std::string InferSymbolicShapeContext::GetNextSymName() {
  return "S" + std::to_string(next_sym_idx_++);
}

bool InferSymbolicShapeContext::HasShapeOrDataForValue(Value val) const {
  if (!val) {
    return false;
  }
  return value_id_to_shape_or_data_.count(val.impl()->id()) > 0;
}

const symbol::ShapeOrDataDimExprs&
InferSymbolicShapeContext::GetShapeOrDataForValue(Value val) const {
  // TODO(Hongqing-work): define a default empty ShapeOrDataDimExprs
  if (!val) {
    static symbol::ShapeOrDataDimExprs empty{
        symbol::TensorShapeOrDataDimExprs{}};
    return empty;
  }
  if (!HasShapeOrDataForValue(val)) {
    PADDLE_THROW(phi::errors::Fatal(
        "Fail to GetShapeOrDataForValue on InferSymbolicShape!"));
  }

  return value_id_to_shape_or_data_.at(val.impl()->id());
}

void InferSymbolicShapeContext::SetStaticShapeForValue(Value val) {
  if (!val) {
    PADDLE_THROW(
        phi::errors::Fatal("Set static shape for null value is FOBBIDEN!"));
  }
  if (!val.type().isa<DenseTensorType>()) {
    PADDLE_THROW(phi::errors::Fatal(
        "Set static shape for non-tensor value is FOBBIDEN!"));
  }
  auto type_info = val.type().dyn_cast<pir::DenseTensorType>();
  std::vector<symbol::DimExpr> static_shape;
  for (int i = 0; i < type_info.dims().size(); ++i) {
    int dim = type_info.dims()[i];
    if (dim > 0) {
      static_shape.emplace_back(symbol::DimExpr{dim});
    } else {
      static_shape.emplace_back(GetNextSymName());
    }
  }
  SetShapeOrDataForValue(val, symbol::TensorShapeOrDataDimExprs(static_shape));
}

void InferSymbolicShapeContext::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
      symbol::SubstituteShapeOrData(shape_or_data, substitution_pattern_);
  auto iter = value_id_to_shape_or_data_.find(val.impl()->id());
  if (iter == value_id_to_shape_or_data_.end()) {
    value_id_to_shape_or_data_.emplace(val.impl()->id(),
                                       substituted_shape_or_data);
  } else {
    iter->second = substituted_shape_or_data;
  }
}

void InferSymbolicShapeContext::AddEqualCstr(const symbol::DimExpr& lhs,
                                             const symbol::DimExpr& rhs) {
  constraints_manager_.AddEqCstr(lhs, rhs);
}

bool InferSymbolicShapeContext::IsEqual(const symbol::DimExpr& lhs,
                                        const symbol::DimExpr& rhs) const {
  return constraints_manager_.IsEqual(lhs, rhs);
}

void InferSymbolicShapeContext::AddGreatThanOneCstr(
    const symbol::DimExpr& dim_expr) {
  constraints_manager_.AddGTOneCstr(dim_expr);
}

bool InferSymbolicShapeContext::IsGreatThanOne(
    const symbol::DimExpr& dim_expr) const {
  return constraints_manager_.IsGTOne(dim_expr);
}

void InferSymbolicShapeContext::AddBroadcastableCstr(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
  constraints_manager_.AddBroadcastableCstr(lhs, rhs);
}

bool InferSymbolicShapeContext::IsBroadcastable(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const {
  return constraints_manager_.IsBroadcastable(lhs, rhs);
}

namespace {

bool CanSubstituteInShapeAnalysis(const symbol::DimExpr& lhs,
                                  const symbol::DimExpr& rhs) {
  auto CanSubstitutePredictor = symbol::Overloaded{
      [](std::int64_t lhs, const auto& rhs) { return true; },
      [](const std::string& lhs, const std::string& rhs) { return true; },
      [](const std::string& lhs,
         const symbol::Broadcast<symbol::DimExpr>& rhs) { return true; },
      [](const auto& lhs, const auto& rhs) { return false; }};
  return std::visit(CanSubstitutePredictor, lhs.variant(), rhs.variant()) ||
         std::visit(CanSubstitutePredictor, rhs.variant(), lhs.variant());
}

}  // namespace

void InferSymbolicShapeContext::SubstituteDimExpr(
    const symbol::DimExpr& origin, const symbol::DimExpr& substituted) {
  if (!CanSubstituteInShapeAnalysis(origin, substituted)) return;

  substitution_pattern_[origin] = substituted;
  for (auto it = substitution_pattern_.begin();
       it != substitution_pattern_.end();
       it++) {
    if (it->second == origin) it->second = substituted;
  }

  for (auto it = value_id_to_shape_or_data_.begin();
       it != value_id_to_shape_or_data_.end();
       it++) {
    const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
        symbol::SubstituteShapeOrData(it->second, substitution_pattern_);
    it->second = substituted_shape_or_data;
  }
}

void InferSymbolicShapeContext::PrintShapeOrDatas() const {
  LOG(INFO) << "shape analysis : @" << this
            << " value_id_to_shape_or_data_ size : "
            << value_id_to_shape_or_data_.size();
  LOG(INFO) << "----------- ShapeOrData for Values ------------";
  for (const auto& [value_id, shape_or_data] : value_id_to_shape_or_data_) {
    LOG(INFO) << value_id << " : " << shape_or_data;
  }
}

void ShapeConstraintIRAnalysis::Init() { context_.Init(); }

const std::string ShapeConstraintIRAnalysis::GetNextSymName() {
  return context_.GetNextSymName();
}

void ShapeConstraintIRAnalysis::SetStaticShapeForValue(Value val) {
  context_.SetStaticShapeForValue(val);
}

void ShapeConstraintIRAnalysis::InferShapeOrDataForValue(Value val) {
  std::unordered_set<Operation*> subgraph_ops;
  std::vector<Operation*> start_ops;
  const auto& VisitNotInferedInputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (auto& operand : op->operands_source()) {
          if (operand.impl() && !context_.HasShapeOrDataForValue(operand)) {
            if (!operand.defining_op()) {
              SetStaticShapeForValue(operand);
            } else {
              Visit(operand.defining_op());
            }
          }
        }
      };

  ::common::BfsWalker<Operation*> build_subgraph_walker(VisitNotInferedInputOp);
  build_subgraph_walker(val.defining_op(), [&](Operation* op) {
    subgraph_ops.insert(op);
    bool has_prev_op = false;
    for (auto& operand : op->operands_source()) {
      if (operand.impl() && !context_.HasShapeOrDataForValue(operand)) {
        if (!operand.defining_op()) {
          SetStaticShapeForValue(operand);
        } else {
          has_prev_op = true;
        }
      }
    }
    if (!has_prev_op) {
      start_ops.emplace_back(op);
    }
  });

  const auto& VisitSubgraphInputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (auto& operand : op->operands_source()) {
          if (operand.impl() && subgraph_ops.count(operand.defining_op())) {
            Visit(operand.defining_op());
          }
        }
      };
  const auto& VisitSubgraphOutputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          for (auto iter = op->result(i).use_begin();
               iter != op->result(i).use_end();
               ++iter) {
            if (subgraph_ops.count(iter->owner())) {
              Visit(iter->owner());
            }
          }
        }
      };
  ::common::TopoWalker<Operation*> topo_infer_walker(VisitSubgraphInputOp,
                                                     VisitSubgraphOutputOp);

  topo_infer_walker(start_ops.begin(), start_ops.end(), [&](Operation* op) {
    auto infer_symbolic_shape_interface =
        op->dyn_cast<pir::InferSymbolicShapeInterface>();
    if (infer_symbolic_shape_interface) {
      infer_symbolic_shape_interface.InferSymbolicShape(&context_);
      for (auto& result_value : op->results()) {
        if (result_value && (!context_.HasShapeOrDataForValue(result_value))) {
          PADDLE_THROW(phi::errors::Fatal(op->name() +
                                          " HAS ERROR on InferSymbolicShape!"));
        }
      }
    } else {
      // TODO(Hongqing-work): throw it after the shape analysis reconstruct
      // is done.
      // PADDLE_THROW(phi::errors::Unimplemented(
      //     val.defining_op()->name() +
      //     " DOES NOT have InferSymbolicShapeInterface!"));
      LOG(WARNING) << op->name()
                   << " DOES NOT have InferSymbolicShapeInterface!";
      for (auto& result_value : op->results()) {
        if (result_value && (!context_.HasShapeOrDataForValue(result_value))) {
          SetStaticShapeForValue(result_value);
        }
      }
    }
  });
}

const symbol::ShapeOrDataDimExprs&
ShapeConstraintIRAnalysis::GetShapeOrDataForValue(Value val) {
  // TODO(Hongqing-work): define a default empty ShapeOrDataDimExprs
  if (!val) {
    static symbol::ShapeOrDataDimExprs empty{
        symbol::TensorShapeOrDataDimExprs{}};
    return empty;
  }
  if (!context_.HasShapeOrDataForValue(val)) {
    // backtrack to infer shape from defining op
    if (!val.defining_op()) {
      SetStaticShapeForValue(val);
    } else {
      VLOG(3) << "InferShapeOrDataForValue,  defining_op: "
              << val.defining_op()->name();
      InferShapeOrDataForValue(val);
    }
  }

  return context_.GetShapeOrDataForValue(val);
}

void ShapeConstraintIRAnalysis::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  context_.SetShapeOrDataForValue(val, shape_or_data);
}

bool ShapeConstraintIRAnalysis::IsEqual(const symbol::DimExpr& lhs,
                                        const symbol::DimExpr& rhs) const {
  return context_.IsEqual(lhs, rhs);
}

bool ShapeConstraintIRAnalysis::IsGreatThanOne(
    const symbol::DimExpr& dim_expr) const {
  return context_.IsGreatThanOne(dim_expr);
}

bool ShapeConstraintIRAnalysis::IsBroadcastable(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const {
  return context_.IsBroadcastable(lhs, rhs);
}

void ShapeConstraintIRAnalysis::PrintShapeOrDatas() const {
  context_.PrintShapeOrDatas();
}

// Currently, we only support TensorShapeOrDataDimExprs but not
// TensorListShapeOrDataDimExprs to compare the shape.
bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  PADDLE_ENFORCE_EQ(
      lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
          rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
      true,
      phi::errors::InvalidArgument(
          "Currently, IsShapeEqual only support TensorShapeOrDataDimExprs "
          "but not TensorListShapeOrDataDimExprs."));

  // For static shape, directly compare the shapes.
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    return lhs_type.GetShape() == rhs_type.GetShape();
  }

  // For dynamic shape, compare the symbolic dimensions.
  return lhs_shape_data.variant() == rhs_shape_data.variant();
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs,
    const std::vector<int>& lhs_dim_idxs,
    Value rhs,
    const std::vector<int>& rhs_dim_idxs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    int64_t lhs_product = 1;
    int64_t rhs_product = 1;
    for (int i : lhs_dim_idxs) {
      lhs_product *= lhs_type.GetShape()[i];
    }
    for (int i : rhs_dim_idxs) {
      rhs_product *= rhs_type.GetShape()[i];
    }
    return lhs_product == rhs_product;
  }

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  PADDLE_ENFORCE_EQ(
      lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
          rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
      true,
      phi::errors::InvalidArgument(
          "Currently, IsProductEqual only support TensorShapeOrDataDimExprs "
          "but not TensorListShapeOrDataDimExprs."));

  symbol::DimExpr lhs_product(1);
  symbol::DimExpr rhs_product(1);
  for (int i : lhs_dim_idxs) {
    lhs_product = lhs_product * lhs_shape_data.shape()[i];
  }
  for (int i : rhs_dim_idxs) {
    rhs_product = rhs_product * rhs_shape_data.shape()[i];
  }
  return symbol::SimplifyDimExpr(lhs_product) ==
         symbol::SimplifyDimExpr(rhs_product);
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

bool ShapeConstraintIRAnalysis::IsSameNumel(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    auto lhs_shape = lhs_type.GetShape();
    auto rhs_shape = rhs_type.GetShape();
    if (lhs_shape == rhs_shape) {
      return true;
    }
    return common::product(lhs_shape) == common::product(rhs_shape);
  }

  return IsProductEqual(lhs,
                        0,
                        static_cast<int>(lhs_type.GetRank()),
                        rhs,
                        0,
                        static_cast<int>(rhs_type.GetRank()));
}

symbol::DimExpr ShapeConstraintIRAnalysis::GetProductDimExpr(
    Value value, const std::vector<int>& dim_idxs) {
  // For static shape
  auto value_type = value.type().dyn_cast<ShapedTypeInterface>();
  if (value_type.IsStaticShape()) {
    int64_t product = 1;
    for (int i : dim_idxs) {
      product *= value_type.GetShape()[i];
    }
    return symbol::DimExpr{product};
  }

  // For dynamic shape
  const auto& shape_data = GetShapeOrDataForValue(value);
  symbol::DimExpr product{1};
  for (int i : dim_idxs) {
    product = product * shape_data.shape()[i];
  }
  return symbol::SimplifyDimExpr(product);
}

pir::PrintHooks ShapeConstraintIRAnalysis::PrintHook() {
  pir::PrintHooks print_hook;
  print_hook.op_print_hook = [&](Operation* op, IrPrinter& printer) {
    printer.IrPrinter::PrintOperation(op);
    printer.os << " { ";
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      if (context_.HasShapeOrDataForValue(op->result(i))) {
        printer.os << "(" << this->GetShapeOrDataForValue(op->result(i)) << ")";
      } else {
        printer.os << "()";
      }
      if (i < op->num_results() - 1) {
        printer.os << ", ";
      }
    }
    printer.os << " }";
    printer.os << "\t(op_" << op->id() << ")";
  };
  return print_hook;
}

ShapeAnalysisManager& ShapeAnalysisManager::Instance() {
  static ShapeAnalysisManager instance;
  return instance;
}

ShapeConstraintIRAnalysis& ShapeAnalysisManager::Get(pir::Program* program) {
  auto it = tables_.find(program->module_op().operation()->id());

  if (it == tables_.end()) {
    it = tables_
             .emplace(program->module_op().operation()->id(),
                      std::make_shared<ShapeConstraintIRAnalysis>())
             .first;
  }

  return *it->second;
}

}  // namespace pir
