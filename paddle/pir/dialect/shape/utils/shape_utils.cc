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

#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include <string>
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
namespace pir {

bool ShapeAnalysis::IsSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  return IsProductEqual(lhs, 0, lhs_type.GetRank(), rhs, 0, rhs_type.GetRank());
}

bool ShapeAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(ModuleOp m)
    : m_(m), mgr_(m) {
  mgr_.Load();
  for (auto op : *(m_.block())) {
    auto tie_shape_op = op->dyn_cast<dialect::TieShapeOp>();
    if (!tie_shape_op) continue;
    Value result = tie_shape_op.value();
    auto& symbols = value_to_sym_dims_[result];
    auto attrs =
        tie_shape_op
            .attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName())
            .AsVector();
    for (const auto& attr : attrs) {
      auto sym_op = mgr_.symbolTable().Lookup<SymbolicDim>(
          attr.dyn_cast<StrAttribute>().AsString());
      if (!sym_op) continue;
      symbols.push_back(sym_op);
    }
  }
}

ShapeConstraintIRAnalysis::~ShapeConstraintIRAnalysis() { mgr_.Save(); }

bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  if (lhs_type.HasStaticShape() && rhs_type.HasStaticShape()) {
    return vectorize(lhs_type.GetShape()) == vectorize(rhs_type.GetShape());
  }

  auto lhs_it = value_to_sym_dims_.find(lhs);
  auto rhs_it = value_to_sym_dims_.find(rhs);

  if (lhs_it == value_to_sym_dims_.end() ||
      rhs_it == value_to_sym_dims_.end() ||
      lhs_it->second.size() != rhs_it->second.size())
    return false;

  std::vector<SymbolicDim> lhs_syms;
  std::vector<SymbolicDim> rhs_syms;
  for (auto sym : lhs_it->second) {
    lhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  for (auto sym : rhs_it->second) {
    rhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  return lhs_syms == rhs_syms;
}

bool ShapeConstraintIRAnalysis::IsProductEqual(Value lhs,
                                               std::vector<int> lhs_dim_idxs,
                                               Value rhs,
                                               std::vector<int> rhs_dim_idxs) {
  SymbolicDimProduct lhs_prod;
  SymbolicDimProduct rhs_prod;

  auto build_symbolic_dim_product =
      [&](SymbolicDimProduct& prod, Value value, std::vector<int> dim_idxs) {
        auto type = value.type().dyn_cast<ShapedTypeInterface>();
        auto it = value_to_sym_dims_.find(value);
        if (!type || !type.HasRank()) return false;
        for (int idx : dim_idxs) {
          if (type.GetShape()[idx] == ShapedTypeInterface::kDynamic) {
            if (it == value_to_sym_dims_.end() ||
                static_cast<int>(it->second.size()) <= idx)
              return false;
            prod.symbols.push_back(it->second[idx]);
          } else {
            prod.factor *= type.GetShape()[idx];
          }
        }
        return true;
      };

  if (!build_symbolic_dim_product(lhs_prod, lhs, lhs_dim_idxs) ||
      !build_symbolic_dim_product(rhs_prod, rhs, rhs_dim_idxs)) {
    return false;
  }

  return mgr_.IsSymbolicDimProductEqual(lhs_prod, rhs_prod);
}

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(ModuleOp m,
                                                       SymbolicDimMgr& mgr)
    : m_(m), mgr_(mgr) {}

bool ShapeComputationIRAnalysis::Run() {
  // Make sure only run once.
  if (initialized_) return false;
  initialized_ = true;
  auto buildShapeFunc =
      std::bind(&ShapeComputationIRAnalysis::BuildShapeOnOperation,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), buildShapeFunc)) return false;
  auto applyOpConstraintFunc =
      std::bind(&ShapeComputationIRAnalysis::ApplyOpConstraint,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), applyOpConstraintFunc)) return false;
  return true;
}

bool ShapeComputationIRAnalysis::RunOnRegion(Region* region, func fn) {
  for (Block* block : *region) {
    if (!RunOnBlock(block, fn)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::RunOnBlock(Block* block, func fn) {
  // TODO(liujinnan): mapping block arguments

  std::vector<Operation*> op_list;
  for (Operation* op : *block) op_list.push_back(op);
  for (Operation* op : op_list) {
    if (!RunOnOperation(op, fn)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::RunOnOperation(Operation* op, func fn) {
  for (size_t i = 0; i < op->num_regions(); ++i) {
    if (!RunOnRegion(&(op->region(i)), fn)) return false;
  }
  return fn(op);
}

bool ShapeComputationIRAnalysis::BuildShapeOnOperation(Operation* op) {
  if (op->isa<dialect::FuncOp>()) return true;
  if (op->isa<dialect::TieShapeOp>()) {
    Value value = op->operand_source(0);
    std::vector<SymbolicDim> symbols;
    if (op->HasAttribute(SymbolicDim::GetSymbolicDimAttrName())) {
      auto attrs =
          op->attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName())
              .AsVector();
      for (Attribute attr : attrs) {
        auto sym = mgr_.symbolTable().Lookup<SymbolicDim>(
            attr.dyn_cast<StrAttribute>().AsString());
        assert(sym);
        SymbolicDim root = mgr_.GetRootSymbolicDim(sym);
        symbols.push_back(root);
      }
    } else {
      symbols = mgr_.CreateSymbolicDimsForRankedValue(value);
      std::vector<Attribute> attrs;
      for (SymbolicDim sym : symbols) {
        Attribute rootSymbol =
            StrAttribute::get(m_->ir_context(), sym.GetSymName());
        attrs.push_back(rootSymbol);
      }
      op->set_attribute(SymbolicDim::GetSymbolicDimAttrName(),
                        ArrayAttribute::get(m_->ir_context(), attrs));
    }
    rankedTensor2SymDims_[value] = std::move(symbols);
    return true;
  }
  for (size_t i = 0; i < op->num_results(); ++i) {
    if (!BuildShapeOnValue(op->result(i))) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::BuildShapeOnValue(Value value) {
  Type type = value.type();
  if (IsIntOrIndex(type)) {
    SymbolicDim sym = mgr_.NewSymbolicDim();
    value2SymDim_[value] = sym;
  } else if (IsCandidateShapeTensorType(type)) {
    auto shapedTy = type.dyn_cast<ShapedTypeInterface>();
    std::vector<SymbolicDim> symbols;
    for (size_t i = 0, d = shapedTy.GetShape()[0]; i < d; ++i)
      symbols.push_back(mgr_.NewSymbolicDim());
    shapeTensor2SymDims_[value] = std::move(symbols);
  }
  return true;
}

bool ShapeComputationIRAnalysis::ApplyOpConstraint(Operation* op) {
  IR_ENFORCE(ApplyIndexOpConstraint(op),
             "Fail to apply constraint for index op");
  IR_ENFORCE(ApplyTieShapeOpConstraint(op),
             "Fail to apply constraint for tie_shape op");

  // TODO(zhangbo63): add more constraints
  return true;
}

bool ShapeComputationIRAnalysis::ApplyIndexOpConstraint(Operation* op) {
  if (op->num_results() == 0) return true;

  Type type = op->result(0).type();
  if (!IsIntOrIndex(type)) return true;

  if (auto dimOp = op->dyn_cast<dialect::TensorDimOp>()) {
    int64_t dimIndex = dimOp.index()
                           .dyn_cast<OpResult>()
                           .owner()
                           ->attribute<Int64Attribute>("value")
                           .data();
    value2SymDim_[dimOp.out()].UpdateKnownNonNegative(true);
    if (!mgr_.MapSymbolicDimEqual(
            value2SymDim_[dimOp.out()],
            rankedTensor2SymDims_[dimOp.source()][dimIndex])) {
      return false;
    }

  } else if (auto constOp = op->dyn_cast<ConstantOp>()) {
    int64_t val = constOp.value().dyn_cast<Int64Attribute>().data();
    if (!mgr_.MapSymbolicDimEqual(value2SymDim_[op->result(0)],
                                  mgr_.NewConstantSymbolicDim(val))) {
      return false;
    }
  }
  // TODO(zhangbo63): add support for reifyInferShape. (e.g. mul/add)
  return true;
}

bool ShapeComputationIRAnalysis::ApplyTieShapeOpConstraint(Operation* op) {
  if (auto tieShape = op->dyn_cast<dialect::TieShapeOp>()) {
    auto& value = rankedTensor2SymDims_[op->operand_source(0)];
    for (size_t idx = 0; idx < tieShape.dims().size(); ++idx) {
      if (!mgr_.MapSymbolicDimEqual(value2SymDim_[tieShape.dims()[idx]],
                                    value[idx]))
        return false;
      mgr_.GetRootSymbolicDim(value[idx]).UpdateKnownNonNegative(true);
    }
  }
  return true;
}

bool IsIntOrIndex(Type type) {
  return type.isa<IndexType>() || type.isa<Int8Type>() ||
         type.isa<UInt8Type>() || type.isa<Int16Type>() ||
         type.isa<Int32Type>() || type.isa<Int64Type>();
}

bool IsCandidateShapeTensorType(Type type) {
  if (auto tensorTy = type.dyn_cast<paddle::dialect::DenseTensorType>()) {
    auto shapedTy = tensorTy.dyn_cast<ShapedTypeInterface>();
    return (shapedTy.GetRank() == 1 && shapedTy.HasStaticShape() &&
            IsIntOrIndex(shapedTy.GetElementType()) &&
            shapedTy.GetShape()[0] < 32);
  }
  return false;
}

}  // namespace pir
