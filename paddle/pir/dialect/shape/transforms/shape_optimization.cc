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

#include "paddle/pir/dialect/shape/transforms/shape_optimization.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"

namespace pir {

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
}  // namespace pir
