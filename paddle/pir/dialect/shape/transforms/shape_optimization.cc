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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/pass_registry.h"

namespace pir {
namespace {
using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

bool InsertTieShapeOnValue(pir::Value value,
                           pir::Builder& builder) {  // NOLINT
  auto ty = value.type().dyn_cast<paddle::dialect::DenseTensorType>();

  if (!ty || ty.dims().size() == 0) return true;
  std::vector<pir::Value> dimSizes;
  for (int64_t dim = 0, rank = ty.dims().size(); dim < rank; ++dim) {
    auto dimOp = builder.Build<pir::dialect::TensorDimOp>(value, dim);
    dimSizes.push_back(dimOp.out());
  }
  builder.Build<pir::dialect::TieShapeOp>(value, dimSizes);
  return true;
}

bool InsertTieShapeOnRegion(pir::Region* region);

bool InsertTieShapeOnOperation(pir::Operation* op,
                               pir::Builder& builder) {  // NOLINT
  // TODO(zhangbo63): skip more specialized Ops.
  if (op->isa<pir::dialect::TieShapeOp>() || op->isa<pir::dialect::FuncOp>())
    return true;

  for (size_t i = 0; i < op->num_regions(); ++i) {
    if (!InsertTieShapeOnRegion(&(op->region(i)))) return false;
  }
  builder.SetInsertionPointAfter(op);
  for (pir::OpResult v : op->results()) {
    if (!InsertTieShapeOnValue(v, builder)) return false;
  }

  return true;
}

bool InsertTieShapeOnBlock(pir::Block* block) {
  pir::Builder builder =
      pir::Builder(pir::IrContext::Instance(), block, block->begin());
  // TODO(liujinnan): mapping block arguments

  std::vector<pir::Operation*> op_list;
  for (pir::Operation* op : *block) op_list.push_back(op);
  for (pir::Operation* op : op_list) {
    if (!InsertTieShapeOnOperation(op, builder)) return false;
  }
  return true;
}

bool InsertTieShapeOnRegion(pir::Region* region) {
  for (pir::Block* block : *region) {
    if (!InsertTieShapeOnBlock(block)) return false;
  }
  return true;
}

bool MaterializeShapeComputation(pir::ModuleOp m) {
  if (!InsertTieShapeOnRegion(&(m->region(0)))) return false;
  // TODO(liujinnan): add rewitter pattern for reifyInferShape.
  return true;
}

bool IsCandidateShapeTensorType(Type type) {
  auto tensor_type = type.dyn_cast<DenseTensorType>();
  auto shaped_type = tensor_type.dyn_cast<ShapedTypeInterface>();

  return (tensor_type && tensor_type && shaped_type.GetRank() == 1 &&
          shaped_type.HasStaticShape() &&
          shaped_type.GetElementType().IsIntOrIndex() &&
          shaped_type.GetShape()[0] < 32);
}

class ShapeComputationIRAnalysis {
 public:
  using func = std::function<bool(Operation* op)>;
  explicit ShapeComputationIRAnalysis(ModuleOp m,
                                      SymbolicDimMgr& mgr);  // NOLINT
  bool Run();

 private:
  bool RunOnRegion(Region* region, func fn);
  bool RunOnBlock(Block* block, func fn);
  bool RunOnOperation(Operation* op, func fn);

  bool BuildShapeOnOperation(Operation* op);
  bool BuildShapeOnValue(Value value);

  bool ApplyOpConstraint(Operation* op);
  bool ApplyIndexOpConstraint(Operation* op);
  bool ApplyTieShapeOpConstraint(Operation* op);

  bool initialized_ = false;
  ModuleOp m_;
  SymbolicDimMgr& mgr_;

  std::unordered_map<Value, SymbolicDim> value_to_sym_dim_;

  // shape tensor is the 1D ranked tensor with int/index dtype.
  std::unordered_map<Value, std::vector<SymbolicDim>> shape_tensor_to_sym_dims_;

  std::unordered_map<Value, std::vector<SymbolicDim>> dense_tensor_to_sym_dims_;
};

// Returns true if the type is possible to be a shape tensor type.
// Shape tensor type :
//    - rank-1 static-shaped tensor type
//    - element type of the tensor is int or index
//    - number of elements of the tensor < 32, supposing that the
//      higiest possible rank is smaller than 32.

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(ModuleOp m,
                                                       SymbolicDimMgr& mgr)
    : m_(m), mgr_(mgr) {}

bool ShapeComputationIRAnalysis::Run() {
  // Make sure only run once.
  if (initialized_) return false;
  initialized_ = true;
  auto build_shape_func =
      std::bind(&ShapeComputationIRAnalysis::BuildShapeOnOperation,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), build_shape_func)) return false;
  auto apply_op_constraint_func =
      std::bind(&ShapeComputationIRAnalysis::ApplyOpConstraint,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), apply_op_constraint_func)) return false;
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
    dense_tensor_to_sym_dims_[value] = std::move(symbols);
    return true;
  }
  for (size_t i = 0; i < op->num_results(); ++i) {
    if (!BuildShapeOnValue(op->result(i))) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::BuildShapeOnValue(Value value) {
  Type type = value.type();
  if (type.IsIntOrIndex()) {
    SymbolicDim sym = mgr_.NewSymbolicDim();
    value_to_sym_dim_[value] = sym;
  } else if (IsCandidateShapeTensorType(type)) {
    auto shaped_type = type.dyn_cast<ShapedTypeInterface>();
    std::vector<SymbolicDim> symbols;
    for (size_t i = 0, d = shaped_type.GetShape()[0]; i < d; ++i)
      symbols.push_back(mgr_.NewSymbolicDim());
    shape_tensor_to_sym_dims_[value] = std::move(symbols);
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
  if (!type.IsIntOrIndex()) return true;

  if (auto dim_op = op->dyn_cast<dialect::TensorDimOp>()) {
    int64_t dim_index = dim_op.index()
                            .dyn_cast<OpResult>()
                            .owner()
                            ->attribute<Int64Attribute>("value")
                            .data();
    value_to_sym_dim_[dim_op.out()].UpdateKnownNonNegative(true);
    if (!mgr_.MapSymbolicDimEqual(
            value_to_sym_dim_[dim_op.out()],
            dense_tensor_to_sym_dims_[dim_op.source()][dim_index])) {
      return false;
    }

  } else if (auto const_op = op->dyn_cast<ConstantOp>()) {
    int64_t val = const_op.value().dyn_cast<Int64Attribute>().data();
    if (!mgr_.MapSymbolicDimEqual(value_to_sym_dim_[op->result(0)],
                                  mgr_.NewConstantSymbolicDim(val))) {
      return false;
    }
  }
  // TODO(zhangbo63): add support for reifyInferShape. (e.g. mul/add)
  return true;
}

bool ShapeComputationIRAnalysis::ApplyTieShapeOpConstraint(Operation* op) {
  if (auto tie_shape = op->dyn_cast<dialect::TieShapeOp>()) {
    auto& value = dense_tensor_to_sym_dims_[op->operand_source(0)];
    for (size_t idx = 0; idx < tie_shape.dims().size(); ++idx) {
      if (!mgr_.MapSymbolicDimEqual(value_to_sym_dim_[tie_shape.dims()[idx]],
                                    value[idx]))
        return false;
      mgr_.GetRootSymbolicDim(value[idx]).UpdateKnownNonNegative(true);
    }
  }
  return true;
}

bool OptimizeShapeComputation(pir::ModuleOp m, PassPipelineRunner runner) {
  // TODO(liujinnan): Do some Canonicalizer.
  pir::SymbolicDimMgr mgr(m);
  IR_ENFORCE(mgr.Load(),
             "SymbolicDimMgr Load failed in OptimizeShapeComputation.");
  ShapeComputationIRAnalysis analysis(m, mgr);
  if (!analysis.Run()) {
    return false;
  }
  IR_ENFORCE(mgr.Save(),
             "SymbolicDimMgr save failed in OptimizeShapeComputation.");
  return true;
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization_pass", 0) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
    MaterializeShapeComputation(module_op);
    // runner is for Canonicalizer.
    PassPipelineRunner runner = [this](pir::PassManager& pm, pir::ModuleOp m) {
      return pm.Run(m.program());
    };
    if (!OptimizeShapeComputation(module_op, runner)) {
      return;
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(shape_optimization_pass, pir::ShapeOptimizationPass);
