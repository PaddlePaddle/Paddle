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

#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace pir {
namespace {

bool InsertTieShapeOnValue(pir::Value value,
                           pir::Builder& builder) {  // NOLINT
  // Insert TieShapeOp only for non-zero ranked tensor type.
  auto type = value.type().dyn_cast<DenseTensorType>();
  if (!type || type.dims().size() == 0) return true;

  std::vector<pir::Value> dim_sizes;
  for (int64_t dim = 0, rank = type.dims().size(); dim < rank; ++dim) {
    auto dim_op = builder.Build<shape::TensorDimOp>(value, dim);
    dim_sizes.push_back(dim_op.out());
  }
  builder.Build<shape::TieShapeOp>(value, dim_sizes);
  return true;
}

// Forward declaration
bool InsertTieShapeOnRegion(pir::Region* region);

bool InsertTieShapeOnOperation(pir::Operation* op,
                               pir::Builder& builder) {  // NOLINT
  // TODO(zhangbopd): skip more specialized Ops.
  if (op->isa<shape::TieShapeOp>() || op->isa<shape::FuncOp>()) return true;

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
  // TODO(zhangbopd): mapping block arguments

  std::vector<pir::Operation*> op_list;
  for (auto& op : *block) op_list.push_back(&op);
  for (pir::Operation* op : op_list) {
    if (!InsertTieShapeOnOperation(op, builder)) return false;
  }
  return true;
}

bool InsertTieShapeOnRegion(pir::Region* region) {
  for (auto& block : *region) {
    if (!InsertTieShapeOnBlock(&block)) return false;
  }
  return true;
}

// Convert:
//   %shape = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
// To:
//   %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//   %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//   %shape = tensor.from_elements %d0, %d1 : tensor<2xindex>
struct ExpandShapeOfOpPattern : public OpRewritePattern<shape::ShapeOfOp> {
  using OpRewritePattern<shape::ShapeOfOp>::OpRewritePattern;

  bool MatchAndRewrite(shape::ShapeOfOp op,
                       PatternRewriter& rewriter) const override {
    VLOG(3) << "Apply ExpandShapeOfOpPattern...";

    auto type = op.out().type().dyn_cast<pir::DenseTensorType>();

    if (!type || !type.dyn_cast<ShapedTypeInterface>().HasStaticShape() ||
        !type.dyn_cast<ShapedTypeInterface>().GetElementType().IsIndex())
      return false;

    std::vector<Value> dim_sizes;
    for (int dim = 0,
             rank = type.dyn_cast<ShapedTypeInterface>().GetDyShape()[0];
         dim < rank;
         ++dim) {
      dim_sizes.push_back(
          rewriter.Build<shape::TensorDimOp>(op.input(), dim).out());
    }
    rewriter.ReplaceOpWithNewOp<shape::FromElementsOp>(op, dim_sizes);
    return true;
  }
};

// Fold dim of an operation that implements the InferSymbolicShapeInterface
template <typename OpTy>
struct DimOfShapedTypeOpInterfacePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  bool MatchAndRewrite(OpTy dim_op, PatternRewriter& rewriter) const override {
    return true;
  }
};

using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

// Returns true if the type is possible to be a shape tensor type.
// Shape tensor type :
//    - rank-1 static-shaped tensor type
//    - element type of the tensor is int or index
//    - number of elements of the tensor < 32, supposing that the
//      higiest possible rank is smaller than 32.
bool IsCandidateShapeTensorType(Type type) {
  auto tensor_type = type.dyn_cast<DenseTensorType>();
  auto shaped_type = tensor_type.dyn_cast<ShapedTypeInterface>();

  return (tensor_type && tensor_type && shaped_type.GetRank() == 1 &&
          shaped_type.HasStaticShape() &&
          shaped_type.GetElementType().IsIntOrIndex() &&
          shaped_type.GetDyShape()[0] < 32);
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

  std::unordered_map<Value, SymbolicDimOp> value_to_sym_dim_;

  // shape tensor is the 1D ranked tensor with int/index dtype.
  std::unordered_map<Value, std::vector<SymbolicDimOp>>
      shape_tensor_to_sym_dims_;

  std::unordered_map<Value, std::vector<SymbolicDimOp>>
      dense_tensor_to_sym_dims_;
};

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(ModuleOp m,
                                                       SymbolicDimMgr& mgr)
    : m_(m), mgr_(mgr) {}

bool ShapeComputationIRAnalysis::Run() {
  // Make sure only run once.
  if (initialized_) return false;
  initialized_ = true;
  return true;
}

bool ShapeComputationIRAnalysis::RunOnRegion(Region* region, func fn) {
  for (auto& block : *region) {
    if (!RunOnBlock(&block, fn)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::RunOnBlock(Block* block, func fn) {
  // TODO(zhangbopd): mapping block arguments

  std::vector<Operation*> op_list;
  for (auto& op : *block) op_list.push_back(&op);
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
  if (op->isa<shape::FuncOp>()) return true;
  if (op->isa<shape::TieShapeOp>()) {
    Value value = op->operand_source(0);
    std::vector<SymbolicDimOp> symbols;
    if (op->HasAttribute(SymbolicDimOp::GetSymbolicDimAttrName())) {
      auto attrs =
          op->attribute<ArrayAttribute>(SymbolicDimOp::GetSymbolicDimAttrName())
              .AsVector();
      for (Attribute attr : attrs) {
        auto sym = mgr_.symbolTable().Lookup<SymbolicDimOp>(
            attr.dyn_cast<StrAttribute>().AsString());
        IR_ENFORCE(sym);
        SymbolicDimOp root = mgr_.GetRootSymbolicDim(sym);
        symbols.push_back(root);
      }
    } else {
      symbols = mgr_.CreateSymbolicDimsForRankedValue(value);
      std::vector<Attribute> attrs;
      for (SymbolicDimOp sym : symbols) {
        Attribute rootSymbol =
            StrAttribute::get(m_->ir_context(), sym.GetSymName());
        attrs.push_back(rootSymbol);
      }
      op->set_attribute(SymbolicDimOp::GetSymbolicDimAttrName(),
                        ArrayAttribute::get(m_->ir_context(), attrs));
    }
    dense_tensor_to_sym_dims_[value] = std::move(symbols);
    return true;
  }
  for (auto& result : op->results()) {
    if (!BuildShapeOnValue(result)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::BuildShapeOnValue(Value value) {
  Type type = value.type();
  if (type.IsIntOrIndex()) {
    SymbolicDimOp sym = mgr_.NewSymbolicDim();
    value_to_sym_dim_[value] = sym;
  } else if (IsCandidateShapeTensorType(type)) {
    auto shaped_type = type.dyn_cast<ShapedTypeInterface>();
    std::vector<SymbolicDimOp> symbols;
    for (size_t i = 0, d = shaped_type.GetDyShape()[0]; i < d; ++i)
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

  // TODO(zhangbopd): add more constraints
  return true;
}

bool ShapeComputationIRAnalysis::ApplyIndexOpConstraint(Operation* op) {
  if (op->num_results() == 0) return true;

  Type type = op->result(0).type();
  if (!type.IsIntOrIndex()) return true;

  if (auto dim_op = op->dyn_cast<shape::TensorDimOp>()) {
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
  // TODO(zhangbopd): add support for reifyInferShape. (e.g. mul/add)
  return true;
}

bool ShapeComputationIRAnalysis::ApplyTieShapeOpConstraint(Operation* op) {
  if (auto tie_shape = op->dyn_cast<shape::TieShapeOp>()) {
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
  // TODO(zhangbopd): Do some Canonicalizer.
  pir::SymbolicDimMgr mgr(m);

  ShapeComputationIRAnalysis analysis(m, mgr);
  if (!analysis.Run()) {
    return false;
  }

  return true;
}

void PrintProgram(pir::ModuleOp m, std::string mgs) {
  std::ostringstream print_stream;
  print_stream << "\n\n";
  m.program()->Print(print_stream);
  print_stream << "\n\n";
  VLOG(3) << "===================== " << mgs << " =====================\n"
          << print_stream.str();
}

void DebugPrintOpInfo(
    pir::Operation* op,
    pir::ShapeConstraintIRAnalysis* shape_analysis = nullptr) {
  for (auto& res : op->results()) {
    std::ostringstream print_stream;

    print_stream << "result(" << res.index() << ") "
                 << "ShapeOrData: ";

    if (shape_analysis != nullptr) {
      auto shape_data = shape_analysis->value_to_shape_or_data_[res];
      print_stream << "shape: [";

      for (size_t i = 0; i < shape_data.shape().size(); ++i) {
        if (i != shape_data.shape().size() - 1) {
          print_stream << symbol::ToString(shape_data.shape()[i]) << ",";
        } else {
          print_stream << symbol::ToString(shape_data.shape()[i]);
        }
      }

      print_stream << "], data: [";
      if (shape_data.data().has_value()) {
        for (size_t i = 0; i < shape_data.data().value().size(); ++i) {
          if (i != shape_data.data().value().size() - 1) {
            print_stream << symbol::ToString(shape_data.data().value()[i])
                         << ",";
          } else {
            print_stream << symbol::ToString(shape_data.data().value()[i]);
          }
        }
      } else {
        print_stream << "nullopt";
      }

      print_stream << "]\n";
    }
    VLOG(3) << print_stream.str();
  }
}

void InferSymExprForAllValues(ModuleOp module_op) {
  auto shape_analysis_mgr = ShapeAnalysisManager::Instance();
  ShapeConstraintIRAnalysis& shape_analysis =
      shape_analysis_mgr.Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (auto& block : module_op->region(i)) {
      for (auto& op : block) {
        auto infer_symbolic_shape_interface =
            op.dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
        if (infer_symbolic_shape_interface) {
          VLOG(3) << op.name() << " has InferSymbolicShapeInterface.";
          PADDLE_ENFORCE(infer_symbolic_shape_interface.InferSymbolicShape(
              &shape_analysis));
        }
        DebugPrintOpInfo(&op, &shape_analysis);
      }
    }
  }
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization_pass", 0) {}

  void Run(pir::Operation* op) override {
    VLOG(3) << "===================== ShapeOptimizationPass Run start... "
               "=============================";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
    PrintProgram(module_op, "Origin Program");

    InferSymExprForAllValues(module_op);
    // Runner is for Canonicalizer.
    PassPipelineRunner runner = [this](pir::PassManager& pm, pir::ModuleOp m) {
      return pm.Run(m.program());
    };
    VLOG(3) << "===================== ShapeOptimizationPass Run End. "
               "=============================";
    PrintProgram(module_op, "ShapeOptimizationPass Program");
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
