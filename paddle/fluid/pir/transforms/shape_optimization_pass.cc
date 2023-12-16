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
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/interface/reify_infer_shape.h"
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

namespace {

void InferUnaryElementwiseSymbolicShape(
    const pir::Operation& op,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  auto input = op.operand_source(0);
  auto output = op.result(0);
  const auto& in_sym_dims =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(input);
  const auto& out_sym_dims =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(output);
  pir::SymbolicDimMgr& sym_dim_mgr = shape_analysis->symbolicDimMgr();
  for (auto i = 0; i < out_sym_dims.size(); ++i) {
    if (in_sym_dims[i].IsDynamic() || out_sym_dims[i].IsDynamic()) {
      sym_dim_mgr.MapSymbolicDimEqual(in_sym_dims[i], out_sym_dims[i]);
    } else {
      // do nothing
    }
  }
}

// TODO(zyfncg): support broadcast for elementwise ops.
void InferBinaryElementwiseSymbolicShape(
    const pir::Operation& op,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  auto input0 = op.operand_source(0);
  auto input1 = op.operand_source(1);
  auto output = op.result(0);
  const auto& in_sym_dims0 =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(input0);
  const auto& in_sym_dims1 =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(input1);
  const auto& out_sym_dims =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(output);
  pir::SymbolicDimMgr& sym_dim_mgr = shape_analysis->symbolicDimMgr();
  for (auto i = 0; i < out_sym_dims.size(); ++i) {
    if (in_sym_dims0[i].IsDynamic() || in_sym_dims1[i].IsDynamic() ||
        out_sym_dims[i].IsDynamic()) {
      sym_dim_mgr.MapSymbolicDimEqual(in_sym_dims0[i], out_sym_dims[i]);
      sym_dim_mgr.MapSymbolicDimEqual(in_sym_dims1[i], out_sym_dims[i]);
    } else {
      // do nothing
    }
  }
}

class InferSymbolicShapePass : public pir::Pass {
 public:
  InferSymbolicShapePass(
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::Pass("infer_symbolic_shape_pass", /*opt_level=*/1),
        shape_analysis_(shape_analysis) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "infer_symbolic_shape_pass should run on module op.");

    for (auto& op : module_op.block()) {
      if (op.isa<cinn::dialect::GroupOp>()) {
        for (auto* local_op : op.dyn_cast<cinn::dialect::GroupOp>().ops()) {
          InferSymbolicShape(*local_op);
        }
      } else {
        InferSymbolicShape(op);
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  typedef void (*InferSymShapeFunc)(
      const pir::Operation&,
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>&);
  void InferSymbolicShape(const pir::Operation& op) {
    thread_local static std::unordered_map<std::string, InferSymShapeFunc>
        infer_sym_shape_map(GetInferSymShapeMap());
    auto it = infer_sym_shape_map.find(op.name());

    if (it != infer_sym_shape_map.end()) {
      it->second(op, shape_analysis_);
    } else {
      LOG(WARNING) << "[" << op.name()
                   << "] is not supported for infer_symbolic_shape pass.";
    }
  }

  static std::unordered_map<std::string, InferSymShapeFunc>
  GetInferSymShapeMap() {
    return std::unordered_map<std::string, InferSymShapeFunc>{
        {paddle::dialect::ExpOp::name(), &InferUnaryElementwiseSymbolicShape},
        {paddle::dialect::SubtractOp::name(),
         &InferBinaryElementwiseSymbolicShape}};
  }

  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateInferSymbolicShapePass(
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  return std::make_unique<InferSymbolicShapePass>(shape_analysis);
}

}  // namespace pir

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

// Fold dim of an operation that implements the ReifyInferShapeInterface
template <typename OpTy>
struct DimOfShapedTypeOpInterfacePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  bool MatchAndRewrite(OpTy dim_op, PatternRewriter& rewriter) const override {
    OpResult dim_value = dim_op.source().template dyn_cast<OpResult>();
    if (!dim_value) return false;

    auto shaped_type_op =
        dim_value.owner()
            ->dyn_cast<paddle::dialect::ReifyInferShapeInterface>();
    if (!shaped_type_op) return false;

    std::optional<int64_t> dim_index = dim_op.GetConstantIndex();
    if (!dim_index) return false;

    std::vector<Value> reified_result_shapes;
    if (!shaped_type_op.ReifyInferShape(
            rewriter, shaped_type_op->operands(), reified_result_shapes))
      return false;

    if (reified_result_shapes.size() != shaped_type_op->num_results())
      return false;

    Value result_shape = reified_result_shapes[dim_value.index()];
    auto result_shape_type = result_shape.type().dyn_cast<DenseTensorType>();
    auto shaped_type = result_shape_type.dyn_cast<ShapedTypeInterface>();
    if (!result_shape_type || !shaped_type.GetElementType().IsIntOrIndex())
      return false;

    // TODO(zhangbopd): BuildOrFold required.
    std::vector<Value> indices;
    indices.push_back(rewriter.Build<shape::ConstantIndexOp>(*dim_index).out());

    Value new_value =
        rewriter.Build<shape::ExtractOp>(result_shape, indices).out();

    if (!new_value.type().isa<IndexType>())
      new_value =
          rewriter.Build<shape::IndexCastOp>(rewriter.index_type(), new_value)
              .out();

    rewriter.ReplaceOp(dim_op, {new_value});
    return true;
  }
};

bool MaterializeShapeComputation(pir::ModuleOp m) {
  // if (!InsertTieShapeOnRegion(&(m->region(0)))) return false;
  // TODO(zhangbopd): add rewitter pattern for reifyInferShape.
  RewritePatternSet patterns(m.ir_context());

  patterns.Add<ExpandShapeOfOpPattern,
               DimOfShapedTypeOpInterfacePattern<shape::TensorDimOp>>(
      patterns.ir_context());

  IR_ENFORCE(ApplyPatternsGreedily(m, std::move(patterns)).first,
             "fail to materialize shape computation\n");
  return true;
}

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
  // auto build_shape_func =
  //     std::bind(&ShapeComputationIRAnalysis::BuildShapeOnOperation,
  //               this,
  //               std::placeholders::_1);
  // if (!RunOnRegion(&(m_->region(0)), build_shape_func)) return false;
  // auto apply_op_constraint_func =
  //     std::bind(&ShapeComputationIRAnalysis::ApplyOpConstraint,
  //               this,
  //               std::placeholders::_1);
  // // TODO(zhangbopd): Delete the following 1 line and fix UT
  // // `shape_optimization_test`
  // return true;
  // if (!RunOnRegion(&(m_->region(0)), apply_op_constraint_func)) return false;
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
    // Runner is for Canonicalizer.
    PassPipelineRunner runner = [this](pir::PassManager& pm, pir::ModuleOp m) {
      return pm.Run(m.program());
    };
    // if (!OptimizeShapeComputation(module_op, runner)) {
    //   return;
    // }
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
