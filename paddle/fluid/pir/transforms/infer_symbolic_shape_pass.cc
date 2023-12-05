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

#include "paddle/fluid/pir/transforms/infer_symbolic_shape_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

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
    PrintStatistics(num_rewrites_);
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
      num_rewrites_++;
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
  int64_t num_rewrites_{0};
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateInferSymbolicShapePass(
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  return std::make_unique<InferSymbolicShapePass>(shape_analysis);
}

}  // namespace pir
