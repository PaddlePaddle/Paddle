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
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_bool(pir_apply_shape_optimization_pass);

constexpr int vlog_level = 3;

namespace pir {
namespace {

using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

void PrintProgram(pir::ModuleOp m, std::string mgs) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(m.program());
  VLOG(vlog_level) << "===================== " << mgs
                   << " =====================\n"
                   << pir::CustomPrintHelper(*m.program(),
                                             shape_analysis.PrintHook());
}

void DebugPrintOpInfo(
    pir::Operation* op,
    pir::ShapeConstraintIRAnalysis* shape_analysis = nullptr) {
  for (auto& res : op->results()) {
    std::ostringstream print_stream;

    print_stream << "  result(" << res.dyn_cast<pir::OpResult>().index() << ") "
                 << "ShapeOrData: {";

    if (shape_analysis != nullptr) {
      auto shape_data = shape_analysis->GetShapeOrDataForValue(res);
      if (shape_data.isa<symbol::TensorListShapeOrDataDimExprs>()) continue;
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

      print_stream << "]";
    }
    print_stream << " }";
    VLOG(vlog_level) << print_stream.str();
  }
}

void InferSymExprForAllValues(ModuleOp module_op) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(module_op.program());
  shape_analysis.Init();
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (auto& block : module_op->region(i)) {
      InferSymExprForBlock(block, &shape_analysis);
    }
  }
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization_pass", 0) {}

  void Run(pir::Operation* op) override {
    VLOG(vlog_level)
        << "===================== ShapeOptimizationPass Run start... "
           "=====================";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
    PrintProgram(module_op, "Origin Program");

    InferSymExprForAllValues(module_op);
    // Runner is for Canonicalizer.
    PassPipelineRunner runner = [this](pir::PassManager& pm, pir::ModuleOp m) {
      pm.EnableIRPrinting();
      return pm.Run(m.program());
    };

    PrintProgram(module_op, "ShapeOptimizationPass Program");
    VLOG(vlog_level) << "===================== ShapeOptimizationPass Run End. "
                        "=====================";
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

void InferSymExprForBlock(const Block& block,
                          ShapeConstraintIRAnalysis* shape_analysis) {
  for (auto& op : block) {
    auto infer_symbolic_shape_interface =
        op.dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
    if (infer_symbolic_shape_interface) {
      VLOG(vlog_level) << op.name() << "(op_id: op_" << op.id() << ")"
                       << " has InferSymbolicShapeInterface.";
      PADDLE_ENFORCE_EQ(
          infer_symbolic_shape_interface.InferSymbolicShape(shape_analysis),
          true,
          "InferSymbolicShape for %s failed.",
          op.name());
      if (op.num_results() > 0) {
        // TODO(lanxianghit): deal with the ops which have more than 1
        // ACTUAL results
        pir::shape::SetShapeAttrForOp(
            &op, shape_analysis->GetShapeOrDataForValue(op.result(0)));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          op.name() + " DOES NOT have InferSymbolicShapeInterface!"));
    }
    DebugPrintOpInfo(&op, shape_analysis);
  }
}

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

namespace pir::shape {

bool HasDynamicShape(const pir::Program& program) {
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          VLOG(vlog_level) << "###### HasDynamicShape == true";
          return true;
        }
      }
    }
  }
  VLOG(vlog_level) << "###### HasDynamicShape == false";
  return false;
}

void AddShapeOptimizationPass(
    std::shared_ptr<pir::PassManager>& pass_manager,  // NOLINT
    pir::Program& program) {                          // NOLINT
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  if (HasDynamicShape(program) && FLAGS_pir_apply_shape_optimization_pass) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
}

}  // namespace pir::shape

REGISTER_IR_PASS(shape_optimization_pass, pir::ShapeOptimizationPass);
