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
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/pass_registry.h"

namespace pir {
namespace {

using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

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

    print_stream << "  result(" << res.dyn_cast<pir::OpResult>().index() << ") "
                 << "ShapeOrData: {";

    if (shape_analysis != nullptr) {
      auto shape_data = shape_analysis->GetShapeOrDataForValue(res);
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
    VLOG(3) << print_stream.str();
  }
}

void InferSymExprForAllValues(ModuleOp module_op) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(module_op.program());
  shape_analysis.Init();
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (auto& block : module_op->region(i)) {
      for (auto& op : block) {
        auto infer_symbolic_shape_interface =
            op.dyn_cast<paddle::dialect::InferSymbolicShapeInterface>();
        if (infer_symbolic_shape_interface) {
          VLOG(3) << op.name() << " has InferSymbolicShapeInterface.";
          PADDLE_ENFORCE(infer_symbolic_shape_interface.InferSymbolicShape(
                             &shape_analysis),
                         "InferSymbolicShape for %s failed.",
                         op.name());
        } else {
          VLOG(3) << op.name()
                  << " DOES NOT have InferSymbolicShapeInterface!!!!";
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
    VLOG(3) << "===================== ShapeOptimizationPass Run End. "
               "=====================";
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
