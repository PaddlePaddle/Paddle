// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include <functional>
#include <memory>
#include <optional>
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

COMMON_DECLARE_bool(check_infer_symbolic);
PD_DECLARE_bool(prim_all);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

DimExprs4ValueT MakeDimExprs4Value(
    pir::Program* program, const PassManagerCreater& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
  auto* shape_analysis = &pir::ShapeAnalysisManager::Instance().Get(program);
  return
      [shape_analysis](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
        // TODO(Hongqing-work): define a default empty ShapeOrDataDimExprss
        if (!value) {
          static symbol::ShapeOrDataDimExprs empty{
              symbol::TensorShapeOrDataDimExprs{}};
          return empty;
        }
        return shape_analysis->GetShapeOrDataForValue(value);
      };
}

}  // namespace

void CheckInferSymbolicIfNeed(pir::Program* program,
                              const PassManagerCreater& CreatePassManager) {
  if (!FLAGS_prim_all || !FLAGS_check_infer_symbolic) return;
  const auto& GraphDimExprs4Value =
      MakeDimExprs4Value(program, CreatePassManager);
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateCheckInferSymbolicPass(GraphDimExprs4Value));
  pass_manager->AddPass(CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
