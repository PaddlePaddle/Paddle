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

#include <memory>
#include <optional>
#include <functional>
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"

COMMON_DECLARE_bool(check_infer_symbolic);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

OptDimExprs4ValueT MakeOptDimExprs4Value(
    pir::Program* program,
    const PassManagerCreater& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
  const auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(program);
  return [shape_analysis](pir::Value value, const pir::Block*)
    -> std::optional<const symbol::ShapeOrDataDimExprs*> {
    if (!shape_analysis->HasShapeOrDataForValue(value)) return std::nullopt;
    return &shape_analysis->GetShapeOrDataForValue(value);
  };
}

}

void CheckInferSymbolicIfNeed(pir::Program* program,
                              const PassManagerCreater& CreatePassManager) {
  if (!FLAGS_check_infer_symbolic) return;
  const auto& GraphDimExprs4Value =
    MakeOptDimExprs4Value(program, CreatePassManager);
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateCheckInferSymbolicPass(GraphDimExprs4Value));
  pass_manager->AddPass(CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
