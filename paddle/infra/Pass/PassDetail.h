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

#pragma once

/// The code and design is mainly from mlir, very thanks to the great project.

#include "Pass/Pass.h"
#include "Pass/PassManager.h"

namespace infra {
namespace detail {

//==------------------------------------------------------------------==//
// Adaptor pass
// Used to run operation passes over nested operations.
//==------------------------------------------------------------------==//
class AdaptorPass : public Pass {
 public:
  explicit AdaptorPass(PassManager* pm) : Pass("AdaptorPass", 0), mgr(pm) {}

  void Run(mlir::Operation*) override {}

  void Run(mlir::Operation*, int opt_level, bool verify);

  // bool CanScheduleOn(mlir::RegisteredOperationName op_name) const override;

 private:
  void RunImpl(mlir::Operation* op, int opt_level, bool verifyPasses);

  static mlir::LogicalResult RunPass(Pass* pass,
                                     mlir::Operation* op,
                                     AnalysisManager am,
                                     int opt_level,
                                     bool verify);

  static mlir::LogicalResult RunPipeline(PassManager& pm,  // NOLINT
                                         mlir::Operation* op,
                                         AnalysisManager am,
                                         int opt_level,
                                         bool verify);

  PassManager* mgr;

  // For accessing RunPipeline.
  friend class infra::PassManager;
};

}  // namespace detail
}  // namespace infra
