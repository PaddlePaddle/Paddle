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

#include <algorithm>
#include <cassert>
#include <optional>

#include "Pass/AnalysisManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {
class PassRegistry;
namespace detail {
class AdaptorPass;

struct PassExecutionState {
  explicit PassExecutionState(mlir::Operation* ir, AnalysisManager am)
      : ir(ir), pass_failed(false), am(am) {}

  mlir::Operation* ir;
  bool pass_failed;
  AnalysisManager am;
  detail::PreservedAnalyses preserved_analyses;
};

}  // namespace detail

struct PassInfo {
  PassInfo(const std::string& name,
           int opt_level,
           const std::vector<std::string>& dependents = {})
      : name(name), opt_level(opt_level), dependents(dependents) {}

  // Pass name.
  std::string name;

  // opt_level=0: the basic pass which framework need.
  // opt_level=1: the fusion logical pass.
  // opt_level=2: constant fold, cse, memory optimize, etc.
  // opt_level=3: layout.
  int opt_level;

  // The list which pass depends on. PassManager will check the constraint.
  std::vector<std::string> dependents;
};

/// We can access pass only from PassManager.
class Pass {
 public:
  virtual ~Pass() = default;
  explicit Pass(const std::string& name,
                int opt_level,
                const std::vector<std::string>& dependents = {})
      : info_(name, opt_level, dependents) {}

  PassInfo GetPassInfo() const { return info_; }

  std::unique_ptr<Pass> Clone() const { return ClonePass(); }

 protected:
  virtual void Run(mlir::Operation* op) = 0;

  virtual inline bool CanScheduleOn(mlir::Operation* op) const {
    return op->getNumRegions() > 0;
  }

  virtual mlir::LogicalResult Initialize(mlir::MLIRContext* context) {
    return mlir::success();
  }

  // TODO(wilber): need to consider pure virtual.
  /// A clone method to create a copy of this pass.
  virtual std::unique_ptr<Pass> ClonePass() const { return nullptr; }

  AnalysisManager GetAnalysisManager() { return pass_state_->am; }

  void SignalPassFailure() { pass_state_->pass_failed = true; }

  PassInfo info_;
  std::optional<detail::PassExecutionState> pass_state_;

  friend class PassManager;
  friend class detail::AdaptorPass;
  friend class PassRegistry;
};

}  // namespace infra
