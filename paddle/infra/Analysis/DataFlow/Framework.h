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

#include <queue>
#include "llvm/ADT/PointerUnion.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {

class DataFlowAnalysis;

// The base class of analysis state, which contains the information
// in the analysis process.
class AnalysisState {
 public:
  using ProgramPoint = ::mlir::ProgramPoint;
  virtual ~AnalysisState() = default;

  ProgramPoint GetPoint() const { return point_; }

 private:
  ProgramPoint point_;
};

// Launch the data flow analyses, running the algotithm.
class DataFlowSolver {
 public:
  using Operation = ::mlir::Operation;
  using ProgramPoint = ::mlir::ProgramPoint;

  template <typename AnalysisT, typename... Args>
  AnalysisT* Load(Args&&... args);

  void InitializeAndRun(Operation* op);

  template <typename StateT, typename PointT>
  const StateT* LookupState(PointT point) const;

  void AddDependency(AnalysisState* state,
                     DataFlowAnalysis* analysis,
                     ProgramPoint point);

  void PropagateIfChanged(AnalysisState* state, bool changed);

 private:
  std::queue<std::pair<ProgramPoint, DataFlowAnalysis*>> worklist_;
  std::vector<std::unique_ptr<DataFlowAnalysis>> analyses_;
  friend class DataFlowAnalysis;
};

template <typename AnalysisT, typename... Args>
AnalysisT* DataFlowSolver::Load(Args&&... args) {
  analyses_.emplace_back(
      std::make_unique<AnalysisT>(*this, std::forward<Args>(args)...));
  return static_cast<AnalysisT*>(analyses_.back().get());
}

// Base class of all data flow analyses.
class DataFlowAnalysis {
 public:
  using Operation = ::mlir::Operation;
  using ProgramPoint = ::mlir::ProgramPoint;

  explicit DataFlowAnalysis(DataFlowSolver& solver);  // NOLINT
  virtual void Initialize(Operation* top) = 0;
  virtual void Visit(ProgramPoint point) = 0;

 protected:
  void AddDependency(AnalysisState* state,
                     DataFlowAnalysis* analysis,
                     ProgramPoint point);

  void PropagateIfChanged(AnalysisState* state, bool changed);

 private:
  DataFlowSolver& solver_;

  friend class DataFlowSolver;
};

}  // namespace infra
