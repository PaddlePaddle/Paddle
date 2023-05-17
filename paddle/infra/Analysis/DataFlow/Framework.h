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

#include <map>
#include <queue>
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/Support/TypeID.h"

namespace infra {

class DataFlowAnalysis;
class AnalysisState;

enum class ChangeStatus : int8_t {
  NoChange,
  Change,
};

ChangeStatus operator|(ChangeStatus lhs, ChangeStatus rhs) {
  return lhs == ChangeStatus::Change ? lhs : rhs;
}

ChangeStatus operator&(ChangeStatus lhs, ChangeStatus rhs) {
  return lhs == ChangeStatus::NoChange ? lhs : rhs;
}

// Launch the data flow analyses, running the algotithm.
class DataFlowSolver {
 public:
  using Operation = ::mlir::Operation;
  using ProgramPoint = ::mlir::ProgramPoint;
  using WorkItem = std::pair<ProgramPoint, DataFlowAnalysis*>;
  using TypeID = ::mlir::TypeID;

  template <typename AnalysisT, typename... Args>
  AnalysisT* Load(Args&&... args);

  void InitializeAndRun(Operation* op);

  template <typename StateT, typename PointT>
  const StateT* LookupState(PointT point) const;

  void AddDependency(AnalysisState* state,
                     DataFlowAnalysis* analysis,
                     ProgramPoint point);

  void PropagateIfChanged(AnalysisState* state, ChangeStatus changed);

  template <typename StateT, typename PointT>
  StateT* GetOrCreateState(PointT point);

  template <typename PointT, typename... Args>
  PointT* GetOrCreatePoint(Args&&... args) {
    return PointT::get(uniquer_, std::forward<Args>(args)...);
  }

 private:
  std::queue<WorkItem> worklist_;
  std::vector<std::unique_ptr<DataFlowAnalysis>> analyses_;
  ::llvm::DenseMap<std::pair<ProgramPoint, TypeID>,
                   std::unique_ptr<AnalysisState>>
      analysis_states_;

  ::mlir::StorageUniquer uniquer_;
  friend class DataFlowAnalysis;
};

template <typename AnalysisT, typename... Args>
AnalysisT* DataFlowSolver::Load(Args&&... args) {
  analyses_.emplace_back(
      std::make_unique<AnalysisT>(*this, std::forward<Args>(args)...));
  return static_cast<AnalysisT*>(analyses_.back().get());
}

template <typename StateT, typename PointT>
const StateT* DataFlowSolver::LookupState(PointT point) const {
  auto it = std::find(analyses_, {ProgramPoint(point), TypeID::get<StateT>()});
  return it == analyses_.end() ? nullptr
                               : static_cast<const StateT*>(it->second.get());
}

template <typename StateT, typename PointT>
StateT* DataFlowSolver::GetOrCreateState(PointT point) {
  auto& state = analysis_states_[{ProgramPoint(point), TypeID::get<StateT>()}];
  if (!state) {
    state = std::make_unique<StateT>(point);
  }
  return static_cast<StateT*>(state.get());
}

// The base class of analysis state, which contains the information
// in the analysis process.
class AnalysisState {
 public:
  using ProgramPoint = ::mlir::ProgramPoint;
  virtual ~AnalysisState() = default;

  explicit AnalysisState(ProgramPoint point) : point_(point) {}

  ProgramPoint GetPoint() const { return point_; }

 protected:
  virtual void PropagateUpdate(DataFlowSolver* solver) const {}

  ::llvm::SetVector<DataFlowSolver::WorkItem> deps_;

  ProgramPoint point_;

  friend class DataFlowSolver;
};

class PredecessorState : public AnalysisState {
 public:
  using AnalysisState::AnalysisState;

  bool allPredecessorsKnown() const { return all_known_; }

  ::llvm::ArrayRef<::mlir::Operation*> getKnownPredecessors() const {
    return known_predecessors_;
  }

 private:
  bool all_known_{true};
  ::llvm::ArrayRef<::mlir::Operation*> known_predecessors_;
};

// Base class of all data flow analyses.
class DataFlowAnalysis {
 public:
  using Operation = ::mlir::Operation;
  using ProgramPoint = ::mlir::ProgramPoint;

  explicit DataFlowAnalysis(DataFlowSolver& solver);  // NOLINT
  virtual bool Initialize(Operation* top) = 0;
  virtual bool Visit(ProgramPoint point) = 0;

 protected:
  void AddDependency(AnalysisState* state, ProgramPoint point);

  void PropagateIfChanged(AnalysisState* state, ChangeStatus changed);

  template <typename PointT>
  void RegisterPointKind() {
    solver_.uniquer_.registerParametricStorageType<PointT>();
  }

  template <typename PointT, typename... Args>
  PointT* GetOrCreatePoint(Args&&... args) {
    return solver_.GetOrCreatePoint<PointT>(std::forward<Args>(args)...);
  }

  template <typename StateT, typename PointT>
  StateT* GetOrCreate(PointT point) {
    return solver_.GetOrCreateState<StateT>(point);
  }

  template <typename StateT, typename PointT>
  const StateT* GetOrCreateFor(ProgramPoint dependent, PointT point) {
    auto* state = GetOrCreate<StateT>(point);
    AddDependency(state, dependent);
    return state;
  }

 private:
  DataFlowSolver& solver_;

  friend class DataFlowSolver;
};

}  // namespace infra
