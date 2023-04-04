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

#include "Analysis/DataFlow/Framework.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace infra {

class RegionBranchOpInterface;

namespace dataflow {

// A dense lattice is attached to operations to represent the program
// state after execution, or to blocks to represent the program state
// at the beginning of the block. It is propagated through the analysis.
class AbstractDenseLattice : public AnalysisState {
 public:
  using AnalysisState::AnalysisState;

  virtual ChangeStatus Join(const AbstractDenseLattice& rhs) = 0;
};

// Implements a transfer function from the lattice between operations.
class AbstractDenseAnalysis : public DataFlowAnalysis {
 public:
  using DataFlowAnalysis::DataFlowAnalysis;
  using Operation = ::mlir::Operation;
  using Block = ::mlir::Block;
  using RegionBranchOpInterface = ::mlir::RegionBranchOpInterface;
  using CallOpInterface = ::mlir::CallOpInterface;
  using CallableOpInterface = ::mlir::CallableOpInterface;

  // Traversals every operation and block and initialize them.
  bool Initialize(Operation* top) override;

  // Visit a program point and modifiy the state of the program.
  bool Visit(ProgramPoint point) override;

 protected:
  virtual void VisitOperationImpl(Operation* op,
                                  const AbstractDenseLattice& before,
                                  AbstractDenseLattice* after) = 0;

  virtual AbstractDenseLattice* GetLattice(ProgramPoint point) = 0;

  virtual void SetToEntryState(AbstractDenseLattice* lattice) = 0;

  const AbstractDenseLattice* GetLatticeFor(ProgramPoint dependent,
                                            ProgramPoint point);

  void Join(AbstractDenseLattice* lhs, const AbstractDenseLattice& rhs) {
    PropagateIfChanged(lhs, lhs->Join(rhs));
  }

 protected:
  // If the operation is a call or region, the state is set by control-flow.
  // Otherwise it calls the transfer function.
  virtual void VisitOperation(Operation* op);

  void VisitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch);

  void VisitCallOperation(ProgramPoint point, CallOpInterface call);

  void VisitCallableOperation(ProgramPoint point, CallableOpInterface callable);

  void VisitBlock(Block* block);
};

template <typename LatticeT>
class DenseAnalysis : public AbstractDenseAnalysis {
  static_assert(
      std::is_base_of<AbstractDenseLattice, LatticeT>::value,
      "The class `LatticeT` must derive from `AbstractDenseLattice`.");

 public:
  using AbstractDenseAnalysis::AbstractDenseAnalysis;

  virtual void VisitOperation(Operation* op,
                              const LatticeT& before,
                              LatticeT* after) = 0;
};

}  // namespace dataflow
}  // namespace infra
