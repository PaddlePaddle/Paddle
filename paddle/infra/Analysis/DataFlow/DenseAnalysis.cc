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

#include "Analysis/DataFlow/DenseAnalysis.h"

namespace infra {
namespace dataflow {

bool AbstractDenseAnalysis::Initialize(Operation* top) {
  VisitOperation(top);
  bool ret = true;
  for (auto& region : top->getRegions()) {
    for (auto& block : region) {
      VisitBlock(&block);
      for (auto& op : block) {
        ret = ret && Initialize(&op);
      }
    }
  }
  return ret;
}

bool AbstractDenseAnalysis::Visit(ProgramPoint point) {
  if (auto* op = point.dyn_cast<Operation*>()) {
    VisitOperation(op);
  } else if (auto* block = point.dyn_cast<Block*>()) {
    VisitBlock(block);
  } else {
    return false;
  }
  return true;
}

void AbstractDenseAnalysis::VisitOperation(Operation* op) {
  if (auto branch = ::mlir::dyn_cast<::mlir::RegionBranchOpInterface>(op)) {
    VisitRegionBranchOperation(op, branch);
  } else if (auto call = ::mlir::dyn_cast<::mlir::CallOpInterface>(op)) {
    VisitCallOperation(op, call);
  } else {
    const AbstractDenseLattice* before;
    if (auto* prev = op->getPrevNode()) {
      before = GetLatticeFor(op, prev);
    } else if (auto* prev = op->getBlock()) {
      before = GetLatticeFor(op, prev);
    }
    VisitOperationImpl(op, *before, GetLattice(op));
  }
}

void AbstractDenseAnalysis::VisitBlock(Block* block) {
  if (block->isEntryBlock()) {
    if (auto callable =
            ::mlir::dyn_cast<CallableOpInterface>(block->getParentOp())) {
      VisitCallableOperation(block, callable);
    } else if (auto branch = ::mlir::dyn_cast<RegionBranchOpInterface>(
                   block->getParentOp())) {
      VisitRegionBranchOperation(block, branch);
    } else {
      SetToEntryState(GetLattice(block));
    }
  } else {
    for (auto it = block->pred_begin(); it != block->pred_end(); ++it) {
      Block* pred = *it;
      Operation* terminator = pred->getTerminator();
      Join(GetLattice(block), *GetLatticeFor(block, terminator));
    }
  }
}

void AbstractDenseAnalysis::VisitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch) {
  auto* after = GetLattice(point);
  const auto* predecessors = GetOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown());
  for (Operation* op : predecessors->getKnownPredecessors()) {
    const AbstractDenseLattice* before;
    if (op == branch) {
      if (auto* prev = op->getPrevNode()) {
        before = GetLatticeFor(op, prev);
      } else if (auto* prev = op->getBlock()) {
        before = GetLatticeFor(op, prev);
      }
    } else {
      before = GetLatticeFor(point, op);
    }
    Join(after, *before);
  }
}

void AbstractDenseAnalysis::VisitCallOperation(ProgramPoint op,
                                               CallOpInterface call) {
  auto* after = GetLattice(op);
  const auto* predecessors = GetOrCreateFor<PredecessorState>(op, call);
  if (!predecessors->allPredecessorsKnown()) {
    SetToEntryState(after);
    return;
  }
  for (auto* predecessor : predecessors->getKnownPredecessors()) {
    Join(after, *GetLatticeFor(op, predecessor));
  }
}

void AbstractDenseAnalysis::VisitCallableOperation(
    ProgramPoint block, CallableOpInterface callable) {
  auto* after = GetLattice(block);
  assert(callable.getCallableRegion() == block.get<Block*>()->getParent());
  const auto* callsites = GetOrCreateFor<PredecessorState>(block, callable);
  if (!callsites->allPredecessorsKnown()) {
    return SetToEntryState(after);
  }
  for (Operation* op : callsites->getKnownPredecessors()) {
    const AbstractDenseLattice* before;
    if (auto* prev = op->getPrevNode()) {
      before = GetLatticeFor(op, prev);
    } else if (auto* prev = op->getBlock()) {
      before = GetLatticeFor(op, prev);
    }
    Join(after, *before);
  }
}

const AbstractDenseLattice* AbstractDenseAnalysis::GetLatticeFor(
    ProgramPoint dependent, ProgramPoint point) {
  AbstractDenseLattice* state = GetLattice(point);
  AddDependency(state, dependent);
  return state;
}

}  // namespace dataflow
}  // namespace infra
