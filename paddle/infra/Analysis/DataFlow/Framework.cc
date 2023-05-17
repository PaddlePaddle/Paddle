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

#include "Analysis/DataFlow/Framework.h"
#include <cassert>

namespace infra {

void DataFlowSolver::InitializeAndRun(Operation* top) {
  for (auto& analysis : analyses_) {
    analysis->Initialize(top);
  }
  do {
    while (!worklist_.empty()) {
      auto pair = worklist_.front();
      worklist_.pop();
      pair.second->Visit(pair.first);
    }
  } while (!worklist_.empty());
}

DataFlowAnalysis::DataFlowAnalysis(DataFlowSolver& solver) : solver_{solver} {}

void DataFlowAnalysis::AddDependency(AnalysisState* state, ProgramPoint point) {
  solver_.AddDependency(state, this, point);
}

void DataFlowAnalysis::PropagateIfChanged(AnalysisState* state,
                                          ChangeStatus changed) {
  solver_.PropagateIfChanged(state, changed);
}

}  // namespace infra
