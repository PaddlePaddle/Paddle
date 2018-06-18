// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/inference/analysis/pass_manager.h"

namespace paddle {
namespace inference {
namespace analysis {

class DfgPassManagerImpl final : public DfgPassManager {
 public:
  DfgPassManagerImpl() {
    // TODO(Superjomn) set the key with pass reprs.
    Register("fluid-to-data-flow-graph", new FluidToDataFlowGraphPass);
    Register("data-flow-graph-to-fluid", new DataFlowGraphToFluidPass);
  }
  std::string repr() const override { return "dfg-pass-manager"; }
  std::string description() const override { return "DFG pass manager."; }
};

Analyzer::Analyzer() { Register("manager1", new DfgPassManagerImpl); }

void Analyzer::Run(Argument* argument) {
  for (auto& x : data_) {
    PADDLE_ENFORCE(x->Initialize(argument));
    x->RunAll();
    PADDLE_ENFORCE(x->Finalize());
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle