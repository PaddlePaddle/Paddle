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