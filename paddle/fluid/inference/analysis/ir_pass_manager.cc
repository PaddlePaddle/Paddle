#include "paddle/fluid/inference/analysis/ir_pass_manager.h"
#include <string>

namespace paddle {
namespace inference {
namespace analysis {

IRPassManager::IRPassManager(const ProgramDesc& program) {
  graph_.reset(new framework::ir::Graph(program));
}

void IRPassManager::Apply(const std::vector<std::string>& passes) {
  graph_->Set("graph_viz_path", new std::string("./1.dot"));
  // Apply all the passes
  std::string pre_pass;
  for (const std::string& pass_name : passes) {
    LOG(WARNING) << "Running IR pass [" << pass_name << "]";
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);
    if (pass_name == "graph_viz_pass") {
      std::string dot_file_path =
          "ir_" + (pre_pass.empty() ? "origin" : pre_pass) + ".dot";
      pass->Set("graph_viz_path", new std::string(std::move(dot_file_path)));
    }
    graph_ = pass->Apply(std::move(graph_));
    pre_pass = pass_name;
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
