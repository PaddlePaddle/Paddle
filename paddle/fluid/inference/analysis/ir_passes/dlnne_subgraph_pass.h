#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {

int ConvertGraph(std::string graph_name);

namespace analysis {

class DlnneSubgraphPass : public framework::ir::FusePassBase {
 public:
  void ApplyImpl(framework::ir::Graph *graph) const override;

 private:
  void CleanIntermediateOutputs(framework::ir::Node *node);
  void CreateDlnneOp(framework::ir::Node *x, framework::ir::Graph *graph,
                     const std::vector<std::string> &graph_params,
                     std::vector<std::string> *repetitive_params) const;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
