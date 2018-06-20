#include "paddle/fluid/inference/analysis/tensorrt_subgraph_node_mark_pass.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/node_attr_flags.h"

namespace paddle {
namespace inference {
namespace analysis {

void TensorRTSubgraphNodeMarkPass::Run(DataFlowGraph *graph) {
  for (auto &node : graph->nodes.nodes()) {
    node->attr(ATTR_supported_by_tensorrt).Bool() = teller_(node.get());
  }
}

class DfgDebuggerPass : public DFG_GraphvizDrawPass {
 public:
  DfgDebuggerPass(const DFG_GraphvizDrawPass::Config &config)
      : DFG_GraphvizDrawPass(config) {}

  std::string repr() const override {
    return "tensorrt-subgraph-node-mark-debugger";
  }

  bool Finalize() override { return true; }

 protected:
  std::string Draw(DataFlowGraph *graph) override {
    Dot dot;
    // Add nodes
    for (size_t i = 0; i < graph->nodes.size(); i++) {
      const Node &node = graph->nodes.Get(i);
      if (config_.display_deleted_node || !node.deleted()) {
        auto dot_attr = node.dot_attrs();
        if (node.attr(ATTR_supported_by_tensorrt).Bool()) {
          dot_attr.assign(
              {Dot::Attr{"color", "green"}, Dot::Attr{"style", "filled"}});
        }
        dot.AddNode(node.repr(), dot_attr);
      }
    }
    // Add edges
    for (size_t i = 0; i < graph->nodes.size(); i++) {
      const Node &node = graph->nodes.Get(i);
      if (!config_.display_deleted_node && node.deleted()) continue;
      for (auto &in : node.inlinks) {
        if (!config_.display_deleted_node && in->deleted()) continue;
        dot.AddEdge(in->repr(), node.repr(), {});
      }
    }
    return dot.Build();
  }
};

Pass *TensorRTSubgraphNodeMarkPass::CreateGraphvizDebugerPass() const {
  DFG_GraphvizDrawPass::Config config(
      FLAGS_inference_analysis_graphviz_log_root, "tensorrt_marked_node");
  return new DfgDebuggerPass(config);
}
bool TensorRTSubgraphNodeMarkPass::Finalize() { return true; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
