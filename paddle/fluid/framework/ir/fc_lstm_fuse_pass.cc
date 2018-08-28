#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FCLstmFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  GraphPatternDetecter gpd;
  auto* pattern = gpd.mutable_pattern();

  std::unordered_set<int> fused_ops({// first lstm
                                     13, 15, 16,
                                     // second lstm
                                     23, 25, 26});

  pattern->NewNode([&](Node* x) { return fused_ops.count(x->id()); },
                   "any_node");

  std::unordered_set<Node*> marked_nodes;

  auto handler = [&](const GraphPatternDetecter::subgraph_t& subgraph,
                     Graph* g) {

    auto* id = subgraph.at(gpd.pattern().RetriveNode("any_node"));
    marked_nodes.insert(id);
  };
  gpd(graph.get(), handler);

  // Create New OpDesc
  auto lstm_creator = [&](int input, int weight_x, int weight_h, int bias,
                          int hidden, int cell, int xx) {
#define GET_NODE(x) auto* x##_n = graph->RetriveNode(x);
    GET_NODE(input);
    GET_NODE(weight_x);
    GET_NODE(weight_h);
    GET_NODE(bias);
    GET_NODE(hidden);
    GET_NODE(cell);
    GET_NODE(xx);

    OpDesc op_desc;
    op_desc.SetType("fusion_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__##_n->Name()});
    SET_IN(X, input);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef GET_NODE
#undef SET_IN

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden_n->Name()});
    op_desc.SetOutput("Cell", {cell_n->Name()});
    op_desc.SetOutput("XX", {xx_n->Name()});
    op_desc.SetOutput("BatchedGate", {"blstm_0.tmp_2"});
    op_desc.SetOutput("BatchCellPreAct", {"blstm_1.tmp_2"});
    op_desc.SetAttr("use_peepholes", false);
    auto* op = graph->CreateOpNode(&op_desc);

#define LINK_TO(a, b)      \
  a->outputs.push_back(b); \
  b->inputs.push_back(a);
    LINK_TO(input_n, op);
    LINK_TO(weight_x_n, op);
    LINK_TO(weight_h_n, op);
    LINK_TO(bias_n, op);
    LINK_TO(op, hidden_n);
#undef LINK_TO
    return op;

  };

  lstm_creator(12, 14, 18, 17, 22, 21, 19);
  lstm_creator(12, 24, 28, 27, 32, 31, 29);

  // remove all the nodes

  for (auto* node : marked_nodes) {
    graph->RemoveNode(const_cast<Node*>(node));
  }

  for (auto* node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (marked_nodes.count(*it)) {
        it = const_cast<Node*>(node)->inputs.erase(it);
      } else
        it++;
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (marked_nodes.count(*it)) {
        it = const_cast<Node*>(node)->outputs.erase(it);
      } else
        it++;
    }
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);
