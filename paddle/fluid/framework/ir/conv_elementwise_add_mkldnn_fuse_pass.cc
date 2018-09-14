#include "paddle/fluid/framework/ir/conv_elementwise_add_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct Pattern : public PatternBase {
  Pattern(PDPattern* pattern, const std::string& name_scope)
  : PatternBase{pattern, name_scope, ""}
  { }
  
 private: 
  std::string name_scope() { return name_scope_; }
  std::string repr() { return repr_; } 
  size_t id() { return id_; }
  PDPattern* node_pattern() { return pattern; }
 
 public:
  std::string node_name(std::string op_name) {
    return PDNodeName(name_scope(), repr(), id(), op_name);
  }

  PDNode* retrieve_node(std::string op_name) {
    return node_pattern()->RetrieveNode(node_name(op_name));
  }

  PDNode* new_node(std::string op_name) {
    return node_pattern()->NewNode(node_name(op_name));
  }
};

struct Conv {
  std::string op_name() { return "conv2d"; }
  std::string input_name() { return "Input"; }
  std::string filter_name() { return "Filter"; }
  std::string output_name() { return "Output"; }

  std::function<PDNode* ()> operator()(std::shared_ptr<Pattern> pattern) {
    return [&]() -> PDNode* {
      auto conv_op = pattern->new_node(op_name())
                            ->assert_is_op("conv2d");

      auto input_var = pattern->new_node(input_name())
                              ->assert_is_op_input(op_name(),
                                                   input_name());
                          
      auto filter_var = pattern->new_node(filter_name())
                               ->assert_is_op_input(op_name(),
                                                    filter_name());

      auto output_var = pattern->new_node(output_name())
                               ->assert_is_op_output(op_name(),
                                                     output_name());

      conv_op->LinksFrom({input_var, filter_var});
      conv_op->LinksTo({output_var});

      return output_var;
    };
  }
};

struct ElementwiseAdd {
  std::string op_name() { return "elementwise_add"; }
  std::string x_name() { return "X"; }
  std::string y_name() { return "Y"; }
  std::string out_name() { return "Out"; }

  std::function<PDNode* (PDNode*)> operator()(std::shared_ptr<Pattern> pattern) {
    return [&](PDNode* conv_output) -> PDNode* {
      auto elementwise_add_op = pattern->new_node(op_name())
                                       ->assert_is_op("elementwise_add");

      auto y_var = pattern->new_node(y_name())
                          ->assert_is_op_input(op_name(),
                                               y_name());
  
      conv_output->assert_is_op_input(op_name(),
                                      x_name());

      auto out_var = pattern->new_node(out_name())
                            ->AsOutput()
                            ->assert_is_op_output(op_name(),
                                                  out_name());

      elementwise_add_op->LinksFrom({y_var, conv_output});
      elementwise_add_op->LinksTo({out_var});

      return out_var;
    };
  }
};

Node* GetNodeFromSubgraph(const GraphPatternDetector::subgraph_t& subgraph,
                         std::shared_ptr<patterns::Pattern> pattern,
                         const std::string& op_name) {
  PADDLE_ENFORCE(subgraph.count(pattern->retrieve_node(op_name)),
                 "Node not found for PDNode %s", pattern->node_name(op_name));
  Node* var = subgraph.at(pattern->retrieve_node(op_name));
  PADDLE_ENFORCE(var, "node %s not exists in the sub-graph");
  
  return var;
}

void LinkNodes(Node* from, Node* to) {
  from->outputs.push_back(to);
  to->inputs.push_back(from);
}

void CorrectGraphEdges(Graph* graph, Node* from, Node* to) {
  for (auto& node : GraphTraits::DFS(*graph)) {
    std::vector<Node*> to_remove;
    auto same = std::find_if(std::begin(node.inputs),
                             std::end(node.inputs),
                             [from](Node* n) { return n == from; });

    if (same != std::end(node.inputs)) {
      LinkNodes(to, &node);
    }
  }
}
}  // namespace patterns
using graph_ptr = std::unique_ptr<ir::Graph>;

graph_ptr ConvElementwiseAddMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init("conv_elementwise_add_mkldnn_fuse_pass", graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  auto pattern_ptr = std::make_shared<patterns::Pattern>(pattern, name_scope_);

  patterns::Conv conv_pattern;
  auto conv_output = conv_pattern(pattern_ptr)();

  patterns::ElementwiseAdd elementwise_add_pattern;
  elementwise_add_pattern(pattern_ptr)(conv_output);

  conv_output->AsIntermediate();

  auto fuse_conv = [](Graph* g, Node* conv_input, Node* conv_filter, Node* y) {
    OpDesc op_desc;
    op_desc.SetType("conv2d");

    op_desc.SetInput("Input", {conv_input->Name()});
    op_desc.SetInput("Filter", {conv_filter->Name()});
    op_desc.SetOutput("Output", {y->Name()});

    op_desc.SetAttr("use_mkldnn", true);
    op_desc.SetAttr("fuse_eltwise", true);

    auto fused_conv_op = g->CreateOpNode(&op_desc);

    patterns::LinkNodes(conv_input, fused_conv_op);
    patterns::LinkNodes(conv_filter, fused_conv_op);
    patterns::LinkNodes(fused_conv_op, y);
  };

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* g) {
    auto conv_op = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                       conv_pattern.op_name());
    auto conv_input = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                          conv_pattern.input_name());
    auto conv_filter = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                           conv_pattern.filter_name());
    auto conv_output = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                           conv_pattern.output_name());

    auto elementwise_add_op = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                                  elementwise_add_pattern.op_name());
    auto elementwise_add_y = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                                 elementwise_add_pattern.y_name());
    auto elementwise_add_out = patterns::GetNodeFromSubgraph(subgraph, pattern_ptr,
                                                   elementwise_add_pattern.out_name());

    fuse_conv(g, conv_input, conv_filter, elementwise_add_y);
    patterns::CorrectGraphEdges(g, elementwise_add_out, elementwise_add_y);
    GraphSafeRemoveNodes(g, {conv_output, elementwise_add_out, conv_op, elementwise_add_op});
  };

  gpd(graph.get(), handler);

  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass, paddle::framework::ir::ConvElementwiseAddMKLDNNFusePass);
