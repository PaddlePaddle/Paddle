#include "paddle/fluid/framework/ir/conv_elementwise_add_mkldnn_fuse_pass.h"

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
  std::string node_name(std::string op_name)
  {
    return PDNodeName(name_scope(), repr(), id(), op_name);
  }

  PDNode* retrieve_node(std::string op_name)
  {
    return node_pattern()->RetrieveNode(node_name(op_name));
  }

  PDNode* new_node(std::string op_name)
  {
    return node_pattern()->NewNode(node_name(op_name));
  }
};

struct Conv {
  std::string conv_name() { return "conv2d"; }
  std::string input_name() { return "Input"; }
  std::string filter_name() { return "Filter"; }
  std::string output_name() { return "Output"; }

  std::function<PDNode* ()> operator()(std::shared_ptr<Pattern> pattern) {
    return [&]() -> PDNode* {
        auto conv_op = pattern->new_node(conv_name())
                              ->assert_is_op("conv2d");

        auto input_var = pattern->new_node(input_name())
                                ->AsInput()
                                ->assert_is_op_input(conv_name());
                            
        auto filter_var = pattern->new_node(filter_name())
                                 ->AsInput()
                                 ->assert_is_persistable_var()
                                 ->assert_is_op_input(conv_name());

        auto output_var = pattern->new_node(output_name())
                                 ->AsOutput()
                                 ->assert_is_op_output(conv_name());

        conv_op->LinksFrom({input_var, filter_var});
        conv_op->LinksTo({output_var});

        return output_var;
    };
  }
};

struct ElementwiseAdd {
  std::string elementwise_add_name() { return "elementwise_add"; }
  std::string x_name() { return "X"; }
  std::string y_name() { return "Y"; }
  std::string out_name() { return "Out"; }

  std::function<PDNode* (PDNode*)> operator()(std::shared_ptr<Pattern> pattern) {
    return [&](PDNode* conv_output) -> PDNode* {
      auto elementwise_add_op = pattern->new_node(elementwise_add_name())
                                       ->assert_is_op("elementwise_add");

      auto y_var = pattern->new_node(y_name())
                          ->AsInput()
                          ->assert_is_op_input(elementwise_add_name());
  
      conv_output->assert_is_op_input(pattern->node_name(elementwise_add_name()),
                                      pattern->node_name(x_name()));
//    auto y_var = pattern->NewNode(y_name())
//                        ->AsInput()
//                        ->assert_is_op_input(elementwise_add_name());

      auto out_var = pattern->new_node(out_name())
                            ->AsOutput()
                            ->assert_is_op_output(
                                      pattern->node_name(elementwise_add_name()));

      elementwise_add_op->LinksFrom({y_var, conv_output});
      elementwise_add_op->LinksTo({out_var});

      return out_var;
    };
  }
};
}  // namespace patterns

Node* node_from_subgraph(const GraphPatternDetector::subgraph_t& subgraph,
                         std::shared_ptr<patterns::Pattern> pattern, const std::string& op_name)
{
  PADDLE_ENFORCE(subgraph.count(pattern->retrieve_node(op_name)),
                 "Node not found for PDNode %s", pattern->node_name(op_name));
  Node* var = subgraph.at(pattern->retrieve_node(op_name));
  PADDLE_ENFORCE(var, "node %s not exists in the sub-graph");
  
  return var;
}

using graph_ptr = std::unique_ptr<ir::Graph>;

graph_ptr ConvElementwiseAddMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init("conv_elementwise_add_mkldnn_fuse_pass", graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  auto pattern_ptr = std::make_shared<patterns::Pattern>(pattern, name_scope_);

  patterns::Conv conv_pattern;
  auto conv_output = conv_pattern(pattern_ptr)();
  conv_output->AsIntermediate();

  patterns::ElementwiseAdd elementwise_add_pattern;
  elementwise_add_pattern(pattern_ptr)(conv_output);

  auto link_nodes_to = [](Node* a, Node* b) {
    a->outputs.push_back(b);
    b->inputs.push_back(a);
  };

  auto fuse_conv = [&](Graph* g, Node* conv_input, Node* conv_filter, Node* y) {
    OpDesc op_desc;
    op_desc.SetType("conv2d");

    op_desc.SetInput("Input", {conv_input->Name()});
    op_desc.SetInput("Filter", {conv_filter->Name()});
    op_desc.SetOutput("Ouput", {y->Name()});

    op_desc.SetAttr("fuse_sum", true);

    auto fused_conv_op = g->CreateOpNode(&op_desc);

    link_nodes_to(conv_input, fused_conv_op);
    link_nodes_to(conv_filter, fused_conv_op);
    link_nodes_to(fused_conv_op, y);
  };

  auto remove_unused_nodes = [](Graph* g, const std::unordered_set<const Node*>& removed_nodes) {
    GraphSafeRemoveNodes(g, removed_nodes);
  };

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* g) {
    auto elementwise_add_x = node_from_subgraph(subgraph, pattern_ptr, elementwise_add_pattern.x_name());
    auto elementwise_add_y = node_from_subgraph(subgraph, pattern_ptr, elementwise_add_pattern.y_name());
    auto elementwise_add_out = node_from_subgraph(subgraph, pattern_ptr, elementwise_add_pattern.out_name());

    auto conv_filter = node_from_subgraph(subgraph, pattern_ptr, conv_pattern.filter_name());
    auto conv_input = node_from_subgraph(subgraph, pattern_ptr, conv_pattern.input_name());
    auto conv_output = node_from_subgraph(subgraph, pattern_ptr, conv_pattern.output_name());

    fuse_conv(g, conv_input, conv_filter, elementwise_add_y);
    remove_unused_nodes(g, {elementwise_add_x, conv_output, elementwise_add_out});
  };

  gpd(graph.get(), handler);

  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass, paddle::framework::ir::ConvElementwiseAddMKLDNNFusePass);
