#include "paddle/fluid/framework/ir/mkldnn_conv_elementwise_add_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct PatternNode {
  PatternNode(PDPattern* pattern,
              const std::string& name,
              const std::string& name_scope,
              const std::string& repr,
              size_t id)
  : nodeName{PDNodeName(name_scope, repr, id, name)}
  , node{pattern->RetrieveNode(nodeName)
  { }

  std::string name() { return nodeName };
  PDNode* node() { return node };

 private:
  std::string nodeName;
  PDNode* node;
};
/*

struct Conv : public PatternBase {
  Conv(PDPattern* pattern, const std::string& name_scope)
  : PatternBase{pattern, name_scope, "conv"}
  , conv{pattern, "conv", name_scope_, repr_, id_}
  , input{pattern, "Input", name_scope_, repr_, id_}
  , filter{pattern, "Filter", name_scope_, repr_, id_}
  , output{pattern, "Output", node_scope_, repr_ id_}
  { }

 private:
  PatternNode conv;
  PatternNode input;
  PatternNode filter;
  PatternNode output;

 public:
  PDNode* operator()() {
    auto conv_op = pattern->NewNode(conv.name())
                          ->assert_is_op("conv2d");

    auto input_var = pattern->NewNode(input.name())
                            ->AsInput()
                            ->assert_is_op_input(conv.name());
                            
    auto filter_var = pattern->NewNode(filter.name())
                             ->AsInput()
                             ->assert_is_persistable_var()
                             ->assert_is_op_input(conv.name());

    auto output_var = patterh->NewNode(output.name())
                             ->AsOutput()
                             ->assert_is_op_output(conv.name());

    conv_op->LinksFrom({input_var, filter_var});
    conv_op->LinksTo({output_var};

    return output_var;
  }
};
*/

struct Conv : public PatternBase {
  Conv(PDPattern* pattern, const std::string& name_scope)
  : PatternBase{pattern, name_scope, "conv"}
  { }

  std::string conv_name() { return PDNodeName(name_scope_, repr_, id_, "conv2d"); }
  PDNode* conv_node() { return pattern->RetrieveNode(conv_name()); }

  std::string input_name() { return PDNodeName(name_scope, repr_, id_, "Input"); }
  PDNode* input_node() { return pattern->RetrieveNode(input_name()); }
  
  std::string filter_name() { return PDNodeName(name_scope_, repr_, id_, "Filter"); }
  PDNode* filter_node() { return pattern->RetrieveNode(filter_name()); }
  
  std::string output_name() { return PDNodeName(name_scope, repr_, id_, "Output"); }
  PDNode* output_node() { return pattern->RetrieveNode(output_name()); }

  PDNode* operator()() {
    auto conv_op = pattern->NewNode(conv_name())
                          ->assert_is_op("conv2d");

    auto input_var = pattern->NewNode(input_name())
                            ->AsInput()
                            ->assert_is_op_input(conv_name());
                            
    auto filter_var = pattern->NewNode(filter_name())
                             ->AsInput()
                             ->assert_is_persistable_var()
                             ->assert_is_op_input(conv_name());

    auto output_var = patterh->NewNode(output_name())
                             ->AsOutput()
                             ->assert_is_op_output(conv_name());

    conv_op->LinksFrom({input_var, filter_var});
    conv_op->LinksTo({output_var};

    return output_var;
  }
};

struct ElementwiseAdd : public PatternBase {
  Conv(PDPattern* pattern, const std::string& name_scope)
  : PatternBase{pattern, name_scope, "elementwise_add"}
  { }

  std::string elementwise_add_name() { return PDNodeName(name_scope_, repr_, id_, "elementwise_add"); }
  PDNode* elementwise_add_node() { return pattern->RetrieveNode(elementwise_add_name()); }

  std::string x_name() { return PDNodeName(name_scope, repr_, id_, "X"); }
  PDNode* x_node() { return pattern->RetrieveNode(x_name()); }
  
  std::string y_name() { return PDNodeName(name_scope_, repr_, id_, "Y"); }
  PDNode* y_node() { return pattern->RetrieveNode(y_name()); }
  
  std::string out_name() { return PDNodeName(name_scope, repr_, id_, "Out"); }
  PDNode* out_node() { return pattern->RetrieveNode(out_name()); }

  PDNode* operator()(PDNode* conv_output) {
    auto elementwise_add_op = pattern->NewNode(conv_name())
                                     ->assert_is_op("elementwise_add");

    auto x_var = pattern->NewNode(x_name())
                        ->AsInput()
                        ->assert_is_op_input(elementwise_add_name());
  
    conv_output->assert_is_op_input(elementwise_add_name(), y_name());
//    auto y_var = pattern->NewNode(y_name())
//                        ->AsInput()
//                        ->assert_is_op_input(elementwise_add_name());

    auto out_var = pattern->NewNode(out_name())
                          ->AsOutput()
                          ->assert_is_op_output(elementwise_add_name());

    conv_op->LinksFrom({x_var, conv_output});
    conv_op->LinksTo({out_var};

    return out_var;
  }
};


}  // namespace patterns

using graph_ptr = std::unique_ptr<ir::Graph>;

graph_ptr MKLDNNConvElementwiseAddFusePass::ApplyImpl(graph_ptr) const {
  FusePassBase::Init("mkldnn_conv_elementwise_add_fuse", graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern(pattern, name_scope_);
  auto conv_output = conv_pattern();
  conv_output->AsIntermediate();

  patterns::ElementwiseAdd elementwise_add_pattern(pattern, name_scope_);
  auto elementwis_add_output = elementwise_add_pattern(conv_output);


}


}  // namespace ir
}  // namespace framework
}  // namespace paddle
