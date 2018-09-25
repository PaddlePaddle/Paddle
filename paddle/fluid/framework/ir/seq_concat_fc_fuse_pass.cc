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

#include "paddle/fluid/framework/ir/seq_concat_fc_fuse_pass.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

struct FuseExpr {};

namespace patterns {

// sequence expand and concat fusion pattern, return concat's output
PDNode* SeqConcatFCPattern::BuildSeqExpandConcatPattern(PDPattern* pattern,
                                                        PDNode* x0, PDNode* x1,
                                                        PDNode* y) {
  // assert input
  x0->assert_is_op_input("sequence_expand");
  x1->assert_is_op_input("sequence_expand");
  y->assert_is_op_input("concat");
  // nodes of three operators
  auto* seq_expand0_op =
      pattern->NewNode(seq_expand0_repr())->assert_is_op("sequence_expand");
  auto* seq_expand1_op =
      pattern->NewNode(seq_expand1_repr())->assert_is_op("sequence_expand");
  auto* concat_op = pattern->NewNode(concat_repr())->assert_is_op("concat");
  // nodes of sequence_expand op output
  auto* seq_expand0_out = pattern->NewNode(seq_expand0_out_repr())
                              ->AsOutput()
                              ->AsIntermediate()
                              ->assert_is_op_output("sequence_expand")
                              ->assert_is_op_input("concat");
  auto* seq_expand1_out = pattern->NewNode(seq_expand1_out_repr())
                              ->AsOutput()
                              ->AsIntermediate()
                              ->assert_is_op_output("sequence_expand")
                              ->assert_is_op_input("concat");
  // node of concat output
  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->AsOutput()
                         ->AsIntermediate()
                         ->assert_is_op_output("concat");
  // create SeqExpandConcat pattern graph
  seq_expand0_op->LinksFrom({x0}).LinksTo({seq_expand0_out});
  seq_expand1_op->LinksFrom({x1}).LinksTo({seq_expand1_out});
  concat_op->LinksFrom({seq_expand0_out, seq_expand1_out, y})
      .LinksTo({concat_out});
  return concat_out;
}
// generate patten dynamically for input node x.
// return final output node of current pattern.
PDNode* SeqConcatFCPattern::operator()(PDNode* x0, PDNode* x1, PDNode* y) {
  // create sequence_expand and concat pattern
  auto* concat_out = BuildSeqExpandConcatPattern(pattern, x0, x1, y);
  concat_out->assert_is_op_input("mul");

  // create fc operators
  auto* mul_op = pattern->NewNode(mul_repr())->assert_is_op("mul");
  auto* elementwise_add_op =
      pattern->NewNode(elementwise_add_repr())->assert_is_op("elementwise_add");

  std::unordered_set<std::string> activation_types(
      {"sigmoid", "tanh", "relu", "identity"});
  auto* activation_op =
      pattern->NewNode(activation_repr())->assert_is_any_op(activation_types);

  // create variables
  auto* weight = pattern->NewNode(weight_repr())
                     ->AsInput()
                     ->assert_is_persistable_var()
                     ->assert_is_op_input("mul", "Y");

  auto* bias = pattern->NewNode(bias_repr())
                   ->AsInput()
                   ->assert_is_persistable_var()
                   ->assert_is_op_input("elementwise_add");

  auto* mul_out = pattern->NewNode(mul_out_repr())
                      ->AsOutput()
                      ->AsIntermediate()
                      ->assert_is_op_output("mul")
                      ->assert_is_op_input("elementwise_add");

  auto* elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                  ->AsOutput()
                                  ->AsIntermediate()
                                  ->assert_is_op_output("elementwise_add")
                                  ->assert_is_any_op_input(activation_types);

  auto* activation_out = pattern->NewNode(activation_out_repr())
                             ->AsOutput()
                             ->assert_is_any_op_output(activation_types);

  // create pattern graph
  mul_op->LinksFrom({concat_out, weight}).LinksTo({mul_out});
  elementwise_add_op->LinksFrom({mul_out, bias}).LinksTo({elementwise_add_out});
  activation_op->LinksFrom({elementwise_add_out}).LinksTo({activation_out});
  // return activation output node
  return activation_out;
}

}  // namespace patterns

std::unique_ptr<ir::Graph> SeqConcatFcFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init("seq_concat_fc_fuse", graph.get());
  GraphPatternDetector detector;
  auto* pattern = detector.mutable_pattern();
  auto* x0 = pattern->NewNode("seq_concat_fc_fuse/x0")->AsInput();
  auto* x1 = pattern->NewNode("seq_concat_fc_fuse/x1")->AsInput();
  auto* y = pattern->NewNode("seq_concat_fc_fuse/y")->AsInput();
  patterns::SeqConcatFCPattern fuse_pattern(pattern, "seq_concat_fc_fuse");
  fuse_pattern(x0, x1, y);

  int fuse_count = 0;

  detector(graph.get(),
           [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
             VLOG(4) << "get one seq_concat_fc pattern";
             // declare persistable vars
             GET_IR_NODE_FROM_SUBGRAPH(weight, weight, fuse_pattern);
             GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fuse_pattern);
             // used to get activation type
             GET_IR_NODE_FROM_SUBGRAPH(act, activation, fuse_pattern);
             // declare output
             GET_IR_NODE_FROM_SUBGRAPH(out, activation_out, fuse_pattern);
             // declare inputs
             auto* seq_expand0_in = subgraph.at(x0);
             auto* seq_expand1_in = subgraph.at(x1);
             auto* concat_in0 = subgraph.at(y);

             OpDesc op_desc;
             op_desc.SetType("fusion_seqexpand_concat_fc");
             op_desc.SetInput("X", {concat_in0->Name(), seq_expand0_in->Name(),
                                    seq_expand1_in->Name()});
             op_desc.SetInput("FCWeight", {weight->Name()});
             op_desc.SetInput("FCBias", {bias->Name()});

             const std::string fc_out_tmp = out->Name() + ".tmp";
             param_scope()->Var(fc_out_tmp)->GetMutable<framework::LoDTensor>();
             op_desc.SetOutput("FCOut", {fc_out_tmp});
             op_desc.SetOutput("Out", {out->Name()});
             op_desc.SetAttr("fc_activation", act->Op()->Type());

             auto* op_node = graph->CreateOpNode(&op_desc);
             // Add links
             IR_NODE_LINK_TO(weight, op_node);
             IR_NODE_LINK_TO(bias, op_node);
             IR_NODE_LINK_TO(concat_in0, op_node);
             IR_NODE_LINK_TO(seq_expand0_in, op_node);
             IR_NODE_LINK_TO(seq_expand1_in, op_node);
             IR_NODE_LINK_TO(op_node, out);

             // Clean nodes
             std::unordered_set<const Node*> marked_nodes;
             for (auto& item : subgraph) {
               marked_nodes.insert(item.second);
             }
             marked_nodes.erase(weight);
             marked_nodes.erase(bias);
             marked_nodes.erase(concat_in0);
             marked_nodes.erase(seq_expand0_in);
             marked_nodes.erase(seq_expand1_in);
             marked_nodes.erase(out);
             GraphSafeRemoveNodes(graph, marked_nodes);

             ++fuse_count;
           });

  AddStatis(fuse_count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(seq_concat_fc_fuse_pass,
              paddle::framework::ir::SeqConcatFcFusePass);
