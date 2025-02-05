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

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

struct FuseExpr {};

// sequence expand, concat fuse pattern, return concat's output
PDNode* BuildSeqExpandConcatPattern(PDPattern* pattern) {
  // The following operators will be fused:
  // concat
  // sequence_expand
  // sequence_expand

  // The following variables will be treat as inputs:
  // concat mid input, 0th input for fused op
  // sequence_expand input, 1st input for fused op
  // sequence_expand input, 2nd input for fused op

  // The following variables will be treat as outputs:
  // concat output

  // So the following variables will be removed:
  // sequence-expand output
  // sequence-expand output

  // Three operators
  auto* sequence_expand0 = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "sequence_expand";
      },
      "sequence_expand0");

  auto* sequence_expand1 = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "sequence_expand";
      },
      "sequence_expand1");

  auto* concat = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "concat" &&  // basic check
               x->Op()->Input("X").size() == 3;                  // Special case
      },
      "concat");

  auto* sequence_expand0_in = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() && VarLinksToOp(x, "sequence_expand");
      },
      "sequence_expand0_in");
  auto* sequence_expand1_in = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() && VarLinksToOp(x, "sequence_expand");
      },
      "sequence_expand1_in");

  // The variables
  auto* sequence_expand0_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&
               VarLinksFromOp(x, "sequence_expand") &&  // basic check
               VarLinksToOp(x, "concat") &&             // is concat's input
               IsNthInput(x, x->outputs[0], "X", 1);    // X[0]
      },
      "sequence_expand0_out");

  auto* sequence_expand1_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&
               VarLinksFromOp(x, "sequence_expand") &&  // basic check
               VarLinksToOp(x, "concat") &&             // is concat's input
               IsNthInput(x, x->outputs[0], "X", 2);    // x[2]
      },
      "sequence_expand1_out");

  auto* concat_in0 = pattern->NewNode(
      [](Node* x) { return x && x->IsVar() && VarLinksToOp(x, "concat"); },
      "concat_in0");

  auto* concat_out = pattern->NewNode(
      [](Node* x) { return x && x->IsVar() && VarLinksFromOp(x, "concat"); },
      "concat_out");

  // Links
  sequence_expand0->LinksFrom({sequence_expand0_in})
      .LinksTo({sequence_expand0_out});
  sequence_expand1->LinksFrom({sequence_expand1_in})
      .LinksTo({sequence_expand1_out});
  concat->LinksFrom({sequence_expand0_out, sequence_expand1_out, concat_in0})
      .LinksTo({concat_out});
  return concat_out;
}

PDNode* BuildFCPattern(PDPattern* pattern, PDNode* fc_x) {
  PDNode* fc_w = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                 // basic
               VarLinksToOp(x, "mul") &&          // link
               x->Var()->Proto()->persistable();  // is a parameter
      },
      "fc_w");

  PDNode* mul_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                     // basic
               VarLinksFromOp(x, "mul") &&            // link
               VarLinksToOp(x, "elementwise_add") &&  //
               !x->Var()->Proto()->persistable();     // is a parameter
      },
      "mul_out");

  PDNode* fc_mul = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "mul";  // basic
      },
      "fc_mul");

  PDNode* fc_bias = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                     // basic
               VarLinksToOp(x, "elementwise_add") &&  // link
               x->Var()->Proto()->persistable();      // is a parameter
      },
      "fc_bias");

  PDNode* elementwise_add = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "elementwise_add";
      },
      "elementwise_add");

  PDNode* add_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                       // basic
               VarLinksFromOp(x, "elementwise_add") &&  // link
               !x->Var()->Proto()->persistable();       // is a parameter
      },
      "add_out");

  std::set<std::string> acts({"sigmoid", "tanh", "relu", "identity"});
  PDNode* act = pattern->NewNode(
      [=](Node* x) { return x && x->IsOp() && acts.count(x->Op()->Type()); },
      "act");

  PDNode* fc_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                  // basic
               !x->Var()->Proto()->persistable();  // is a parameter
      },
      "fc_out");

  fc_mul->LinksFrom({fc_w, fc_x}).LinksTo({mul_out});
  elementwise_add->LinksFrom({mul_out, fc_bias}).LinksTo({add_out});
  act->LinksFrom({add_out}).LinksTo({fc_out});
  return fc_out;
}

SeqConcatFcFusePass::SeqConcatFcFusePass() {
  AddOpCompat(OpCompat("sequence_expand"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("ref_level")
      .IsNumEQ(0)
      .End();

  AddOpCompat(OpCompat("concat"))
      .AddInput("X")  // Input("X"): vector<tensors>
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("tanh"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void SeqConcatFcFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("seq_concat_fc_fuse", graph);
  GraphPatternDetector detector;
  auto* pattern = detector.mutable_pattern();
  auto* concat_out = BuildSeqExpandConcatPattern(pattern);
  BuildFCPattern(pattern, concat_out);

#define GET_NODE(id, pattern)                                           \
  PADDLE_ENFORCE_GT(                                                    \
      subgraph.count(pattern.RetrieveNode(#id)),                        \
      0,                                                                \
      common::errors::NotFound("Pattern has no node called %s.", #id)); \
  auto* id = subgraph.at(pattern.RetrieveNode(#id));                    \
  PADDLE_ENFORCE_NOT_NULL(                                              \
      id, common::errors::NotFound("Subgraph has no node %s.", #id));

  int fuse_count{0};

  detector(graph,
           [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
             if (!IsCompat(subgraph, graph)) {
               LOG(WARNING) << "seq_concat_fc_fuse_pass in op compat failed.";
               return;
             }
             VLOG(4) << "get one concat pattern";
             // fc
             GET_NODE(fc_w, detector.pattern());
             GET_NODE(fc_bias, detector.pattern());
             GET_NODE(act, detector.pattern());
             GET_NODE(fc_out, detector.pattern());

             // concat
             GET_NODE(concat_in0, detector.pattern());
             GET_NODE(sequence_expand0_in, detector.pattern());
             GET_NODE(sequence_expand1_in, detector.pattern());

             OpDesc op_desc;
             op_desc.SetType("fusion_seqexpand_concat_fc");
             op_desc.SetInput("X",
                              {concat_in0->Name(),
                               sequence_expand0_in->Name(),
                               sequence_expand1_in->Name()});
             op_desc.SetInput("FCWeight", {fc_w->Name()});
             op_desc.SetInput("FCBias", {fc_bias->Name()});
             const std::string fc_out_tmp = fc_out->Name() + ".tmp";
             VarDesc fc_out_key(fc_out_tmp);
             fc_out_key.SetPersistable(false);
             auto* fc_out_node = graph->CreateVarNode(&fc_out_key);
             op_desc.SetOutput("FCOut", {fc_out_tmp});
             op_desc.SetOutput("Out", {fc_out->Name()});
             op_desc.SetAttr("fc_activation", act->Op()->Type());

             auto* op_node = graph->CreateOpNode(&op_desc);
             // Add links
             IR_NODE_LINK_TO(fc_w, op_node);
             IR_NODE_LINK_TO(fc_bias, op_node);
             IR_NODE_LINK_TO(concat_in0, op_node);
             IR_NODE_LINK_TO(sequence_expand0_in, op_node);
             IR_NODE_LINK_TO(sequence_expand1_in, op_node);
             IR_NODE_LINK_TO(op_node, fc_out);
             IR_NODE_LINK_TO(op_node, fc_out_node);

             // Clean nodes.
             std::unordered_set<const Node*> marked_nodes;
             for (auto& item : subgraph) {
               marked_nodes.insert(item.second);
             }
             marked_nodes.erase(fc_w);
             marked_nodes.erase(fc_bias);
             marked_nodes.erase(concat_in0);
             marked_nodes.erase(sequence_expand0_in);
             marked_nodes.erase(sequence_expand1_in);
             marked_nodes.erase(fc_out);
             GraphSafeRemoveNodes(graph, marked_nodes);

             ++fuse_count;
           });

  AddStatis(fuse_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(seq_concat_fc_fuse_pass,
              paddle::framework::ir::SeqConcatFcFusePass);
REGISTER_PASS_CAPABILITY(seq_concat_fc_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("sequence_expand", 0)
            .EQ("concat", 0)
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("sigmoid", 0)
            .EQ("tanh", 0)
            .EQ("relu", 0)
            .EQ("identity", 0)
            .EQ("fusion_seqexpand_concat_fc", 0));
