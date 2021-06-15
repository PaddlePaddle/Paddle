// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/generate_pass.h"

namespace paddle {
namespace framework {
namespace ir {

#define IsMapTensor(var) (var.has_from_op_type() && var.has_from_op_var())

std::string PDNodeName(const proto::PassDesc::Op& op) {
  return string::Sprintf("%s", op.type());
}

std::string PDNodeName(const proto::PassDesc::Var& var) {
  return string::Sprintf("%s:%s", var.from_op_type(), var.from_op_var());
}

std::string PDNodeName(const proto::PassDesc::Op& op,
                       const proto::PassDesc::Var& var) {
  return string::Sprintf("%s:%s", op.type(), var.name());
}

GeneratePass::GeneratePass(const std::string& binary_str) {
  multi_pass_desc_.ParseFromString(binary_str);
}

GeneratePass::GeneratePass(const proto::PassDesc& pass_desc) {
  multi_pass_desc_.add_pass_desc()->CopyFrom(pass_desc);
}

void GeneratePass::ApplyImpl(Graph* graph) const {
  VerifyDesc();
  for (const proto::PassDesc& pass_desc : multi_pass_desc_.pass_desc()) {
    // only substitute? set attr...
    GraphPatternDetector detector;
    InitPattern(detector.mutable_pattern(), pass_desc);
    detector(graph, Substitute(detector.pattern(), pass_desc));
  }
  VerifyGraph();
}

void GeneratePass::InitPattern(PDPattern* pattern,
                               const proto::PassDesc& pass_desc) const {
  PADDLE_ENFORCE_GT(pass_desc.pattern_op_size(), 0);
  PADDLE_ENFORCE_GT(pass_desc.algebra_op_size(), 0);
  // 1. process Op and out Var
  for (const proto::PassDesc::Op& op : pass_desc.pattern_op()) {
    PDNode* op_pdnode = pattern->NewNode(PDNodeName(op));
    op_pdnode->assert_is_op(op.type());
    for (const proto::PassDesc::Var& out : op.output()) {
      PDNode* out_pdnode = pattern->NewNode(PDNodeName(op, out));
      out_pdnode->AsOutput()->assert_is_op_output(op.type());
      pattern->AddEdge(op_pdnode, out_pdnode);
    }
  }
  // 2. process in Var and out Var
  for (const proto::PassDesc::Op& op : pass_desc.pattern_op()) {
    PDNode* op_pdnode = pattern->RetrieveNode(PDNodeName(op));
    for (const proto::PassDesc::Var& in : op.input()) {
      PDNode* in_pdnode = nullptr;
      if (IsMapTensor(in)) {
        in_pdnode = pattern->RetrieveNode(PDNodeName(in));
        // out Var used by Op in pattern is intermediate role
        in_pdnode->AsIntermediate();
      } else {
        in_pdnode = pattern->NewNode(PDNodeName(op, in));
        in_pdnode->AsInput()->assert_is_op_input(op.type());
      }
      if (in.persistable()) {
        in_pdnode->assert_is_persistable_var();
      }
      PADDLE_ENFORCE_NOT_NULL(in_pdnode);
      pattern->AddEdge(in_pdnode, op_pdnode);
    }
  }
}

GraphPatternDetector::handle_t GeneratePass::Substitute(
    const PDPattern& pattern, const proto::PassDesc& pass_desc) const {
  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    std::map<const proto::PassDesc::Var*, Node*> out_var_node_map;
    for (const proto::PassDesc::Op& op : pass_desc.algebra_op()) {
      std::vector<Node *> in_nodes, out_nodes;
      OpDesc op_desc;
      op_desc.SetType(op.type());
      for (const proto::PassDesc::Var& in : op.input()) {
        if (IsMapTensor(in)) {
          PDNode* in_pdnode = pattern.RetrieveNode(PDNodeName(in));
          Node* in_node = subgraph.at(in_pdnode);
          op_desc.SetInput(in.name(), {in_node->Name()});
          in_nodes.push_back(in_node);
        } else {
          // create node
        }
      }
      for (const proto::PassDesc::Var& out : op.output()) {
        if (IsMapTensor(out)) {
          PDNode* out_pdnode = pattern.RetrieveNode(PDNodeName(out));
          Node* out_node = subgraph.at(out_pdnode);
          op_desc.SetOutput(out.name(), {out_node->Name()});
          out_nodes.push_back(out_node);
        } else {
          // create node
        }
      }
      // attr
      Node* op_node = graph->CreateOpNode(&op_desc);
      for (Node* node : in_nodes) {
        IR_NODE_LINK_TO(node, op_node);
      }
      for (Node* node : out_nodes) {
        IR_NODE_LINK_TO(op_node, node);
      }
    }
    std::unordered_set<const Node*> remove_nodes;
    for (const auto& pdnode : pattern.nodes()) {
      remove_nodes.emplace(subgraph.at(pdnode.get()));
    }
    GraphSafeRemoveNodes(graph, remove_nodes);
  };
  return handler;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
