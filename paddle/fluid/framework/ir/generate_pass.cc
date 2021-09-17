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

void InitGeneratePattern(const proto::PassDesc& pass_desc, PDPattern* pattern) {
  const proto::BlockDesc& block = pass_desc.pattern().blocks(0);
  // Traverse all operators to create subgraph.
  for (int index = 0; index < block.ops_size(); ++index) {
    const proto::OpDesc& op = block.ops(index);
    // Create a PDNode for current operator. Use the index as name to avoid
    // multiple operators with same type. Get a PDNode from pattern subgraph
    // through index in rewrite phase.
    PDNode* op_pdnode =
        pattern->NewNode(std::to_string(index))->assert_is_op(op.type());
    // Create PDNodes for inputs of current operator.
    for (const proto::OpDesc::Var& var : op.inputs()) {
      for (const std::string& argument : var.arguments()) {
        // The input may be the output of other operator.
        PDNode* var_pdnode = pattern->RetrieveNode(argument);
        if (nullptr == var_pdnode) {
          var_pdnode = pattern->NewNode(argument)->AsInput();
        } else if (var_pdnode->IsOutput()) {
          var_pdnode->AsIntermediate();
        }
        var_pdnode->assert_is_op_input(op.type());
        pattern->AddEdge(var_pdnode, op_pdnode);
      }
    }
    // Create PDNodes for outputs of current operator.
    for (const proto::OpDesc::Var& var : op.outputs()) {
      for (const std::string& argument : var.arguments()) {
        // The output may be the input of other operator.
        PDNode* var_pdnode = pattern->RetrieveNode(argument);
        if (nullptr == var_pdnode) {
          var_pdnode = pattern->NewNode(argument)->AsOutput();
        } else if (var_pdnode->IsInput()) {
          var_pdnode->AsIntermediate();
        }
        var_pdnode->assert_is_op_output(op.type());
        pattern->AddEdge(op_pdnode, var_pdnode);
      }
    }
    // Set attribute condition for current operator.
    for (const proto::OpDesc::Attr& attr : op.attrs()) {
      op_pdnode->assert_more([&](Node* x) {
        if (x && x->IsOp()) {
          OpDesc* op_desc = x->Op();
          if (op_desc->HasAttr(attr.name())) {
            return GetAttrValue(attr) == op_desc->GetAttr(attr.name());
          }
          return false;
        }
        return false;
      });
    }
  }
}

GraphPatternDetector::handle_t GetGenerateRewrite(
    const PDPattern& pattern, const proto::PassDesc& pass_desc) {
  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t subgraph, Graph* graph) {
    // There are some duplicate patterns.
    for (auto iter : subgraph) {
      if (nullptr == graph->RetrieveNode(iter.second->id())) {
        VLOG(3) << "Node [" << iter.second->Name()
                << "] of subgraph has been removed. So skip this optimize.";
        return;
      }
    }
    const proto::BlockDesc& block = pass_desc.replace().blocks(0);
    // `var_node_maps` record the mapping of variable to the pattern subgraph.
    std::map<std::string, Node*> var_node_maps;
    for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
      Node* node = subgraph.at(pattern.RetrieveNode(var_map.pattern_var()));
      var_node_maps.insert({var_map.replace_var(), node});
    }
    // Traverse all operators to create subgraph.
    for (const proto::OpDesc& op : block.ops()) {
      OpDesc op_desc;
      std::vector<Node *> in_nodes, out_nodes;
      op_desc.SetType(op.type());
      // Create Nodes for inputs of current operator.
      for (const proto::OpDesc::Var& var : op.inputs()) {
        std::vector<std::string> arguments;
        for (const std::string& argument : var.arguments()) {
          // The input may be mapped on the operator of pattern subgraph.
          Node* node = nullptr;
          auto iter = var_node_maps.find(argument);
          if (var_node_maps.end() == iter) {
            VarDesc var_desc(patterns::UniqueKey(argument));
            node = graph->CreateVarNode(&var_desc);
            var_node_maps.insert({argument, node});
          } else {
            node = iter->second;
          }
          in_nodes.push_back(node);
          arguments.push_back(node->Name());
        }
        op_desc.SetInput(var.parameter(), arguments);
      }
      // Create Nodes for outputs of current operator.
      for (const proto::OpDesc::Var& var : op.outputs()) {
        std::vector<std::string> arguments;
        for (const std::string& argument : var.arguments()) {
          // The output may be mapped on the operator of pattern subgraph.
          Node* node = nullptr;
          auto iter = var_node_maps.find(argument);
          if (var_node_maps.end() == iter) {
            VarDesc var_desc(patterns::UniqueKey(argument));
            node = graph->CreateVarNode(&var_desc);
            var_node_maps.insert({argument, node});
          } else {
            node = iter->second;
          }
          out_nodes.push_back(node);
          arguments.push_back(node->Name());
        }
        op_desc.SetOutput(var.parameter(), arguments);
      }
      // Set attribute for current operator.
      for (const proto::OpDesc::Attr& attr : op.attrs()) {
        op_desc.SetAttr(attr.name(), GetAttrValue(attr));
      }
      // Create a Node for current operator.
      Node* op_node = graph->CreateOpNode(&op_desc);
      for (Node* node : in_nodes) {
        IR_NODE_LINK_TO(node, op_node);
      }
      for (Node* node : out_nodes) {
        IR_NODE_LINK_TO(op_node, node);
      }
    }
    // Remove nodes that are intermediate.
    std::unordered_set<const Node*> remove_nodes;
    for (const std::unique_ptr<PDNode>& pdnode : pattern.nodes()) {
      remove_nodes.emplace(subgraph.at(pdnode.get()));
    }
    for (auto iter : var_node_maps) {
      remove_nodes.erase(iter.second);
    }
    GraphSafeRemoveNodes(graph, remove_nodes);
  };
  return handler;
}

GeneratePass::GeneratePass(const std::string& binary_str) {
  multi_pass_desc_.ParseFromString(binary_str);
  VerifyDesc();
}

GeneratePass::GeneratePass(const proto::MultiPassDesc& multi_pass_desc)
    : multi_pass_desc_(multi_pass_desc) {
  VerifyDesc();
}

void GeneratePass::ApplyImpl(Graph* graph) const {
  for (const proto::PassDesc& pass_desc : multi_pass_desc_.pass_descs()) {
    GraphPatternDetector detector;
    InitGeneratePattern(pass_desc, detector.mutable_pattern());
    detector(graph, GetGenerateRewrite(detector.pattern(), pass_desc));
    // The rewrited graph needs to be verified. Current Pass should be skipped
    // if validation failed. Rewrite based on the original graph cannot
    // implement rollback operation.
    VerifyGraph(*graph);
  }
}

void GeneratePass::VerifyDesc() const {
  PADDLE_ENFORCE_NE(multi_pass_desc_.pass_descs_size(), 0,
                    platform::errors::InvalidArgument(
                        "Size of PassDesc should not be empty."));
  for (const proto::PassDesc& pass_desc : multi_pass_desc_.pass_descs()) {
    // Check inputs/outputs of subgraph should in `var_maps`.
    std::set<std::string> pattern_var_sets, replace_var_sets;
    for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
      pattern_var_sets.emplace(var_map.pattern_var());
      replace_var_sets.emplace(var_map.replace_var());
    }
    auto check_vars = [=](std::set<std::string>* var_sets,
                          const proto::BlockDesc& block) {
      for (const proto::OpDesc& op : block.ops()) {
        for (const proto::OpDesc::Var& var : op.outputs()) {
          for (const std::string& argument : var.arguments()) {
            var_sets->emplace(argument);
          }
        }
      }
      for (const proto::OpDesc& op : block.ops()) {
        for (const proto::OpDesc::Var& var : op.inputs()) {
          for (const std::string& argument : var.arguments()) {
            PADDLE_ENFORCE_NE(
                var_sets->find(argument), var_sets->end(),
                platform::errors::InvalidArgument(
                    "Subgraph of PassDesc has argument [%s] not in `var_maps`.",
                    argument));
          }
        }
      }
    };
    check_vars(&pattern_var_sets, pass_desc.pattern().blocks(0));
    check_vars(&replace_var_sets, pass_desc.replace().blocks(0));
  }
}

bool GeneratePass::VerifyGraph(const Graph& graph) {
  // Return true temporarily.
  return true;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
