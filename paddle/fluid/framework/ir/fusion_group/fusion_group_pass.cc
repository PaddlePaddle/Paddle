/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/elementwise_group_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void FusionGroupPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);

  int num_elementwise_groups = DetectFusionGroup(graph, 0);
  LOG(INFO) << "Detect " << num_elementwise_groups
            << " elementwise fusion groups.";
}

int FusionGroupPass::DetectFusionGroup(Graph* graph, int type) const {
  std::vector<fusion_group::SubGraph> subgraphs;
  std::unordered_set<Node*> all_nodes = graph->Nodes();
  for (Node* n : all_nodes) {
    bool is_found = false;
    for (auto& subgraph : subgraphs) {
      if (subgraph.nodes_set.find(n) != subgraph.nodes_set.end()) {
        is_found = true;
        break;
      }
    }
    if (is_found) {
      continue;
    }

    fusion_group::SubGraph subgraph;
    if (type == 0) {
      fusion_group::ElementwiseGroupDetector detector;
      int num_operations = detector(n);
      if (num_operations >= 2) {
        subgraph = detector.GetSubgraph();
      }
    }

    if (!subgraph.IsEmpty()) {
      subgraphs.push_back(subgraph);
    }
  }

  // TODO(liuyiqun): check whether there are intersection between subgraphs
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    InsertFusionGroupOp(graph, subgraphs[i]);
  }
  return subgraphs.size();
}

void FusionGroupPass::InsertFusionGroupOp(
    Graph* graph, const fusion_group::SubGraph& subgraph) const {
  std::vector<Node*> input_vars_of_subgraph = subgraph.GetInputVarNodes();
  std::vector<Node*> output_vars_of_subgraph = subgraph.GetOutputVarNodes();
  std::unordered_set<Node*> external_nodes;

  OpDesc op_desc;
  op_desc.SetType("fusion_group");

  std::vector<std::string> input_names;
  for (auto* n : input_vars_of_subgraph) {
    input_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetInput("Xs", input_names);

  std::vector<std::string> output_names;
  for (auto* n : output_vars_of_subgraph) {
    output_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetOutput("Outs", output_names);
  op_desc.SetAttr("type", subgraph.type);
  op_desc.SetAttr("func_name", subgraph.func_name);

  auto fusion_group_node = graph->CreateOpNode(&op_desc);
  for (auto* in : input_vars_of_subgraph) {
    IR_NODE_LINK_TO(in, fusion_group_node);
  }
  for (auto* out : output_vars_of_subgraph) {
    IR_NODE_LINK_TO(fusion_group_node, out);
  }

  std::unordered_set<const Node*> internal_nodes;
  for (auto* n : subgraph.nodes_set) {
    if (external_nodes.find(n) == external_nodes.end()) {
      internal_nodes.insert(n);
    }
  }
  GraphSafeRemoveNodes(graph, internal_nodes);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fusion_group_pass, paddle::framework::ir::FusionGroupPass);
