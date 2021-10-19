/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
// #include "cinn/frontend/op_mapper_registry.h"
// #include "cinn/frontend/op_mappers/use_op_mappers.h"

// TODO(jiangcheng05): just for local compile, remove after
// paddle and CINN have been binded
// The APIs are the same as CINN:
// https://github.com/PaddlePaddle/CINN/blob/develop/cinn/utils/registry.h
namespace cinn {
namespace frontend {
class OpMapperRegistry {
 public:
  static OpMapperRegistry* Global() {
    static OpMapperRegistry inst;
    return &inst;
  }

  inline const OpMapperRegistry* Find(const std::string& name) {
    std::unordered_set<std::string> fmap_ = {"mul", "add", "relu", "sigmoid",
                                             "softmax"};
    auto p = fmap_.find(name);
    if (p != fmap_.end()) {
      return this;
    } else {
      return nullptr;
    }
  }
};

}  // namespace frontend
}  // namespace cinn

namespace paddle {
namespace framework {
namespace paddle2cinn {

using framework::ir::Graph;
using framework::ir::Node;

using GraphNodeVec = std::vector<Node*>;
using GraphNodeSet = std::unordered_set<Node*>;

// Create new subgraph with and op nodes are cluster nodes, and all
// var node are from internal nodes
std::unique_ptr<Graph> CreateNewSubGraph(
    const GraphNodeSet& cluster, const GraphNodeSet& cluster_internals) {
  // Graph's constructor must has one parameter, and in our code,
  // the ProgramDesc is useless, so here we pass a temporary object.
  auto sub_graph = std::make_unique<Graph>(framework::ProgramDesc());

  std::unordered_map<Node*, Node*> old_op2new_op;
  for (auto* op : cluster) {
    auto sub_node = sub_graph->CreateOpNode(op->Op());
    old_op2new_op[op] = sub_node;
  }

  std::unordered_map<Node*, Node*> old_var2new_var;
  for (auto* var : cluster_internals) {
    auto sub_node = sub_graph->CreateVarNode(var->Var());
    old_var2new_var[var] = sub_node;
  }

  // the subgraph is independently, so here we only need link
  // to the node in new subgraph, and discard the link to
  // out-graph.
  for (auto* op : cluster) {
    for (auto* var : op->inputs) {
      if (cluster_internals.count(var)) {
        old_op2new_op[op]->inputs.emplace_back(old_var2new_var[var]);
      }
    }
    for (auto* var : op->outputs) {
      if (cluster_internals.count(var)) {
        old_op2new_op[op]->outputs.emplace_back(old_var2new_var[var]);
      }
    }
  }

  for (auto* var : cluster_internals) {
    for (auto* op : var->inputs) {
      if (cluster.count(op)) {
        old_var2new_var[var]->inputs.emplace_back(old_op2new_op[op]);
      }
    }
    for (auto* op : var->outputs) {
      if (cluster.count(op)) {
        old_var2new_var[var]->outputs.emplace_back(old_op2new_op[op]);
      }
    }
  }

  return sub_graph;
}

// This interface is used to classify all variables involved in a cluster into
// three types: inputs, outputs, and internals.
// Specially, the internal node is a node that only used by sub-graph, and
// out-graph should not using this node at all.
// inputs & outputs & internals == NULL
// inputs | outputs | internals == all graph node
void AnalyseClusterVariables(const GraphNodeSet& cluster,
                             GraphNodeSet* cluster_inputs,
                             GraphNodeSet* cluster_outputs,
                             GraphNodeSet* cluster_internals) {
  // collecting all input and output of op
  for (auto* op_node : cluster) {
    for (auto* input_var_node : op_node->inputs) {
      cluster_inputs->insert(input_var_node);
    }
    for (auto* output_var_node : op_node->outputs) {
      cluster_outputs->insert(output_var_node);
    }
  }
  // remove output node from cluster_inputs,
  // and add cluster_internals node
  for (auto* var_node : *cluster_outputs) {
    if (cluster_inputs->count(var_node) > 0) {
      // if a input node also exists in output list, remove
      cluster_inputs->erase(var_node);

      // the internal node is must an output node of sub-graph,
      // but not any input node of out-graph.
      bool is_only_used_internal = true;
      for (auto* next_op_node : var_node->outputs) {
        is_only_used_internal &= (cluster.count(next_op_node) > 0);
      }
      if (is_only_used_internal) {
        cluster_internals->insert(var_node);
      }
    }
  }

  // if a output node also exists in input list, remove.
  for (auto* var_node : *cluster_inputs) {
    cluster_outputs->erase(var_node);
  }
  // if a output node also exists in internal list, remove.
  for (auto* var_node : *cluster_internals) {
    cluster_outputs->erase(var_node);
  }
}

Node* AddSpecialOpToGraph(Graph* graph, const GraphNodeSet& cluster_inputs,
                          const GraphNodeSet& cluster_outputs) {
  // add special cinn op
  framework::OpDesc special_op_desc;
  special_op_desc.SetType(kCinnLaunchOp);
  auto* special_op_node = graph->CreateOpNode(&special_op_desc);
  special_op_node->inputs.assign(cluster_inputs.begin(), cluster_inputs.end());
  special_op_node->outputs.assign(cluster_outputs.begin(),
                                  cluster_outputs.end());
  return special_op_node;
}

void AddLinkToSpecialOp(Node* special_op_node,
                        const GraphNodeSet& cluster_inputs,
                        const GraphNodeSet& cluster_outputs) {
  // add new link from cluster_inputs to special_op_node
  for (auto* var_node : cluster_inputs) {
    var_node->outputs.push_back(special_op_node);
  }

  // add new link from special_op_node to cluster_outputs
  for (auto* var_node : cluster_outputs) {
    var_node->inputs.push_back(special_op_node);
  }
}

void RemoveLinkFromCluster(const GraphNodeSet& cluster,
                           const GraphNodeSet& cluster_inputs,
                           const GraphNodeSet& cluster_outputs) {
  // remove all nodes in cluster
  auto get_preserved_ops = [&cluster](const GraphNodeVec& ops) {
    GraphNodeVec nodes;
    for (auto* op_node : ops) {
      if (cluster.find(op_node) == cluster.end()) {
        nodes.emplace_back(op_node);
      }
    }
    return nodes;
  };

  // removing useless link from cluster_inputs to cluster
  for (auto* var_node : cluster_inputs) {
    auto preserved_nodes = get_preserved_ops(var_node->outputs);
    var_node->outputs.assign(preserved_nodes.begin(), preserved_nodes.end());
  }

  // removing useless link from cluster to cluster_outputs
  for (auto* var_node : cluster_outputs) {
    auto preserved_nodes = get_preserved_ops(var_node->inputs);
    var_node->inputs.assign(preserved_nodes.begin(), preserved_nodes.end());
  }
}

// Removing cluster node and internals node from Graph
void RemoveSubGraphFromGraph(const GraphNodeSet& cluster,
                             const GraphNodeSet& cluster_internals,
                             Graph* graph) {
  for (auto* op_node : cluster) {
    graph->RemoveNode(op_node);
  }
  for (auto* var_node : cluster_internals) {
    graph->RemoveNode(var_node);
  }
}

// Replacing Cinn subgraph to a special op node, whose op_type is
// kCinnLaunchOp, and inputs ares cluster_inputs and outputs are
// cluster_outputs.
// Meanwhile, move all links of cluster to the special op.
void ReplaceSubGraphWithSpecialOpNode(const GraphNodeSet& cluster,
                                      const GraphNodeSet& cluster_inputs,
                                      const GraphNodeSet& cluster_outputs,
                                      const GraphNodeSet& cluster_internals,
                                      Graph* graph) {
  // First, add the special op node whose name is "kCinnLaunchOp" into graph
  auto special_op_node =
      AddSpecialOpToGraph(graph, cluster_inputs, cluster_outputs);
  // Second, remove all graph's links which are from or to cluster nodes
  RemoveLinkFromCluster(cluster, cluster_inputs, cluster_outputs);
  // Third, add new links from or to the the special op node
  AddLinkToSpecialOp(special_op_node, cluster_inputs, cluster_outputs);
  // Finally, remove the cinn sub graph from graph
  RemoveSubGraphFromGraph(cluster, cluster_internals, graph);
}

// Search all subgraphs which all op node supported by CINN,
// Here we using SubgraphDetector to detecte the subgraph that
// all of op node supported by CINN. We using OpMapperRegistry
// to check whether the op node supported by CINN.
void SearchAllSubgraphs(Graph* graph,
                        std::vector<std::unique_ptr<Graph>>* cinn_subgraphs) {
  auto teller = [](const Node* node) {
    return ::cinn::frontend::OpMapperRegistry::Global()->Find(node->Name()) !=
           nullptr;
  };
  std::vector<GraphNodeVec> clusters =
      framework::ir::SubgraphDetector(graph, teller)();

  cinn_subgraphs->clear();
  for (const auto& node_vec : clusters) {
    // classify var node to inputs, outputs, and internals.
    GraphNodeSet cluster_set(node_vec.begin(), node_vec.end());

    GraphNodeSet cluster_inputs, cluster_outputs, cluster_internals;
    AnalyseClusterVariables(cluster_set, &cluster_inputs, &cluster_outputs,
                            &cluster_internals);

    cinn_subgraphs->emplace_back(
        CreateNewSubGraph(cluster_set, cluster_internals));

    // replacing subgraph to a new special op node
    ReplaceSubGraphWithSpecialOpNode(cluster_set, cluster_inputs,
                                     cluster_outputs, cluster_internals, graph);
  }
}

void BuildCinnPass::ApplyImpl(Graph* graph) const {
  auto& cinn_subgraphs =
      Get<std::vector<std::unique_ptr<Graph>>>("cinn_subgraphs");
  SearchAllSubgraphs(graph, &cinn_subgraphs);
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(build_cinn_pass, paddle::framework::paddle2cinn::BuildCinnPass);
