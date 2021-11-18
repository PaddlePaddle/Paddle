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

#include <algorithm>
#include <iterator>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/operators/cinn_launch_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

DECLARE_string(allow_cinn_ops);
DECLARE_string(deny_cinn_ops);

namespace paddle {
namespace framework {
namespace paddle2cinn {

using framework::ir::Graph;
using framework::ir::Node;

using GraphNodeVec = std::vector<Node*>;
using GraphNodeSet = std::unordered_set<Node*>;
using GraphNodeMap = std::unordered_map<Node*, Node*>;

namespace {
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";

const std::unordered_map<std::string, std::unordered_set<std::string>>
    kDenyParamMap = {{"batch_norm", {"ReserveSpace"}},
                     {"batch_norm_grad", {"ReserveSpace"}}};

std::unordered_set<std::string> GetDenyVarNames(const GraphNodeSet& cluster) {
  std::unordered_set<std::string> deny_var_set;

  auto get_debug_info = [](const std::unordered_set<std::string>& var_names) {
    std::string debug_info = "[";
    for (auto& var : var_names) {
      debug_info.append(var);
      debug_info.append(", ");
    }
    debug_info.append("]");
    return debug_info;
  };

  for (auto* op : cluster) {
    if (kDenyParamMap.count(op->Name())) {
      const auto* desc = op->Op();
      PADDLE_ENFORCE_NE(desc, nullptr,
                        platform::errors::PreconditionNotMet(
                            "The Op %s's OpDesc should not be NULL, which has "
                            "a parameter in kDenyParamMap.",
                            op->Name().c_str()));

      auto deny_param_names = kDenyParamMap.at(op->Name());
      VLOG(4) << "We found deny param " << get_debug_info(deny_param_names)
              << " in op [" << op->Name() << "].";

      for (const auto& param_name : deny_param_names) {
        if (desc->Inputs().count(param_name)) {
          const auto& arg_names = desc->Input(param_name);
          for (const auto& arg_name : arg_names) {
            deny_var_set.insert(arg_name);
            VLOG(4) << "deny param [" << param_name << "]'s argument name"
                    << " is [" << arg_name << "].";
          }
        }

        if (desc->HasOutput(param_name)) {
          const auto& arg_names = desc->Output(param_name);
          for (const auto& arg_name : arg_names) {
            deny_var_set.insert(arg_name);
            VLOG(4) << "deny param [" << param_name << "]'s argument name"
                    << " is [" << arg_name << "].";
          }
        }
      }
    }
  }

  VLOG(4) << "All deny var names are " << get_debug_info(deny_var_set);

  return deny_var_set;
}

std::unordered_set<std::string> StringSplit(const std::string& str,
                                            const std::string& delim) {
  std::regex reg(delim);
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator()};
  elems.erase("");
  return elems;
}

int ExtractOpRole(const GraphNodeSet& cluster) {
  std::unordered_set<int> op_roles;
  std::string attr_name = OpProtoAndCheckerMaker::OpRoleAttrName();
  for (auto* n : cluster) {
    if (n->Op() && n->Op()->HasAttr(attr_name)) {
      op_roles.insert(BOOST_GET_CONST(int, n->Op()->GetAttr(attr_name)));
    }
  }
  if (op_roles.size() == 1U) {
    return *(op_roles.begin());
  } else {
    return static_cast<int>(OpRole::kNotSpecified);
  }
}

// Deal with subgraph's feed input var node:
// create a new input var node and it's feed op node
void AddFeedOpAndVar(const GraphNodeSet& feed_vars, const GraphNodeSet& cluster,
                     const GraphNodeMap& old_op2new_op,
                     const GraphNodeMap& old_var2new_var, Graph* graph) {
  for (auto* old_var : feed_vars) {
    // create feed op
    OpDesc desc;
    desc.SetType("feed");
    desc.SetOutput("Out", {old_var->Name()});
    auto op = graph->CreateOpNode(&desc);

    // get new feed var node
    auto* var = old_var2new_var.at(old_var);
    VLOG(4) << "Add Feed Op before: " << var->Name();

    // link feed op and feed var
    IR_NODE_LINK_TO(op, var);

    // link feed var to cluster op
    for (auto* old_op : old_var->outputs) {
      if (cluster.count(old_op)) {
        IR_NODE_LINK_TO(var, old_op2new_op.at(old_op));
      }
      // Do not need relink old op or old var here, they will be
      // fixed in RemoveSubGraphFromGraph, here we just deal with
      // new subgraph's node.
    }
  }
}

// Deal with subgraph's parameter var node:
// create a new input var node, it's data will get by scope,
// so it don't need feed op
void AddParamVar(const GraphNodeSet& param_vars, const GraphNodeSet& cluster,
                 const GraphNodeMap& old_op2new_op,
                 const GraphNodeMap& old_var2new_var, Graph* graph) {
  for (auto* old_var : param_vars) {
    auto* var = old_var2new_var.at(old_var);
    VLOG(4) << "Add Param Var Node: " << var->Name();

    for (auto* old_op : old_var->outputs) {
      if (cluster.count(old_op)) {
        IR_NODE_LINK_TO(var, old_op2new_op.at(old_op));
      }
    }
  }
}

// Deal with subgraph's outputs var node:
// create a new output var node and it's fetch op
void AddOutputVar(const GraphNodeSet& output_vars, const GraphNodeSet& cluster,
                  const GraphNodeMap& old_op2new_op,
                  const GraphNodeMap& old_var2new_var, Graph* graph) {
  for (auto* old_var : output_vars) {
    // create fetch op
    OpDesc desc;
    desc.SetType("fetch");
    desc.SetInput("X", {old_var->Name()});
    auto op = graph->CreateOpNode(&desc);

    auto* var = old_var2new_var.at(old_var);
    VLOG(4) << "Add Output Var Node: " << var->Name();

    // link fetch op and fetch var
    IR_NODE_LINK_TO(var, op);

    for (auto* old_op : old_var->inputs) {
      if (cluster.count(old_op)) {
        IR_NODE_LINK_TO(old_op2new_op.at(old_op), var);
      }
    }
  }
}

// Create new subgraph with and op nodes are cluster nodes, and all
// var node are from internal nodes
std::unique_ptr<Graph> CreateNewSubGraph(const GraphNodeSet& cluster,
                                         const GraphNodeSet& cluster_internals,
                                         const GraphNodeSet& cluster_inputs,
                                         const GraphNodeSet& cluster_outputs) {
  // Graph's constructor must has one parameter, and in our code,
  // the ProgramDesc is useless, so here we pass a temporary object.
  auto subgraph = std::make_unique<Graph>(framework::ProgramDesc());

  GraphNodeMap old_op2new_op;
  for (auto* op : cluster) {
    auto sub_node = subgraph->CreateOpNode(op->Op());
    old_op2new_op[op] = sub_node;
  }

  GraphNodeMap old_var2new_var;
  for (auto* var : cluster_internals) {
    PADDLE_ENFORCE_NOT_NULL(var->Var(),
                            platform::errors::PreconditionNotMet(
                                "The var desc of the node in cluster_internals "
                                "shouldn't be null."));
    auto* sub_node = subgraph->CreateVarNode(var->Var());
    old_var2new_var[var] = sub_node;
  }
  for (auto* var : cluster_inputs) {
    if (var->Var()) {
      auto* sub_node = subgraph->CreateVarNode(var->Var());
      old_var2new_var[var] = sub_node;
    }
  }
  for (auto* var : cluster_outputs) {
    if (var->Var()) {
      auto* sub_node = subgraph->CreateVarNode(var->Var());
      old_var2new_var[var] = sub_node;
    }
  }

  GraphNodeSet need_feed_vars;
  std::unordered_set<Node *> param_vars, output_vars;
  // the subgraph is independently, so here we only need link
  // to the node in new subgraph, and discard the link to
  // out-graph.
  for (auto* op : cluster) {
    for (auto* var : op->inputs) {
      // one output var maybe an input of the cluster
      if (cluster_internals.count(var) ||
          (cluster_outputs.count(var) && old_var2new_var.count(var))) {
        IR_NODE_LINK_TO(old_var2new_var.at(var), old_op2new_op.at(op));
      } else if (cluster_inputs.count(var) && var->Var() != nullptr) {
        if (var->Var()->IsParameter()) {
          // Parameters have been preserved in scope, compared to feed var,
          // param just need add new var and don't need add feed op.
          // The var is used for check whether we need preserve the tensor
          // when transform paddle scope to CINN scope.
          param_vars.insert(var);
        } else {
          // When the var is subgraph input and the var is not parameter,
          // we need add a new feed op to feed the var.
          need_feed_vars.insert(var);
        }
      }
    }
    for (auto* var : op->outputs) {
      if (cluster_internals.count(var)) {
        IR_NODE_LINK_TO(old_op2new_op.at(op), old_var2new_var.at(var));
      } else if (cluster_outputs.count(var) && var->Var() != nullptr) {
        // Create new output var node to guarantee the independency of
        // subgraph. In other words, the subgraph has no connection with
        // other graph, even the input graph.
        output_vars.insert(var);
      }
    }
  }

  AddFeedOpAndVar(need_feed_vars, cluster, old_op2new_op, old_var2new_var,
                  subgraph.get());
  AddParamVar(param_vars, cluster, old_op2new_op, old_var2new_var,
              subgraph.get());
  AddOutputVar(output_vars, cluster, old_op2new_op, old_var2new_var,
               subgraph.get());

  return subgraph;
}

// This interface is used to classify all variables involved in a cluster into
// three types: inputs, outputs, and internals.
// The input node is some subgraph op's input but not any subgraph op's output.
// The output node is some subgraph op's output and some out-graph op's input.
// Specially, the internal node is a node that only used by subgraph, and
// out-graph should not using this node at all.
// cluster_inputs & cluster_outputs & cluster_internals == NULL
// cluster_outputs | cluster_internals == all graph op's outputs node
void AnalyseClusterVariables(
    const GraphNodeSet& cluster,
    const std::unordered_set<std::string>& deny_var_set,
    GraphNodeSet* cluster_inputs, GraphNodeSet* cluster_outputs,
    GraphNodeSet* cluster_internals) {
  // collecting all input and output of op
  for (auto* op_node : cluster) {
    const auto& op_name = op_node->Name();
    for (auto* input_var_node : op_node->inputs) {
      if (!deny_var_set.count(input_var_node->Name())) {
        // ignore deny var node
        cluster_inputs->insert(input_var_node);
      }
    }
    for (auto* output_var_node : op_node->outputs) {
      if (!deny_var_set.count(output_var_node->Name())) {
        cluster_outputs->insert(output_var_node);
      }
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

  // if a output node also exists in internal list, remove.
  for (auto* var_node : *cluster_internals) {
    cluster_outputs->erase(var_node);
  }
}

void AddLinkToCinnOp(const GraphNodeSet& cluster_inputs,
                     const GraphNodeSet& cluster_outputs, Node* cinn_op_node) {
  // add new link from cluster_inputs to cinn_op_node
  for (auto* var_node : cluster_inputs) {
    IR_NODE_LINK_TO(var_node, cinn_op_node);
  }

  // add new link from cinn_op_node to cluster_outputs
  for (auto* var_node : cluster_outputs) {
    IR_NODE_LINK_TO(cinn_op_node, var_node);
  }
}

void AddCinnOpToGraph(const GraphNodeSet& cluster,
                      const GraphNodeSet& cluster_inputs,
                      const GraphNodeSet& cluster_outputs,
                      const std::string& compilation_key,
                      const std::unordered_set<std::string>& deny_var_set,
                      Graph* graph) {
  // Add the cinn launch op
  framework::OpDesc cinn_op_desc;
  cinn_op_desc.SetType(kCinnLaunchOp);
  std::vector<std::string> input_names;

  std::for_each(cluster_inputs.begin(), cluster_inputs.end(),
                [&input_names, &deny_var_set](Node* n) {
                  if (n->Var() != nullptr && !deny_var_set.count(n->Name())) {
                    input_names.emplace_back(n->Name());
                  }
                });
  cinn_op_desc.SetInput(operators::kX, input_names);
  std::vector<std::string> output_names;
  std::for_each(cluster_outputs.begin(), cluster_outputs.end(),
                [&output_names, &deny_var_set](Node* n) {
                  if (n->Var() != nullptr && !deny_var_set.count(n->Name())) {
                    output_names.emplace_back(n->Name());
                  }
                });
  cinn_op_desc.SetOutput(operators::kOutputs, output_names);
  cinn_op_desc.SetAttr(operators::kCompilationKey, compilation_key);
  cinn_op_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                       ExtractOpRole(cluster));
  cinn_op_desc.Flush();
  auto* cinn_op_node = graph->CreateOpNode(&cinn_op_desc);
  // Add new links from or to the the cinn launch op node
  AddLinkToCinnOp(cluster_inputs, cluster_outputs, cinn_op_node);

  VLOG(4) << "Add op [" << kCinnLaunchOp << "] into graph.";
}

// Removing cluster node and internals node from Graph
void RemoveSubGraphFromGraph(const GraphNodeSet& cluster,
                             const GraphNodeSet& cluster_internals,
                             Graph* graph) {
  const std::unordered_set<const Node*> const_cluster{cluster.cbegin(),
                                                      cluster.cend()};
  const std::unordered_set<const Node*> const_internals{
      cluster_internals.cbegin(), cluster_internals.cend()};
  ir::GraphSafeRemoveNodes(graph, const_cluster);
  ir::GraphSafeRemoveNodes(graph, const_internals);
}

// Replacing Cinn subgraph to a cinn op node, whose op_type is
// kCinnLaunchOp, and inputs ares cluster_inputs and outputs are
// cluster_outputs.
// Meanwhile, move all links of cluster to the cinn op.
void ReplaceSubGraphWithCinnOpNode(
    const GraphNodeSet& cluster, const GraphNodeSet& cluster_inputs,
    const GraphNodeSet& cluster_outputs, const GraphNodeSet& cluster_internals,
    const std::string& compilation_key,
    const std::unordered_set<std::string>& deny_var_set, Graph* graph) {
  // Add the cinn op node whose name is "kCinnLaunchOp" into graph
  AddCinnOpToGraph(cluster, cluster_inputs, cluster_outputs, compilation_key,
                   deny_var_set, graph);
  // Remove the cinn subgraph from graph
  RemoveSubGraphFromGraph(cluster, cluster_internals, graph);
}

// Search all subgraphs which all op node supported by CINN,
// Here we using SubgraphDetector to detecte the subgraph that
// all of op node supported by CINN. We using OpMapperRegistry
// to check whether the op node supported by CINN.
void SearchAllSubgraphs(Graph* graph) {
  auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
  auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
  auto teller = [&allow_ops, &deny_ops](const Node* node) {
    bool registered = ::cinn::frontend::OpMapperRegistry::Global()->Find(
                          node->Name()) != nullptr;
    // if the op type is registered in CINN and allow_ops is not empty, return
    // true only when it is in allow_ops
    if (allow_ops.size()) {
      return registered && allow_ops.count(node->Name());
    }
    // if the op type is registered in CINN and deny_ops is not empty, return
    // true only when it is not in deny_ops
    if (deny_ops.size()) {
      return registered && !deny_ops.count(node->Name());
    }
    // if the user doesn't set FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops,
    // return true only when it is registered in CINN
    return registered;
  };
  VLOG(4) << "The allowed Cinn Ops: " << FLAGS_allow_cinn_ops;
  VLOG(4) << "The denied Cinn Ops: " << FLAGS_deny_cinn_ops;
  std::vector<GraphNodeVec> clusters =
      framework::ir::SubgraphDetector(graph, teller)();

  auto cluster_debug_info = [](const GraphNodeSet& cluster) {
    std::string res = "(";
    for (auto* node : cluster) {
      res.append(node->Name());
      res.append(", ");
    }
    res.append(")");
    return res;
  };

  auto* cinn_compiler = CinnCompiler::GetInstance();
  for (const auto& node_vec : clusters) {
    // Classify var node to inputs, outputs, and internals.
    GraphNodeSet cluster_set(node_vec.begin(), node_vec.end());

    auto deny_var_set = GetDenyVarNames(cluster_set);

    GraphNodeSet cluster_inputs, cluster_outputs, cluster_internals;
    AnalyseClusterVariables(cluster_set, deny_var_set, &cluster_inputs,
                            &cluster_outputs, &cluster_internals);

    VLOG(4) << "Cluster Ops: " << cluster_debug_info(cluster_set);
    VLOG(4) << "Cluster input vars: " << cluster_debug_info(cluster_inputs);
    VLOG(4) << "Cluster output vars: " << cluster_debug_info(cluster_outputs);
    VLOG(4) << "Cluster internal vars: "
            << cluster_debug_info(cluster_internals);

    // Create a new subgraph according to the found cluster and
    // save it in CinnCompiler
    std::string compilation_key = cinn_compiler->AddGraph(CreateNewSubGraph(
        cluster_set, cluster_internals, cluster_inputs, cluster_outputs));
    VLOG(4) << "Compilation Key:\n"
            << cinn_compiler->ReadableKey(compilation_key);

    // Replace the found cluster to a new cinn op node
    ReplaceSubGraphWithCinnOpNode(cluster_set, cluster_inputs, cluster_outputs,
                                  cluster_internals, compilation_key,
                                  deny_var_set, graph);
  }
}
}  // namespace

void BuildCinnPass::ApplyImpl(Graph* graph) const { SearchAllSubgraphs(graph); }

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(build_cinn_pass, paddle::framework::paddle2cinn::BuildCinnPass);
