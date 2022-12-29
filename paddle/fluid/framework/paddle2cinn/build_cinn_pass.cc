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
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_subgraph_detector.h"
#include "paddle/fluid/operators/cinn/cinn_launch_op.h"
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
using GraphNodeMap = std::unordered_map<Node*, Node*>;

OpTransInfo::OpTransInfo() {
  // judgment condition for the dynamic slice
  dynamic_op_cond_.emplace("slice", [](const ir::Node& node) -> bool {
    if (!node.IsOp()) {
      return false;
    }
    auto* op_desc = node.Op();
    auto infer_flags =
        op_desc->GetAttrIfExists<std::vector<int>>("infer_flags");
    return std::find_if(infer_flags.begin(), infer_flags.end(), [](int v) {
             return v < 0;
           }) != infer_flags.end();
  });

  // judgment condition for the dynamic reshape
  dynamic_op_cond_.emplace("reshape", [](const ir::Node& node) -> bool {
    if (!node.IsOp()) {
      return false;
    }
    auto* op_desc = node.Op();
    bool has_shape_tensor = op_desc->Inputs().count("ShapeTensor") &&
                            op_desc->Inputs().at("ShapeTensor").size();
    bool has_shape = op_desc->Inputs().count("Shape") &&
                     op_desc->Inputs().at("Shape").size();
    return has_shape_tensor || has_shape;
  });

  // judgment condition for the dynamic reshape2
  dynamic_op_cond_.emplace("reshape2", dynamic_op_cond_.at("reshape"));

  // judgment condition for the dynamic expand
  dynamic_op_cond_.emplace("expand", [](const ir::Node& node) -> bool {
    if (!node.IsOp()) {
      return false;
    }
    auto* op_desc = node.Op();
    bool has_expand_times_tensor =
        op_desc->Inputs().count("expand_times_tensor") &&
        op_desc->Inputs().at("expand_times_tensor").size();
    bool has_expand_times = op_desc->Inputs().count("ExpandTimes") &&
                            op_desc->Inputs().at("ExpandTimes").size();
    return has_expand_times_tensor || has_expand_times;
  });

  // judgment condition for the dynamic expand_v2
  dynamic_op_cond_.emplace("expand_v2", [](const ir::Node& node) -> bool {
    if (!node.IsOp()) {
      return false;
    }
    auto* op_desc = node.Op();
    bool has_expand_shapes_tensor =
        op_desc->Inputs().count("expand_shapes_tensor") &&
        op_desc->Inputs().at("expand_shapes_tensor").size();
    bool has_shape = op_desc->Inputs().count("Shape") &&
                     op_desc->Inputs().at("Shape").size();
    return has_expand_shapes_tensor || has_shape;
  });
}

std::unordered_set<std::string> OpTransInfo::GetDenyVarNames(
    const GraphNodeSet& cluster) const {
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
    if (deny_param_cond_.count(op->Name())) {
      const auto* desc = op->Op();
      PADDLE_ENFORCE_NE(desc,
                        nullptr,
                        platform::errors::PreconditionNotMet(
                            "The Op %s's OpDesc should not be NULL, which has "
                            "a parameter in deny_param_cond_.",
                            op->Name().c_str()));

      auto deny_param_names = deny_param_cond_.at(op->Name());
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

bool OpTransInfo::IsInplaceOp(const OpDesc& op_desc) {
  auto inputs = op_desc.InputArgumentNames();
  std::unordered_set<std::string> input_set(inputs.begin(), inputs.end());
  for (auto& name : op_desc.OutputArgumentNames()) {
    if (input_set.count(name) > 0) return true;
  }
  return false;
}

namespace {
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";

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
      op_roles.insert(PADDLE_GET_CONST(int, n->Op()->GetAttr(attr_name)));
    }
  }
  if (op_roles.size() == 1U) {
    return *(op_roles.begin());
  } else {
    return static_cast<int>(OpRole::kNotSpecified);
  }
}

// Deal with input var nodes of the target subgraph:
// create a new input var node and it's feed op node
void AddFeedOpAndVar(const GraphNodeSet& input_vars,
                     const GraphNodeSet& cluster,
                     const GraphNodeMap& old_op2new_op,
                     const GraphNodeMap& old_var2new_var,
                     Graph* graph) {
  for (auto* old_var : input_vars) {
    // create feed op
    OpDesc desc;
    desc.SetType("feed");
    desc.SetOutput("Out", {old_var->Name()});
    auto op = graph->CreateOpNode(&desc);

    // get new feed var node
    auto* var = old_var2new_var.at(old_var);
    VLOG(4) << "Add Feed Op before the input var: " << var->Name();

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

// Deal with subgraph's outputs var node:
// create a new output var node and it's fetch op
void AddOutputVar(const GraphNodeSet& output_vars,
                  const GraphNodeSet& cluster,
                  const GraphNodeMap& old_op2new_op,
                  const GraphNodeMap& old_var2new_var,
                  Graph* graph) {
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

std::unordered_set<std::string> ExtractNoNeedBufferFeeds(
    const GraphNodeSet& cluster, const GraphNodeSet& cluster_inputs) {
  // 1. Find op with NoNeedBufferVarsInferer defined and collect its input nodes
  std::unordered_map<Node*, GraphNodeSet> op_node2no_need_buffer_nodes;
  for (auto* op_node : cluster) {
    const auto* op = OpInfoMap::Instance().GetNullable(op_node->Name());
    // If op not registered in Paddle, skip
    if (!op) {
      continue;
    }
    auto& inferer = op->NoNeedBufferVarsInferer();
    if (!inferer) {
      continue;
    }
    auto* op_desc = op_node->Op();
    PADDLE_ENFORCE_NOT_NULL(
        op_desc,
        platform::errors::PreconditionNotMet(
            "The op desc of node in cluster shouldn't be null."));
    auto inferred_params =
        inferer(op_desc->Inputs(), op_desc->Inputs(), op_desc->GetAttrMap());
    std::unordered_set<std::string> inferred_args;
    std::for_each(inferred_params.begin(),
                  inferred_params.end(),
                  [&op_desc, &inferred_args](const std::string& param) {
                    const auto& args = op_desc->Input(param);
                    inferred_args.insert(args.begin(), args.end());
                  });
    auto& no_need_buffer_nodes = op_node2no_need_buffer_nodes[op_node];
    for (auto* input_node : op_node->inputs) {
      if (input_node->Var() && inferred_args.count(input_node->Name())) {
        VLOG(4) << "Input node(" << input_node->Name() << ") of op("
                << op_node->Name() << ") is no_need_buffer";
        no_need_buffer_nodes.insert(input_node);
      }
    }
  }

  // 2. Extract no_need_buffer nodes from cluster_inputs by checking
  // all of their outputs are op nodes with NoNeedBufferVarsInferer
  // and they used as no_need_buffer inputs.
  auto check_all_used_as_no_need_buffer_fn =
      [&op_node2no_need_buffer_nodes](Node* var_node) -> bool {
    for (auto* output_node : var_node->outputs) {
      auto it = op_node2no_need_buffer_nodes.find(output_node);
      if (it == op_node2no_need_buffer_nodes.end()) {
        VLOG(4) << "Var node(" << var_node->Name() << ")'s output node("
                << output_node->Name()
                << ") doesn't have NoNeedBufferVarsInferer";
        return false;
      }
      if (it->second.count(var_node) == 0) {
        VLOG(4) << "Var node("
                << ") is not used as no_need_buffer inputs";
        return false;
      }
    }
    return true;
  };
  std::unordered_set<std::string> result;
  for (const auto& op2inputs_pair : op_node2no_need_buffer_nodes) {
    for (auto* input_node : op2inputs_pair.second) {
      if (cluster_inputs.count(input_node) &&
          check_all_used_as_no_need_buffer_fn(input_node)) {
        VLOG(4) << "Input node(" << input_node->Name()
                << ") is declared as no_need_buffer cluster_inputs";
        result.insert(input_node->Name());
      }
    }
  }
  return result;
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
    if (!var->Var()) {
      // skip control var

      // TODO(jiangcheng05): CINN not support control var now, so here we skip
      // it, but it may incur result incorrect problem. In detail, for two
      // unconnected ops, with control var, an op must run before another op.
      // If we remove the control var, the program wouldn't guarantee the run
      // ordering, in other words, the result may incorrect.
      VLOG(4)
          << "The internal var [" << var->Name() << "]'s vardesc empty,"
          << " it may be a control var, but CINN not support control var now.";
      continue;
    }
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
  std::unordered_set<Node*> param_vars, output_vars;
  // the subgraph is independently, so here we only need link
  // to the node in new subgraph, and discard the link to
  // out-graph.
  for (auto* op : cluster) {
    for (auto* var : op->inputs) {
      if (!var->Var()) {
        // skip control var
        continue;
      }
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
      if (!var->Var()) {
        // skip control var
        continue;
      }
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

  AddFeedOpAndVar(
      need_feed_vars, cluster, old_op2new_op, old_var2new_var, subgraph.get());
  AddFeedOpAndVar(
      param_vars, cluster, old_op2new_op, old_var2new_var, subgraph.get());
  AddOutputVar(
      output_vars, cluster, old_op2new_op, old_var2new_var, subgraph.get());
  // Save lists of input variables, internal variables and output variables
  // of the cluster as attributes of the subgraph for convenience.
  auto collect_names_fn =
      [](const GraphNodeSet& nodes,
         const std::unordered_set<std::string>& ignore_names) {
        auto result = std::make_unique<std::vector<std::string>>();
        for (auto* node : nodes) {
          if (!node->Var() || ignore_names.count(node->Name())) {
            continue;
          }
          result->emplace_back(node->Name());
        }
        return result;
      };
  subgraph->Set<std::vector<std::string>>(
      kInternalVars, collect_names_fn(cluster_internals, {}).release());
  subgraph->Set<std::vector<std::string>>(
      kOutputVars, collect_names_fn(cluster_outputs, {}).release());
  // Divide input variables into two parts: one is common and will be used
  // in execution, the other may be empty and it is those variables whose
  // buffer are not needed and only be used in graph symbolization
  auto no_need_buffer_feeds = std::make_unique<std::unordered_set<std::string>>(
      ExtractNoNeedBufferFeeds(cluster, cluster_inputs));
  subgraph->Set<std::vector<std::string>>(
      kInputVars,
      collect_names_fn(cluster_inputs, *no_need_buffer_feeds).release());
  subgraph->Set<std::unordered_set<std::string>>(
      kNoNeedBufferFeeds, no_need_buffer_feeds.release());
  // initialize empty map for kMemOptVarInfoFromMainGraph attribute,
  // it will be filled on the share_mem_opt_info_to_subgraph pass
  subgraph->GetOrInit<Name2VarInfoMap>(kMemOptVarInfoFromMainGraph);
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
    GraphNodeSet* cluster_inputs,
    GraphNodeSet* cluster_outputs,
    GraphNodeSet* cluster_internals,
    bool is_inference_stage) {
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

  if (is_inference_stage) {
    // If part of the output of the Op is not used by other operators, change it
    // to internal. such as transpose2 op's XShape out.
    auto outs = *cluster_outputs;
    for (auto* node : outs) {
      if (node->outputs.empty()) {
        cluster_outputs->erase(node);
        cluster_internals->insert(node);
      }
    }
  }
}

void AddLinkToCinnOp(const GraphNodeSet& cluster_inputs,
                     const GraphNodeSet& cluster_outputs,
                     Node* cinn_op_node) {
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
                      int64_t compilation_key,
                      const std::unordered_set<std::string>& deny_var_set,
                      Graph* graph) {
  // Add the cinn launch op
  framework::OpDesc cinn_op_desc;
  cinn_op_desc.SetType(kCinnLaunchOp);

  const auto& subgraph =
      CinnCompiler::GetInstance()->FindGraph(compilation_key);
  const auto& no_need_buffer_feeds =
      subgraph.Get<std::unordered_set<std::string>>(kNoNeedBufferFeeds);

  cinn_op_desc.SetInput(operators::kX,
                        subgraph.Get<std::vector<std::string>>(kInputVars));
  cinn_op_desc.SetInput(operators::kNoNeedBufferX,
                        std::vector<std::string>(no_need_buffer_feeds.begin(),
                                                 no_need_buffer_feeds.end()));
  cinn_op_desc.SetOutput(operators::kOutputs,
                         subgraph.Get<std::vector<std::string>>(kOutputVars));
  cinn_op_desc.SetAttr(operators::kCompilationKey, compilation_key);
  cinn_op_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                       ExtractOpRole(cluster));
  cinn_op_desc.Flush();
  auto* cinn_op_node = graph->CreateOpNode(&cinn_op_desc);
  // Add new links from or to the cinn launch op node
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
    const GraphNodeSet& cluster,
    const GraphNodeSet& cluster_inputs,
    const GraphNodeSet& cluster_outputs,
    const GraphNodeSet& cluster_internals,
    int64_t compilation_key,
    const std::unordered_set<std::string>& deny_var_set,
    Graph* graph) {
  // Add the cinn op node whose name is "kCinnLaunchOp" into graph
  AddCinnOpToGraph(cluster,
                   cluster_inputs,
                   cluster_outputs,
                   compilation_key,
                   deny_var_set,
                   graph);
  // Remove the cinn subgraph from graph
  RemoveSubGraphFromGraph(cluster, cluster_internals, graph);
}

// Search all subgraphs which all op node supported by CINN,
// Here we using SubgraphDetector to detecte the subgraph that
// all of op node supported by CINN. We using OpMapperRegistry
// to check whether the op node supported by CINN.
void SearchAllSubgraphs(Graph* graph, bool is_inference_stage) {
  auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
  auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
  OpTransInfo trans_info;
  auto teller = [&allow_ops, &deny_ops, &trans_info](const Node* node) {
    const auto& node_name = node->Name();
    bool registered = ::cinn::frontend::OpMapperRegistry::Global()->Find(
                          node_name) != nullptr;
    // skip the dynamic ops
    bool is_dynamic = false;
    if (trans_info.dynamic_op_cond().count(node_name)) {
      is_dynamic = trans_info.dynamic_op_cond().at(node_name)(*node);
    }

    bool is_support =
        registered && !trans_info.default_deny_ops().count(node_name) &&
        !is_dynamic && (node->IsOp() && !trans_info.IsInplaceOp(*node->Op()));
    // if the op type is registered in CINN and allow_ops is not empty, return
    // true only when it is in allow_ops
    if (!allow_ops.empty()) {
      return is_support && allow_ops.count(node_name);
    }
    // if the op type is registered in CINN and deny_ops is not empty, return
    // true only when it is not in deny_ops
    if (!deny_ops.empty()) {
      return is_support && !deny_ops.count(node_name);
    }

    // if the user doesn't set FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops,
    // return true only when it is registered in CINN
    return is_support;
  };
  VLOG(4) << "The allowed Cinn Ops: " << FLAGS_allow_cinn_ops;
  VLOG(4) << "The denied Cinn Ops: " << FLAGS_deny_cinn_ops;
  std::vector<GraphNodeVec> clusters = CinnSubgraphDetector(graph, teller)();
  LOG(INFO) << "--- [build_cinn_pass] detected " << clusters.size()
            << " cinn supported subgraphs";

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

    auto deny_var_set = trans_info.GetDenyVarNames(cluster_set);

    GraphNodeSet cluster_inputs, cluster_outputs, cluster_internals;
    AnalyseClusterVariables(cluster_set,
                            deny_var_set,
                            &cluster_inputs,
                            &cluster_outputs,
                            &cluster_internals,
                            is_inference_stage);

    VLOG(4) << "Cluster Ops: " << cluster_debug_info(cluster_set);
    VLOG(4) << "Cluster input vars: " << cluster_debug_info(cluster_inputs);
    VLOG(4) << "Cluster output vars: " << cluster_debug_info(cluster_outputs);
    VLOG(4) << "Cluster internal vars: "
            << cluster_debug_info(cluster_internals);

    // Create a new subgraph according to the found cluster and
    // save it in CinnCompiler
    auto compilation_key = cinn_compiler->AddGraph(CreateNewSubGraph(
        cluster_set, cluster_internals, cluster_inputs, cluster_outputs));
    VLOG(4) << "Compilation Key:\n"
            << cinn_compiler->ReadableKey(compilation_key);

    // Replace the found cluster to a new cinn op node
    ReplaceSubGraphWithCinnOpNode(cluster_set,
                                  cluster_inputs,
                                  cluster_outputs,
                                  cluster_internals,
                                  compilation_key,
                                  deny_var_set,
                                  graph);
  }
}
}  // namespace

void BuildCinnPass::ApplyImpl(Graph* graph) const {
  bool is_inference_stage{false};
  if (Has("is_inference_stage")) {
    is_inference_stage = Get<bool>("is_inference_stage");
  }
  SearchAllSubgraphs(graph, is_inference_stage);
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(build_cinn_pass, paddle::framework::paddle2cinn::BuildCinnPass);
