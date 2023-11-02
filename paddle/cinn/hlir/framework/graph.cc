// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/graph.h"

#include <atomic>
#include <sstream>

#include "paddle/cinn/hlir/framework/visualize_helper.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#endif
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

using DTypeDict = absl::flat_hash_map<std::string, common::Type>;
using ShapeDict = absl::flat_hash_map<std::string, shape_t>;

void Graph::Initialize(const frontend::Program& prog,
                       const std::unordered_set<std::string>& fetch_var_ids,
                       const Target& target) {
  target_ = target;
  ShapeDict shape_dict;
  DTypeDict dtype_dict;
  int counter = 0;
  for (size_t i = 0; i < prog.size(); i++) {
    auto temp = prog[i];
    VLOG(3) << "operator [" << temp->op_type << "] has [" << temp->inputs.size()
            << "] inputs, and [" << temp->outputs.size() << "] outputs";
    Node* node_tmp = new Node(Operator::Get(temp->op_type),
                              temp->op_type,
                              temp->op_type + "_" + std::to_string(counter++));
    Shared<Node> node_ptr(node_tmp);
    node_tmp->attrs.attr_store = temp->attrs;
    for (auto& input_v : temp->inputs) {
      common::GraphNode* graph_node = this->RetrieveNode(input_v->id);
      if (!graph_node) {
        dtype_dict[input_v->id] = input_v->type;
        shape_dict[input_v->id] = input_v->shape;
        NodeData* input_data =
            new NodeData(nullptr, 0, 0, input_v->id, input_v.is_const());
        input_data->LinkTo(node_tmp);
        this->RegisterNode(input_v->id, input_data);
      } else {
        graph_node->as<NodeData>()->LinkTo(node_tmp);
      }
    }
    int out_idx = 0;
    for (auto& output_v : temp->outputs) {
      common::GraphNode* graph_node = this->RetrieveNode(output_v->id);
      if (!graph_node) {
        dtype_dict[output_v->id] = output_v->type;
        shape_dict[output_v->id] = output_v->shape;
        auto* output_data = new NodeData(node_ptr, out_idx++, 0, output_v->id);
        if (fetch_var_ids.count(output_v->id)) {
          outputs.push_back(output_data);
        }
        node_tmp->LinkTo(output_data);
        this->RegisterNode(output_v->id, output_data);
      } else {
        node_tmp->LinkTo(graph_node->as<NodeData>());
        graph_node->as<NodeData>()->set_const(false);
        graph_node->as<NodeData>()->output_index = out_idx++;
        graph_node->as<NodeData>()->source_node = node_ptr;
      }
    }
    this->RegisterNode(node_tmp->id(), node_tmp);
  }
  this->attrs["infershape"] = std::make_shared<absl::any>(shape_dict);
  this->attrs["inferdtype"] = std::make_shared<absl::any>(dtype_dict);
}

std::vector<std::vector<Node*>> Graph::FusionGroupsToGroups() {
  std::vector<std::vector<Node*>> groups;
  if (fusion_groups.empty()) {
    // if no fusion_groups, the graph will be treated as a big group
    const auto& nodes = this->CollectNodes([](const common::GraphNode* node) {
      return node->safe_as<Node>() != nullptr &&
             node->safe_as<Node>()->op() != nullptr;
    });
    std::vector<Node*> group;
    group.reserve(nodes.size());
    for (auto* node : nodes) {
      group.emplace_back(node->safe_as<Node>());
    }
    groups.emplace_back(std::move(group));
  } else {
    groups.resize(fusion_groups.size());
    for (size_t i = 0; i < fusion_groups.size(); ++i) {
      groups[i] = fusion_groups[i]->CollectNodes();
    }
  }
  return groups;
}

std::string Graph::DebugGroupedGraph(
    const std::unordered_set<std::string>& fetch_var_ids) {
  if (!fusion_groups.empty()) {
    return DebugGroupedGraph(FusionGroupsToGroups(), fetch_var_ids);
  }

  std::vector<Node*> graph_ops;
  auto nodes_inorder = std::get<0>(topological_order());
  for (auto* graph_node : nodes_inorder) {
    auto node = graph_node->safe_as<Node>();
    // if node is NodeData or not op, continue.
    if (!node || node->op() == nullptr) {
      continue;
    }

    graph_ops.emplace_back(node);
  }

  std::stringstream debug_str;
  debug_str << "Graph {\n";
  debug_str << DebugGroupedGraph(graph_ops, fetch_var_ids);
  debug_str << "}\n";
  return debug_str.str();
}

std::string Graph::DebugGroupedGraph(
    const std::vector<Node*>& group,
    const std::unordered_set<std::string>& fetch_var_ids) {
  auto& shape_dict =
      HasAttr("infershape") ? GetAttrs<ShapeDict>("infershape") : ShapeDict{};
  auto& dtype_dict =
      HasAttr("inferdtype") ? GetAttrs<DTypeDict>("inferdtype") : DTypeDict{};

  auto get_all_out_names = [](const std::vector<Node*>& nodes) {
    // collect all op's output var name in group
    std::unordered_set<std::string> out_names;
    for (auto* node : nodes) {
      for (const auto& link : node->outlinks()) {
        auto* out_node = link->sink()->safe_as<NodeData>();
        out_names.emplace(out_node->id());
      }
    }
    return out_names;
  };
  auto get_feed_list = [](const std::vector<Node*>& nodes,
                          const std::unordered_set<std::string>& out_names) {
    // if the op's input var name cannot found in out_names, it is the group's
    // feed var
    std::unordered_set<std::string> feed_list;
    for (auto* node : nodes) {
      for (const auto& link : node->inlinks()) {
        auto* in_node = link->source()->safe_as<NodeData>();
        if (!out_names.count(in_node->id())) {
          feed_list.emplace(in_node->id());
        }
      }
    }
    return std::vector<std::string>(feed_list.begin(), feed_list.end());
  };
  auto get_fetch_list = [&](const std::vector<Node*>& nodes,
                            const std::unordered_set<std::string>& out_names) {
    // if the fetch var in out_names, it's the group's fetch var, otherwise not
    std::unordered_set<std::string> in_names;
    for (auto* node : nodes) {
      for (const auto& link : node->inlinks()) {
        auto* in_node = link->source()->safe_as<NodeData>();
        in_names.emplace(in_node->id());
      }
    }
    std::vector<std::string> fetch_list;
    for (const auto& out : out_names) {
      if (!in_names.count(out) || fetch_var_ids.count(out)) {
        // if the var not any op's input, or in fetch_var_ids, it's the group's
        // fetch list
        fetch_list.emplace_back(out);
      }
    }
    return fetch_list;
  };

  const auto& out_names = get_all_out_names(group);
  const auto& feed_list = get_feed_list(group, out_names);

  std::stringstream debug_str;
  // generator python test code
  for (const auto& id : feed_list) {
    const auto& shape = shape_dict.count(id)
                            ? cinn::utils::Join(shape_dict.at(id), ", ")
                            : "-1";
    const auto& dtype =
        dtype_dict.count(id) ? common::Type2Str(dtype_dict.at(id)) : "float32";

    // generator python create_input code
    debug_str << "    " << id << " = builder.create_input(type=\"" << dtype
              << "\", shape=[" << shape << "], id_hint=\"" << id << "\")\n";
  }
  debug_str << "\n";
  // generator builder.op code
  for (auto* node : group) {
    debug_str << "    " << DebugString(node) << "\n";
  }
  debug_str << "\n";
  // generator
  debug_str << "    feed_list = [" << cinn::utils::Join(feed_list, ", ")
            << "]\n";
  debug_str << "    fetch_list = ["
            << cinn::utils::Join(get_fetch_list(group, out_names), ", ")
            << "]\n";

  return debug_str.str();
}

std::string Graph::GenerateGroupPythonCode(
    const std::vector<Node*>& group,
    const std::unordered_set<std::string>& fetch_var_ids) {
  std::stringstream ss;
  ss << "#!/usr/bin/env python3\n";
  ss << "# Please set \"export "
        "PYTHONPATH=${CINN_ROOT}/build/python:${PYTHONPATH}\" first\n";
  ss << "\n";
  ss << "import unittest\n";
  ss << "import numpy as np\n";
  ss << "from cinn.frontend import NetBuilder\n";
  ss << "from cinn.common import DefaultNVGPUTarget\n";
  ss << "from tests.ops.op_test import OpTest\n";
  ss << "\n";
  ss << "class TestGroup(unittest.TestCase):\n";
  ss << "  def test_group(self):\n";
  ss << "    builder = NetBuilder(\"group_test\")\n";
  ss << "\n";
  ss << DebugGroupedGraph(group, fetch_var_ids);
  ss << "\n";
  ss << "    prog = builder.build()\n";
  ss << "\n";
  ss << "    feed_data = [OpTest.random(shape=var.shape(), dtype=var.type()) "
        "for var in feed_list]\n";
  ss << "    result = prog.build_and_get_output(DefaultNVGPUTarget(), "
        "feed_list, feed_data, fetch_list)\n";
  ss << "\n";
  ss << "    result = [res.numpy(DefaultNVGPUTarget()) for res in result]\n";
  ss << "    for i in range(len(result)):\n";
  ss << "      info_str = fetch_list[i].name()\n";
  ss << "      info_str += \", shape=\" + str(result[i].shape)\n";
  ss << "      info_str += \", dtype=\" + str(result[i].dtype) + \":\\n\"\n";
  ss << "      info_str += str(result[i])\n";
  ss << "      print(info_str)\n";

  ss << "\n";
  ss << "\n";
  ss << "if __name__ == \"__main__\":\n";
  ss << "  unittest.main()\n";
  ss << "\n";
  return ss.str();
}

std::string Graph::DebugGroupedGraph(
    const std::vector<std::vector<Node*>>& groups,
    const std::unordered_set<std::string>& fetch_var_ids) {
  std::unordered_set<std::string> fetch_list;
  if (!fetch_var_ids.empty()) {
    fetch_list = fetch_var_ids;
  } else {
    for (auto* var : this->outputs) {
      if (!var) {
        continue;
      }
      fetch_list.insert(var->id());
    }
  }

  std::stringstream debug_str;
  int group_id = 0;
  for (auto& group : groups) {
    debug_str << "Group " << group_id++ << " {\n";
    debug_str << DebugGroupedGraph(group, fetch_list);
    debug_str << "}\n";
  }
  debug_str << "\n";

  debug_str << "graph_fetch_list=["
            << cinn::utils::Join(std::vector<std::string>(fetch_list.begin(),
                                                          fetch_list.end()),
                                 ", ")
            << "]\n";

  return debug_str.str();
}

void Graph::VisualizeGroupedGraph(
    const std::unordered_set<std::string>& fetch_var_ids) {
  VisualizeGroupedGraph(FusionGroupsToGroups(), fetch_var_ids);
}

void Graph::VisualizeGroupedGraph(
    const std::vector<std::vector<Node*>>& origin_groups,
    const std::unordered_set<std::string>& fetch_var_ids) {
  if (cinn::runtime::CheckStringFlagFalse(
          FLAGS_cinn_fusion_groups_graphviz_dir)) {
    return;
  }

  // Dump debug info for each group
  VLOG(4) << "Dump graph debug info to: "
          << FLAGS_cinn_fusion_groups_graphviz_dir;
  const auto& groups = RemoveAccCheckGroups(origin_groups);
  const auto& group_dots = VisualizeGroups(groups, fetch_var_ids);
  for (int idx = 0; idx < groups.size(); ++idx) {
    // Create fusion_group_x folder
    int device_id = 0;
#ifdef CINN_WITH_CUDA
    cudaGetDevice(&device_id);
#endif
    auto group_path =
        utils::StringFormat("%s/device_%d/fusion_group_%d",
                            FLAGS_cinn_fusion_groups_graphviz_dir.c_str(),
                            device_id,
                            idx);
    if (!MakeDirectory(group_path,
                       S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
      LOG(WARNING) << "Failed to make directory: \"" << group_path
                   << "\", skip dump info for this group.";
      continue;
    }
    // Create test_group_x.py
    auto python_test_file =
        utils::StringFormat("%s/test_group_%d.py", group_path.c_str(), idx);
    WriteToFile(python_test_file,
                GenerateGroupPythonCode(groups[idx], fetch_var_ids));
    // Create x_group_name.dot
    auto graph_group_file =
        utils::StringFormat("%s/graph_group_%d.dot", group_path.c_str(), idx);
    WriteToFile(graph_group_file, group_dots[idx]);
  }

  // Summary
  Summary(groups, FLAGS_cinn_fusion_groups_graphviz_dir);
  // Grouped graph
  auto grouped_graph_file = utils::StringFormat(
      "%s/grouped_graph.dot", FLAGS_cinn_fusion_groups_graphviz_dir.c_str());
  WriteToFile(grouped_graph_file, VisualizeGraph(groups, fetch_var_ids));
}

std::string Graph::VisualizeGraph(
    const std::unordered_set<std::string>& fetch_var_ids) {
  return VisualizeGraph(FusionGroupsToGroups(), fetch_var_ids);
}

std::string Graph::VisualizeGraph(
    const std::vector<std::vector<Node*>>& groups,
    const std::unordered_set<std::string>& fetch_var_ids) {
  auto& shape_dict =
      HasAttr("infershape") ? GetAttrs<ShapeDict>("infershape") : ShapeDict{};
  auto& dtype_dict =
      HasAttr("inferdtype") ? GetAttrs<DTypeDict>("inferdtype") : DTypeDict{};

  std::unordered_map<std::string, int> recompute_nodes;
  FindRecomputeNodes(groups, &recompute_nodes);

  utils::DotLang dot;
  utils::ResetDotCounters();

  // Record the NodeData's actually ids.
  std::unordered_set<std::string> nodedatas_set;

  int group_id = 0;
  for (auto& group : groups) {
    std::string dot_cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(dot_cluster_id, GetGroupAttrs(group.size()));

    std::unordered_map<std::string, std::string> outnode2dot_id;
    for (auto* node : group) {
      AddGroupNode(node,
                   dot_cluster_id,
                   fetch_var_ids,
                   shape_dict,
                   dtype_dict,
                   &recompute_nodes,
                   &outnode2dot_id,
                   &nodedatas_set,
                   &dot);
    }
    group_id++;
  }
  return dot();
}

std::vector<std::string> Graph::VisualizeGroups(
    const std::unordered_set<std::string>& fetch_var_ids) {
  return VisualizeGroups(FusionGroupsToGroups(), fetch_var_ids);
}

std::vector<std::string> Graph::VisualizeGroups(
    const std::vector<std::vector<Node*>>& groups,
    const std::unordered_set<std::string>& fetch_var_ids) {
  auto& shape_dict =
      HasAttr("infershape") ? GetAttrs<ShapeDict>("infershape") : ShapeDict{};
  auto& dtype_dict =
      HasAttr("inferdtype") ? GetAttrs<DTypeDict>("inferdtype") : DTypeDict{};

  std::unordered_map<std::string, int> recompute_nodes;
  FindRecomputeNodes(groups, &recompute_nodes);

  utils::ResetDotCounters();

  std::vector<std::string> dot_vec;
  int group_id = 0;
  for (auto& group : groups) {
    utils::DotLang dot;
    std::unordered_set<Node*> nodes_set;
    std::string dot_cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(dot_cluster_id, GetGroupAttrs(group.size()));

    std::unordered_map<std::string, std::string> outnode2dot_id;
    for (auto* node : group) {
      AddGroupNode(node,
                   dot_cluster_id,
                   fetch_var_ids,
                   shape_dict,
                   dtype_dict,
                   &recompute_nodes,
                   &outnode2dot_id,
                   nullptr,
                   &dot);
      nodes_set.insert(node);
    }

    for (auto& node : group) {
      for (auto& inlink : node->inlinks()) {
        auto* innode = inlink->source()->safe_as<NodeData>();
        if (innode) {
          std::string dot_innode_id = outnode2dot_id[innode->id()];
          for (auto& innode_inlink : innode->inlinks()) {
            auto* in_innode = innode_inlink->source()->safe_as<Node>();
            if (in_innode && !nodes_set.count(in_innode)) {
              nodes_set.insert(in_innode);
              dot.AddNode(in_innode->id(), GetOutlinkOpAttrs());
              dot.AddEdge(in_innode->id(), dot_innode_id, {});
            }
          }
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          std::string dot_outnode_id = outnode2dot_id[outnode->id()];
          for (auto& outnode_outlink : outnode->outlinks()) {
            auto* out_outnode = outnode_outlink->sink()->safe_as<Node>();
            if (out_outnode && !nodes_set.count(out_outnode)) {
              if (IsAccCheckOp(out_outnode)) {
                continue;
              }
              nodes_set.insert(out_outnode);
              dot.AddNode(out_outnode->id(), GetOutlinkOpAttrs());
              dot.AddEdge(dot_outnode_id, out_outnode->id(), {});
            }
          }
        }
      }
    }
    dot_vec.emplace_back(dot());

    group_id++;
  }
  return dot_vec;
}

std::unordered_set<NodeData*> Graph::Group::GetInputNodeDatas() {
  std::unordered_set<NodeData*> group_inputs;

  // count all node's input data
  for (auto node : this->CollectNodes()) {
    for (auto& in_edge : node->inlinks_in_order()) {
      auto input_data = in_edge->source()->safe_as<NodeData>();
      if (!input_data) {
        continue;
      }

      if (!input_data->source_node.get()) {
        // if the input data hasn't input op, it's the group's input
        group_inputs.insert(input_data);
        continue;
      }

      if (std::find(this->input_names.begin(),
                    this->input_names.end(),
                    input_data->id()) != this->input_names.end()) {
        // if the input data in group's input_names
        group_inputs.insert(input_data);
        continue;
      }
    }
  }

  return group_inputs;
}

std::unordered_set<NodeData*> Graph::Group::GetOutputNodeDatas() {
  std::unordered_set<NodeData*> group_outputs;

  for (auto node : this->output_nodes) {
    for (auto& link : node->outlinks_in_order()) {
      auto node_data = link->sink()->safe_as<NodeData>();
      if (!node_data) {
        continue;
      }

      group_outputs.insert(node_data);
    }
  }

  return group_outputs;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
