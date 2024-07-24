// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/visualize_helper.h"

#include <errno.h>
#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/dot_lang.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_string(cinn_pass_visualize_dir);
PD_DECLARE_string(cinn_check_fusion_accuracy_pass);
namespace cinn {
namespace hlir {
namespace framework {

bool PassPrinter::Begin(const std::unordered_set<std::string>& fetch_ids) {
  if (FLAGS_cinn_pass_visualize_dir.empty()) {
    VLOG(3) << "No set \"FLAGS_cinn_pass_visualize_dir\", the pass visualize "
               "information will print directly.";
    save_path_.clear();
    return false;
  }
  pass_id_ = 0;
  fetch_ids_ = fetch_ids;

  save_path_ = utils::StringFormat(
      "%s/fusion_groups_%d/", FLAGS_cinn_pass_visualize_dir.c_str(), graph_id_);
  if (!MakeDirectory(save_path_,
                     S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    LOG_IF(WARNING, graph_id_ == 0)
        << "Failed to make directory: \"" << save_path_
        << "\", the CINN subgraph's pass visualize information will not print.";
    return false;
  }
  LOG_IF(INFO, graph_id_ == 0) << "The CINN subgraph's pass visualize "
                                  "information will writing into path: \""
                               << FLAGS_cinn_pass_visualize_dir << "\"";
  return true;
}

bool PassPrinter::PassBegin(const std::string& pass_name,
                            const frontend::Program& program) {
  if (save_path_.empty()) {
    return false;
  }
  const auto& program_info = utils::GetStreamCnt(program);
  VLOG(3) << "Before " << pass_name << " Pass:\n" << program_info;
  const std::string& file_path = utils::StringFormat("%s/pass_%d_%s_before.txt",
                                                     save_path_.c_str(),
                                                     pass_id_,
                                                     pass_name.c_str());
  WriteToFile(file_path, program_info);
  return true;
}

bool PassPrinter::PassEnd(const std::string& pass_name,
                          const frontend::Program& program) {
  if (save_path_.empty()) {
    return false;
  }
  const auto& program_info = utils::GetStreamCnt(program);
  VLOG(3) << "After " << pass_name << " Pass:\n" << program_info;
  const std::string& file_path = utils::StringFormat("%s/pass_%d_%s_after.txt",
                                                     save_path_.c_str(),
                                                     pass_id_,
                                                     pass_name.c_str());
  WriteToFile(file_path, program_info);

  ++pass_id_;
  return true;
}

bool PassPrinter::End() {
  ++graph_id_;

  pass_id_ = 0;
  fetch_ids_.clear();
  save_path_.clear();
  return true;
}

bool MakeDirectory(const std::string& dirname, mode_t mode) {
  struct stat st;
  std::string path;
  for (int i = 0; i < dirname.size(); ++i) {
    path.push_back(dirname[i]);
    if (!(dirname[i] == '/' || i + 1 == dirname.size())) {
      continue;
    }
    if (stat(path.c_str(), &st) == 0) {
      if (S_ISDIR(st.st_mode)) {
        continue;
      } else {
        LOG(WARNING) << path << " is not a directory, please check your path.";
        return false;
      }
    } else {
      if (mkdir(path.c_str(), mode) == 0) {
        continue;
      } else {
        LOG(WARNING) << "Make directory fail: " << path;
        return false;
      }
    }
  }
  return true;
}

std::string GenNodeDataLabel(
    const NodeData* node,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
    const std::string dot_nodedata_id) {
  std::stringstream ss;
  ss << dot_nodedata_id;
  if (shape_dict.count(node->id())) {
    shape_t node_shape = shape_dict.at(node->id());
    ss << "\\n[";
    for (size_t i = 0; i < node_shape.size(); ++i) {
      if (i > 0) {
        ss << "x";
      }
      ss << node_shape[i];
    }
    ss << "]";
  }
  if (dtype_dict.count(node->id())) {
    ss << "\\n";
    ss << cinn::common::Type2Str(dtype_dict.at(node->id()));
  }

  return ss.str();
}

void Summary(const std::vector<std::vector<Node*>>& groups,
             const std::string& viz_path) {
  std::map<std::string, size_t> group_summary;
  std::map<std::string, size_t> single_group_detail;
  std::map<std::string, size_t> fusion_group_detail;

  for (auto& group : groups) {
    size_t group_size = group.size();
    group_summary[std::to_string(group_size)]++;
    if (group_size == 1) {
      // Like "fill_constant_1", remove the "_1" at the end of the string.
      std::string node_id = group[0]->id();
      int index = node_id.size() - 1;
      while (index != -1) {
        if (node_id[index] >= '0' && node_id[index] <= '9') {
          index--;
        } else {
          break;
        }
      }
      if (node_id[index] == '_') {
        index--;
      }
      if (index >= 0) {
        node_id = node_id.substr(0, index + 1);
        single_group_detail[node_id]++;
      }
    } else {
      std::string key = "others";
      for (auto* node : group) {
        if (node->id().find("reduce") != std::string::npos) {
          key = "reduce";
          break;
        }
      }
      fusion_group_detail[key]++;
    }
  }

  std::stringstream ss;

  auto print_table = [&](const std::map<std::string, size_t>& res) {
    int total = 0;
    for (auto& item : res) {
      ss << std::setw(20) << item.first << item.second << "\n";
      total += item.second;
    }
    ss << "-------------------------------------------\n";
    ss << std::setw(20) << "total" << total << "\n";
    ss << "-------------------------------------------\n";
  };

  ss << "-------------------------------------------\n";
  ss << "             Summary of Groups\n";
  ss << "-------------------------------------------\n";
  ss << std::setiosflags(std::ios::left);
  ss << std::setfill(' ');
  ss << std::setw(20) << "Size"
     << "Numbers\n";
  print_table(group_summary);

  if (single_group_detail.size()) {
    ss << "\n\n-------------------------------------------\n";
    ss << "          Detail of Single Groups\n";
    ss << "-------------------------------------------\n";
    ss << std::setw(20) << "Type"
       << "Numbers\n";
    print_table(single_group_detail);
  }

  ss << "\n\n-------------------------------------------\n";
  ss << "          Detail of Fusion Groups\n";
  ss << "-------------------------------------------\n";
  ss << std::setw(20) << "Type"
     << "Numbers\n";
  print_table(fusion_group_detail);

  std::string filepath = viz_path + "/summary.txt";
  WriteToFile(filepath, ss.str());
}

std::string DebugString(const Node* node) {
  std::vector<std::string> out_names;
  for (auto& outlink : node->outlinks_in_order()) {
    auto* outnode = outlink->sink()->safe_as<NodeData>();
    if (outnode) {
      out_names.emplace_back(outnode->id());
    }
  }

  std::vector<std::string> in_names;
  for (auto& inlink : node->inlinks_in_order()) {
    auto* innode = inlink->source()->safe_as<NodeData>();
    if (innode) {
      in_names.emplace_back(innode->id());
    }
  }

  std::stringstream ss;
  ss << cinn::utils::Join(out_names, ", ") << " = builder." << node->op()->name
     << "(" << cinn::utils::Join(in_names, ", ");

  bool first = true;
  std::map<std::string, std::string> attr_str_map;
  for (const auto& attr_pair : node->attrs.attr_store) {
    attr_str_map[attr_pair.first] = utils::Attribute2String(attr_pair.second);
  }

  for (const auto& attr_pair : attr_str_map) {
    if (!first) {
      ss << ", ";
    } else {
      if (!in_names.empty()) {
        // insert a split letter before if inputs not empty
        ss << ", ";
      }
      first = false;
    }
    ss << attr_pair.first << "=" << attr_pair.second;
  }
  ss << ")";
  return ss.str();
}

void FindRecomputeNodes(const std::vector<std::vector<Node*>>& groups,
                        std::unordered_map<std::string, int>* recompute_nodes) {
  std::unordered_map<std::string, int> op_count;
  for (auto& group : groups) {
    for (auto* node : group) {
      op_count[node->id()]++;
    }
  }
  for (auto& iter : op_count) {
    if (iter.second > 1) {
      (*recompute_nodes)[iter.first] = 0;
    }
  }
}

void AddGroupNode(
    const Node* node,
    const std::string& dot_cluster_id,
    const std::unordered_set<std::string>& fetch_var_ids,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
    std::unordered_map<std::string, int>* recompute_nodes,
    std::unordered_map<std::string, std::string>* outnode2dot_id,
    std::unordered_set<std::string>* nodedatas_set,
    utils::DotLang* dot) {
  bool is_recomputed = recompute_nodes->count(node->id());
  int recompute_id = is_recomputed ? (*recompute_nodes)[node->id()]++ : -1;

  std::string dot_node_id = GenNodeId(node, is_recomputed, recompute_id);
  dot->AddNode(dot_node_id, GetGroupOpAttrs(is_recomputed), "", dot_cluster_id);

  for (auto& inlink : node->inlinks()) {
    auto* innode = inlink->source()->safe_as<NodeData>();
    if (innode) {
      if (!outnode2dot_id->count(innode->id())) {
        (*outnode2dot_id)[innode->id()] = innode->id();
      }
      std::string dot_innode_id = outnode2dot_id->at(innode->id());
      if (!nodedatas_set || !nodedatas_set->count(dot_innode_id)) {
        std::string label =
            GenNodeDataLabel(innode, shape_dict, dtype_dict, dot_innode_id);
        dot->AddNode(dot_innode_id,
                     GetGroupVarAttrs(false),
                     label,
                     dot_cluster_id,
                     true);
        if (nodedatas_set) {
          nodedatas_set->insert(dot_innode_id);
        }
      }
      dot->AddEdge(dot_innode_id, dot_node_id, {});
    }
  }

  for (auto& outlink : node->outlinks()) {
    auto* outnode = outlink->sink()->safe_as<NodeData>();
    if (outnode) {
      std::string dot_outnode_id =
          GenNodeDataId(outnode, is_recomputed, recompute_id);
      (*outnode2dot_id)[outnode->id()] = dot_outnode_id;
      if (!nodedatas_set || !nodedatas_set->count(dot_outnode_id)) {
        bool is_fetched = fetch_var_ids.count(outnode->id());
        std::string label =
            GenNodeDataLabel(outnode, shape_dict, dtype_dict, dot_outnode_id);
        dot->AddNode(dot_outnode_id,
                     GetGroupVarAttrs(is_fetched),
                     label,
                     dot_cluster_id,
                     true);
        if (nodedatas_set) {
          nodedatas_set->insert(dot_outnode_id);
        }
      }
      dot->AddEdge(dot_node_id, dot_outnode_id, {});
    }
  }
}

bool IsAccCheckOp(const Node* op) {
  return op->attrs.node_name.find("_acc_check") != std::string::npos;
}
bool IsAccCheckVar(const NodeData* var) {
  return var->id().find("_acc_check") != std::string::npos;
}

std::string GenerateAccCheckNodeId(const std::string& node_id) {
  return node_id + cinn::common::UniqName("_acc_check");
}

bool IsAccCheckGroup(const std::vector<Node*>& group) {
  for (auto* node : group) {
    if (IsAccCheckOp(node)) {
      return true;
    }
  }
  return false;
}

std::vector<std::vector<Node*>> RemoveAccCheckGroups(
    const std::vector<std::vector<Node*>>& groups) {
  if (cinn::runtime::CheckStringFlagFalse(
          FLAGS_cinn_check_fusion_accuracy_pass)) {
    // no set acc check flag
    return groups;
  }

  std::vector<std::vector<Node*>> new_groups;
  for (const auto& group : groups) {
    if (!IsAccCheckGroup(group)) {
      new_groups.emplace_back(group);
    }
  }
  return new_groups;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
