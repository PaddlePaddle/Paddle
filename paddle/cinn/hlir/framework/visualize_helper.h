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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/utils/dot_lang.h"

#include "paddle/common/errors.h"

namespace cinn {
namespace hlir {
namespace framework {

class PassPrinter {
 public:
  static PassPrinter* GetInstance() {
    static PassPrinter printer;
    return &printer;
  }

  bool Begin(const std::unordered_set<std::string>& fetch_ids = {});
  bool PassBegin(const std::string& pass_name,
                 const frontend::Program& program);
  bool PassEnd(const std::string& pass_name, const frontend::Program& program);
  bool End();

 private:
  std::unordered_set<std::string> fetch_ids_;
  std::string save_path_;
  int64_t graph_id_{0};
  int64_t pass_id_{0};
};

inline void WriteToFile(const std::string& filepath,
                        const std::string& content) {
  VLOG(4) << "Write to " << filepath;
  std::ofstream of(filepath);
  PADDLE_ENFORCE(of.is_open(),
                 ::common::errors::Unavailable("Failed to open %s", filepath));
  of << content;
  of.close();
}

inline std::string GenClusterId(const std::vector<Node*>& group, int group_id) {
  return "group_" + std::to_string(group_id) +
         "(size=" + std::to_string(group.size()) + ")";
}

inline std::string GenNodeId(const Node* node,
                             bool is_recomputed,
                             int recompute_id) {
  if (is_recomputed) {
    return node->id() + "/" + std::to_string(recompute_id);
  } else {
    return node->id();
  }
}

inline std::string GenNodeDataId(const NodeData* data,
                                 bool is_recomputed,
                                 int recompute_id) {
  if (is_recomputed) {
    return data->id() + "/" + std::to_string(recompute_id);
  } else {
    return data->id();
  }
}

inline std::vector<utils::DotAttr> GetGroupOpAttrs(bool is_recomputed = false) {
  std::string color = is_recomputed ? "#836FFF" : "#8EABFF";
  return std::vector<utils::DotAttr>{utils::DotAttr("shape", "Mrecord"),
                                     utils::DotAttr("color", color),
                                     utils::DotAttr("style", "filled")};
}

inline std::vector<utils::DotAttr> GetOutlinkOpAttrs() {
  return std::vector<utils::DotAttr>{utils::DotAttr("shape", "Mrecord"),
                                     utils::DotAttr("color", "#ff7f00"),
                                     utils::DotAttr("style", "filled")};
}

inline std::vector<utils::DotAttr> GetGroupVarAttrs(bool is_fetched = false) {
  if (is_fetched) {
    return std::vector<utils::DotAttr>{utils::DotAttr("peripheries", "2"),
                                       utils::DotAttr("color", "#43CD80"),
                                       utils::DotAttr("style", "filled")};
  } else {
    return std::vector<utils::DotAttr>{utils::DotAttr("color", "#FFDC85"),
                                       utils::DotAttr("style", "filled")};
  }
}

inline std::vector<utils::DotAttr> GetGroupAttrs(size_t group_size) {
  std::string fillcolor;
  if (group_size == 1) {
    fillcolor = "#E8E8E8";
  } else if (group_size <= 3) {
    fillcolor = "#FFFFF0";
  } else if (group_size <= 10) {
    fillcolor = "#F0FFFF";
  } else {
    // group_size > 10
    fillcolor = "#EEE5DE";
  }
  std::vector<utils::DotAttr> attrs = {utils::DotAttr("color", "grey"),
                                       utils::DotAttr("style", "filled"),
                                       utils::DotAttr("fillcolor", fillcolor)};
  return attrs;
}

bool MakeDirectory(const std::string& dirname, mode_t mode);

std::string GenNodeDataLabel(
    const NodeData* node,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
    const std::string dot_nodedata_id);

void Summary(const std::vector<std::vector<Node*>>& groups,
             const std::string& viz_path);

std::string DebugString(const Node* node);

void FindRecomputeNodes(const std::vector<std::vector<Node*>>& groups,
                        std::unordered_map<std::string, int>* recompute_nodes);

void AddGroupNode(
    const Node* node,
    const std::string& dot_cluster_id,
    const std::unordered_set<std::string>& fetch_var_ids,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
    std::unordered_map<std::string, int>* recompute_nodes,
    std::unordered_map<std::string, std::string>* outnode2dot_id,
    std::unordered_set<std::string>* nodedatas_set,
    utils::DotLang* dot);

// used for CheckFusionAccuracyPass
std::string GenerateAccCheckNodeId(const std::string& node_id);

bool IsAccCheckOp(const Node* op);
bool IsAccCheckVar(const NodeData* var);
bool IsAccCheckGroup(const std::vector<Node*>& group);

std::vector<std::vector<Node*>> RemoveAccCheckGroups(
    const std::vector<std::vector<Node*>>& groups);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
