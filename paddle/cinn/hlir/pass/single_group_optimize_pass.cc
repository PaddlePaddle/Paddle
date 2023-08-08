// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/visualize_helper.h"

namespace cinn::hlir::pass {

using framework::Graph;
using Group = framework::Graph::Group;

using common::GraphEdge;
using common::GraphNode;

using framework::Node;
using framework::NodeData;

using ShapeDict = absl::flat_hash_map<std::string, framework::shape_t>;
using DtypeDict = absl::flat_hash_map<std::string, common::Type>;

namespace utils {
template <typename T>
bool IsValueZero(cinn::utils::Attribute value) {
  if (!absl::holds_alternative<T>(value)) {
    return false;
  }
  return absl::get<T>(value) == static_cast<T>(0);
}
}  // namespace utils

class SingleGroupOptimizePass {
 public:
  explicit SingleGroupOptimizePass(Graph* graph);

  std::vector<std::shared_ptr<Group>> Apply();

 private:
  bool TryReplaceNodeToCustomCall(Node* node) const;

  bool CanReplaceToMemset(Node* node) const;
  bool CanReplaceToMemcpy(Node* node) const;

  void InitNodeToGroups();

 private:
  Graph* graph_;

  const ShapeDict& shape_dict_;
  const DtypeDict& dtype_dict_;
  std::unordered_map<Node*, int> node_to_groups_;
};

SingleGroupOptimizePass::SingleGroupOptimizePass(Graph* graph)
    : graph_(graph),
      shape_dict_(graph->GetMutableAttrs<ShapeDict>("infershape")),
      dtype_dict_(graph->GetMutableAttrs<DtypeDict>("inferdtype")) {
  // NOTE(jeff41404): to count how many times each node are used by group.
  // if a node used by more than one group, then will not be optimized.
  InitNodeToGroups();
}

std::vector<std::shared_ptr<Group>> SingleGroupOptimizePass::Apply() {
  std::vector<std::shared_ptr<Group>> optimized_groups;
  for (const auto& group : graph_->fusion_groups) {
    const auto& nodes = group->CollectNodes();
    if (nodes.empty()) {
      // empty group, skip
      continue;
    }
    if (nodes.size() > 1) {
      // The Group has multiple nodes, cannot be optimized, skip
      optimized_groups.emplace_back(group);
      continue;
    }
    CHECK(node_to_groups_.count(nodes.front()))
        << "Can't find node " << nodes.front()->id() << " in node_to_groups_!";
    // NOTE(jeff41404): if a node used by more than one group, then will not be
    // optimized to avoid unexpected changes to other group which has multiple
    // nodes.
    if (node_to_groups_[nodes.front()] > 1) {
      optimized_groups.emplace_back(group);
      continue;
    }
    // replace some const node like fill_constant/const_scalar to Memset,
    // replace some copy node like identity to Memcpy
    bool has_replaced = TryReplaceNodeToCustomCall(nodes.front());
    if (has_replaced) {
      // change the group pattern to kNonFusible
      group->op_pattern_kind = framework::kNonFusible;
    }

    optimized_groups.emplace_back(group);
  }

  return optimized_groups;
}

bool SingleGroupOptimizePass::TryReplaceNodeToCustomCall(Node* node) const {
  if (node->is_variable()) {
    // skip variable
    return false;
  }

  bool can_replace_to_memset = CanReplaceToMemset(node);

  bool can_replace_to_memcpy = false;
  if (!can_replace_to_memset) {
    can_replace_to_memcpy = CanReplaceToMemcpy(node);
  }

  bool can_replace = can_replace_to_memset || can_replace_to_memcpy;

  if (can_replace) {
    // replace single node group to custom call function
    const auto& op_name = node->op()->name;
    VLOG(4) << "Replaced node " << framework::DebugString(node)
            << " by \"custom_call\"";
    node->attrs.attr_store["original_op"] = op_name;
    node->attrs.op = framework::Operator::Get("custom_call");
  }

  if (can_replace_to_memset) {
    node->attrs.attr_store["custom_call"] =
        std::string("cinn_call_cuda_memset");
  }
  if (can_replace_to_memcpy) {
    node->attrs.attr_store["custom_call"] =
        std::string("cinn_call_cuda_memcpy");
  }

  return can_replace;
}

bool SingleGroupOptimizePass::CanReplaceToMemset(Node* node) const {
  const auto& op_name = node->op()->name;
  const auto& attr_store = node->attrs.attr_store;

  if (op_name == "fill_constant" || op_name == "const_scalar") {
    CHECK(attr_store.count("dtype"))
        << "Missing attribute \"dtype\" in op \"fill_constant\"";
    CHECK(absl::holds_alternative<std::string>(attr_store.at("dtype")));

    // if the value is 0, the op can always replaced by memset
    const auto& value_attr = attr_store.at("value");
    bool is_value_zero = utils::IsValueZero<int>(value_attr) ||
                         utils::IsValueZero<float>(value_attr) ||
                         utils::IsValueZero<bool>(value_attr) ||
                         utils::IsValueZero<int64_t>(value_attr) ||
                         utils::IsValueZero<double>(value_attr);
    return is_value_zero;
    // can support memset non-0 ?
  }

  return false;
}

bool SingleGroupOptimizePass::CanReplaceToMemcpy(Node* node) const {
  // the op do not compute and move data
  static std::unordered_set<std::string> can_replace_to_memcpy_op = {
      "identity", "reshape", "bitcast_convert", "squeeze", "expand_dims"};

  return can_replace_to_memcpy_op.count(node->op()->name);
}

void SingleGroupOptimizePassImpl(Graph* graph) {
  if (graph->target_ != common::DefaultNVGPUTarget()) {
    return;
  }
  graph->fusion_groups = SingleGroupOptimizePass(graph).Apply();
}

void SingleGroupOptimizePass::InitNodeToGroups() {
  for (const auto& group : graph_->fusion_groups) {
    const auto& nodes = group->CollectNodes();
    for (const auto& node : nodes) {
      if (!node_to_groups_.count(node)) {
        node_to_groups_[node] = 1;
      } else {
        node_to_groups_[node] += 1;
      }
    }
  }
}
}  // namespace cinn::hlir::pass

CINN_REGISTER_HELPER(SingleGroupOptimizePass) {
  CINN_REGISTER_PASS(SingleGroupOptimizePass)
      .describe("Optimize singel group to improve performance.")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::SingleGroupOptimizePassImpl);

  return true;
}
