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

#include <queue>

#include "paddle/cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace hlir {
namespace pass {

#define CONDITION_FUNC(func)                                   \
  inline bool func(const FusionHelperBase* helper,             \
                   const std::shared_ptr<Graph::Group>& first, \
                   const std::shared_ptr<Graph::Group>& second)

// limit the group args number to less equal 512, as args stack size is 4K.
CONDITION_FUNC(limit_args) {
  std::unordered_set<Node*> args;
  for (auto& group : {first, second}) {
    for (auto node : group->input_nodes) {
      args.insert(node.first);
    }
    for (auto node : group->output_nodes) {
      args.insert(node);
    }
  }

  if (args.size() > 512) {
    return false;
  } else {
    return true;
  }
}

CONDITION_FUNC(always_fuse) { return true; }

CONDITION_FUNC(is_same_shape) {
  if (!limit_args(helper, first, second)) {
    return false;
  }
  auto output_var_0 = helper->GetNodeDataShape(*first->master_nodes.begin());
  auto output_var_1 = helper->GetNodeDataShape(*second->master_nodes.begin());
  return output_var_0 == output_var_1;
}

CONDITION_FUNC(is_same_size) {
  if (!limit_args(helper, first, second)) {
    return false;
  }
  auto output_var_0 = helper->GetNodeDataShape(*first->master_nodes.begin());
  auto output_var_1 = helper->GetNodeDataShape(*second->master_nodes.begin());
  if (output_var_0 == output_var_1) {
    return true;
  }

  auto size_0 = std::accumulate(
      output_var_0.begin(), output_var_0.end(), 1, std::multiplies<int>());
  auto size_1 = std::accumulate(
      output_var_1.begin(), output_var_1.end(), 1, std::multiplies<int>());
  return size_0 == size_1;
}

inline bool is_const_group(const FusionHelperBase* helper,
                           const std::shared_ptr<Graph::Group>& group) {
  return group->CollectNodes().size() == 1 &&
         helper->IsConstOp(group->CollectNodes()[0]);
}

CONDITION_FUNC(elementwise_fuse_broadcast) {
  // if producer just include const op.
  if (is_const_group(helper, first)) {
    return true;
  }
  // if same shape with horizontal relation
  if (is_same_size(helper, first, second)) {
    return true;
  }
  // if first's output is not all in second's input
  for (auto output : first->output_nodes) {
    if (!second->input_nodes.count(output)) {
      return false;
    }
    if (helper->output_nodes_set_.count(output)) {
      return false;
    }
  }
  // 1.compute io-size
  // 2.compute computation-size
  // 3.compute recompute-times
  // 4.compute cost
  // TODO(sunli) : cost-model.
  return true;
}

CONDITION_FUNC(honrizontal_elementwise_fuse_reduce) {
  std::shared_ptr<Graph::Group> ele_group, reduce_group;
  if (first->op_pattern_kind == framework::kReduction) {
    ele_group = second;
    reduce_group = first;
  } else {
    ele_group = first;
    reduce_group = second;
  }
  // if same shape with horizontal relation
  if (is_same_size(helper, first, second)) {
    return true;
  }

  shape_t ele_node_shape =
      helper->GetNodeDataShape(*ele_group->master_nodes.begin());
  int32_t size_ele = std::accumulate(
      ele_node_shape.begin(), ele_node_shape.end(), 1, std::multiplies<int>());
  for (Node* master : reduce_group->master_nodes) {
    shape_t master_node_shape = helper->GetNodeDataShape(master);
    int32_t size_master = std::accumulate(master_node_shape.begin(),
                                          master_node_shape.end(),
                                          1,
                                          std::multiplies<int>());
    if (size_ele == size_master) {
      return true;
    }
  }

  return false;
}

CONDITION_FUNC(elementwise_fuse_reduce) {
  if (helper->target_ == common::DefaultHostTarget()) {
    return true;
  }
  // if same shape with horizontal relation
  if (is_same_size(helper, first, second)) {
    return true;
  }

  // if reduce nodes not in consumers of first group
  std::queue<Node*> candidates;
  std::unordered_set<Node*> first_node_set = first->NodeSet();
  std::unordered_set<Node*> second_node_set = second->NodeSet();
  for (const auto& pair : second->input_nodes) {
    if (first_node_set.find(pair.first) != first_node_set.end()) {
      candidates.push(pair.first);
    }
  }
  std::unordered_set<Node*> visited;
  std::unordered_set<Node*> masters_in_consumers;

  while (!candidates.empty()) {
    Node* candidate = candidates.front();
    candidates.pop();

    std::vector<Node*> consumers = helper->GetConsumerNode(candidate);
    for (auto consumer : consumers) {
      if (visited.count(consumer)) {
        continue;
      }
      if (second_node_set.find(consumer) != second_node_set.end()) {
        visited.insert(consumer);
        candidates.push(consumer);
      }
      if (second->master_nodes.count(consumer)) {
        masters_in_consumers.insert(consumer);
      }
    }
  }
  if (!masters_in_consumers.empty()) {
    bool flag = true;
    shape_t first_node_shape =
        helper->GetNodeDataShape(*first->master_nodes.begin());
    int32_t size_first = std::accumulate(first_node_shape.begin(),
                                         first_node_shape.end(),
                                         1,
                                         std::multiplies<int>());
    for (Node* master : masters_in_consumers) {
      shape_t second_node_shape = helper->GetNodeDataShape(master);
      int32_t size_second = std::accumulate(second_node_shape.begin(),
                                            second_node_shape.end(),
                                            1,
                                            std::multiplies<int>());
      if (size_first != size_second) {
        flag = false;
        break;
      }
    }
    if (flag) {
      return true;
    }
  }

  // if reduce using block_reduce, can't fuse producer.
  Node* reducer = nullptr;
  for (auto& node : second->master_nodes) {
    if (helper->GetOpKind(node) == framework::kReduction) {
      reducer = node;
      break;
    }
  }
  CHECK(reducer) << "Can't find reduce op in group " << second->group_id;

  // If the elementwise's output should be fetched, the output var cannot be
  // computed inline into reduce's loop, in other words, the elementwise's
  // cannot fused into reduce's loop Like: group1 = {cast_0},
  // group2={broadcast_0 -> elementwise_0 -> cast_1 -> reduce_max_0}
  if (helper->output_nodes_set_.count(*first->master_nodes.begin())) {
    return false;
  }

  auto input_shape =
      helper->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
  auto reduce_axes =
      absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));

  int max_num_threads = helper->target_.max_num_threads();
  // if without last dimension in reduce.
  int lane = 1;
  if (helper->WithoutLastDimInReduce(input_shape, reduce_axes)) {
    for (int idx = reduce_axes.back() + 1; idx < input_shape.size(); ++idx) {
      lane *= input_shape[idx];
    }
    if (lane > max_num_threads / 2) {
      return true;
    }
  }

  int index = reduce_axes.size() - 1;
  for (; index >= 0; --index) {
    if (index + 1 < reduce_axes.size() &&
        reduce_axes[index] + 1 != reduce_axes[index + 1]) {
      break;
    }
    lane *= input_shape[reduce_axes[index]];
    if (lane > max_num_threads / 2) {
      break;
    }
  }

  if (lane <= max_num_threads) {
    return true;
  } else {
    int prefix = input_shape[reduce_axes[index]];
    int tail = lane / prefix;
    for (int idx = max_num_threads / tail; idx > (max_num_threads / 2) / tail;
         --idx) {
      if (prefix % idx == 0) {
        return true;
      }
    }
  }
  return false;
}

CONDITION_FUNC(broadcast_fuse_reduce) {
  // if same shape with horizontal relation
  if (is_same_size(helper, first, second)) {
    return true;
  }
  Node* reducer = nullptr;
  for (auto& node : second->master_nodes) {
    if (helper->GetOpKind(node) == OpPatternKind::kReduction) {
      reducer = node;
      break;
    }
  }
  CHECK(reducer) << "Can't find reduce op in group " << second->group_id;

  auto input_shape =
      helper->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
  auto input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

  auto output_shape = helper->GetNodeDataShape(*first->master_nodes.begin());
  auto output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  if (input_size == output_size) {
    return elementwise_fuse_reduce(helper, first, second);
  }
  return false;
}

CONDITION_FUNC(reduce_fuse_elementwise) {
  if (!is_same_size(helper, first, second)) {
    return false;
  }
  // if with last axis in reduce, fuse will waste computation resource.
  // so use a simple model evaluate the cost.
  // TODO(sunli) : cost-model.
  return true;
}

inline bool horizontal_relation(
    const FusionHelperBase* helper,
    const std::shared_ptr<Graph::Group>& first,
    const std::shared_ptr<Graph::Group>& second,
    const framework::OpPatternKind op_pattern_kind) {
  // merge injective
  auto merge_nodes_set = [](const std::shared_ptr<Graph::Group>& group) {
    std::unordered_set<Node*> nodes_set = group->nodes_set;
    for (auto& sub_group : group->fused_sub_groups) {
      nodes_set.insert(sub_group->nodes_set.begin(),
                       sub_group->nodes_set.end());
    }
    return nodes_set;
  };
  auto first_set = merge_nodes_set(first);
  auto second_set = merge_nodes_set(second);

  auto select_node_set = [helper](const std::unordered_set<Node*>& nodes,
                                  framework::OpPatternKind kind) {
    std::unordered_set<Node*> selected;
    for (auto node : nodes) {
      if (helper->GetOpKind(node) == kind) {
        selected.insert(node);
      }
    }
    return selected;
  };
  auto selected_nodes = select_node_set(second_set, op_pattern_kind);

  auto check_depency = [&](const Node* node) {
    std::queue<const Node*> candidates;
    std::unordered_set<const Node*> visited_set;
    candidates.push(node);

    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      // visit all producer node
      for (auto producer : helper->GetProducerNode(candidate)) {
        // check dependency.
        if (first_set.count(producer)) {
          return true;
        }
        // check node is in region.
        if (!second_set.count(producer)) {
          continue;
        }
        // recorded visited node.
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }

    return false;
  };

  for (auto node : selected_nodes) {
    if (check_depency(node)) {
      return false;
    }
  }

  return true;
}

CONDITION_FUNC(horizontal_with_injective) {
  if (is_const_group(helper, first)) {
    return true;
  }

  if (!is_same_size(helper, first, second)) {
    return false;
  }
  return horizontal_relation(
      helper, first, second, framework::OpPatternKind::kInjective);
}

CONDITION_FUNC(injective_horizontal_with_reduce) {
  // check injective with injective.
  if (!horizontal_relation(
          helper, first, second, framework::OpPatternKind::kInjective)) {
    return false;
  }
  return elementwise_fuse_reduce(helper, first, second);
}

CONDITION_FUNC(reduce_fuse_broadcast) {
  // if same shape with horizontal relation
  if (is_same_size(helper, first, second)) {
    return true;
  }

  // Traversing all reducers in all producers requires two types of conditions
  // to be met. The first type is the condition that the reducer itself needs to
  // meet, and the second type is the condition that the relationship between
  // each reducer and its consumers with type of Broadcast needs to meet. It is
  // required that each consumer of type Broadcast meet the same shape after
  // broadcast as before reduce.
  for (auto& node_in_master : first->master_nodes) {
    if (helper->GetOpKind(node_in_master) != OpPatternKind::kReduction) {
      continue;
    }
    Node* reducer = node_in_master;
    // First type conditions
    // Get some reduce information
    auto reducer_input_shape = helper->GetNodeInputShape(reducer);
    auto reducer_output_shape = helper->GetNodeDataShape(reducer);
    auto reduce_axes =
        absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
    auto keep_dim = absl::get<bool>(reducer->attrs.attr_store.at("keep_dim"));
    for (auto& axis : reduce_axes) {
      if (axis == -1) {
        axis = reducer_input_shape.size() - 1;
      }
    }
    // Check if the reduce axes are continuous
    int reduce_size = reducer_input_shape.back();
    for (auto idx = reduce_axes.size() - 1; idx >= 1; --idx) {
      if (reduce_axes[idx] != reduce_axes[idx - 1] + 1) {
        return false;
      }
      reduce_size *= reducer_input_shape[idx - 1];
    }
    // Check if the reduce size exceeds the hardware limit
    if (helper->target_ == common::DefaultNVGPUTarget() &&
        reduce_size > helper->target_.max_num_threads()) {
      return false;
    }

    // Second type conditions
    // Find directly or indirectly consumers with type of Broadcast in the
    // second group
    auto find_broadcasters_in_descendants =
        [&](const Node* producer) -> std::unordered_set<const Node*> {
      std::queue<const Node*> candidates;
      std::unordered_set<const Node*> visited_set;
      std::unordered_set<const Node*> broadcasters;
      candidates.push(producer);

      while (!candidates.empty()) {
        auto candidate = candidates.front();
        candidates.pop();

        for (auto consumer : helper->GetConsumerNode(candidate)) {
          if (!visited_set.count(consumer)) {
            visited_set.insert(consumer);
            candidates.push(consumer);
          }
          if (helper->GetOpKind(consumer) == OpPatternKind::kBroadcast &&
              second->NodeSet().find(consumer) != second->NodeSet().end()) {
            broadcasters.insert(consumer);
          }
        }
      }

      return broadcasters;
    };

    // Check if each broadcast node meets the conditions
    std::unordered_set<const Node*> broadcasters_in_consumers =
        find_broadcasters_in_descendants(reducer);
    for (auto broadcaster : broadcasters_in_consumers) {
      auto broadcaster_output_shape = absl::get<std::vector<int>>(
          broadcaster->attrs.attr_store.at("out_shape"));
      auto broadcast_axes = absl::get<std::vector<int>>(
          broadcaster->attrs.attr_store.at("broadcast_axes"));
      for (auto& axis : broadcast_axes) {
        if (axis == -1) {
          axis = broadcaster_output_shape.size() - 1;
        }
      }

      if (reducer_input_shape != broadcaster_output_shape) {
        return false;
      }

      if (keep_dim) {
        continue;
      } else {
        // if reducer_output_shape = [1]
        if (reducer_output_shape.size() == 1 && reducer_output_shape[0] == 1) {
          continue;
        }
        // check union [reduce_axes, broadcast_axes] = reducer_input_shape
        for (int idx = 0; idx < reducer_input_shape.size(); ++idx) {
          if (!(std::find(broadcast_axes.begin(), broadcast_axes.end(), idx) ==
                broadcast_axes.end()) ^
              std::find(reduce_axes.begin(), reduce_axes.end(), idx) ==
                  reduce_axes.end()) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

CONDITION_FUNC(reduce_fuse_reduce) {
  if (!limit_args(helper, first, second)) {
    return false;
  }
  Node* reducer_0 = nullptr;
  for (auto& reducer : first->master_nodes) {
    if (helper->GetOpKind(reducer) == OpPatternKind::kReduction) {
      reducer_0 = reducer;
      break;
    }
  }
  CHECK(reducer_0) << "Can't find reduce op in group " << first->group_id;

  Node* reducer_1 = nullptr;
  for (auto& reducer : second->master_nodes) {
    if (helper->GetOpKind(reducer) == OpPatternKind::kReduction) {
      reducer_1 = reducer;
      break;
    }
  }
  CHECK(reducer_1) << "Can't find reduce op in group " << second->group_id;

  // check reduce has same input shape and output shape
  auto reducer_0_input_shape =
      helper->shape_dict_.at(reducer_0->inlinks_in_order()[0]->source()->id());
  auto reducer_0_output_shape =
      helper->shape_dict_.at(reducer_0->outlinks_in_order()[0]->sink()->id());

  auto reducer_1_input_shape =
      helper->shape_dict_.at(reducer_1->inlinks_in_order()[0]->source()->id());
  auto reducer_1_output_shape =
      helper->shape_dict_.at(reducer_1->outlinks_in_order()[0]->sink()->id());

  auto reducer_0_reduce_dim =
      absl::get<std::vector<int>>(reducer_0->attrs.attr_store.at("dim"));
  auto reducer_1_reduce_dim =
      absl::get<std::vector<int>>(reducer_1->attrs.attr_store.at("dim"));

  for (auto& dim : reducer_0_reduce_dim) {
    // if dim = -1, set as shape.size() - 1
    if (dim == -1) {
      dim = reducer_0_reduce_dim.size() - 1;
    }
  }

  for (auto& dim : reducer_1_reduce_dim) {
    // if dim = -1,  set as shape.size() - 1
    if (dim == -1) {
      dim = reducer_1_reduce_dim.size() - 1;
    }
  }

  // check shape is same
  if (reducer_0_input_shape == reducer_1_input_shape &&
      reducer_0_output_shape == reducer_1_output_shape &&
      reducer_0_reduce_dim == reducer_1_reduce_dim) {
    auto shared_size = 0;
    for (auto& fusion_group : {first, second}) {
      for (auto* master : fusion_group->master_nodes) {
        if (helper->GetOpKind(master) == framework::kReduction) {
          shared_size += helper->GetSharedSize(master);
        }
      }
    }

#define MAX_AVAILABLE_SHREAD 32 * 1024
    if (shared_size > MAX_AVAILABLE_SHREAD) {
      return false;
    }
#undef MAX_AVAILABLE_SHREAD
    return true;
  }

  if (helper->WithoutLastDimInReduce(reducer_0_input_shape,
                                     reducer_0_reduce_dim) &&
      helper->WithoutLastDimInReduce(reducer_1_input_shape,
                                     reducer_1_reduce_dim) &&
      reducer_0_output_shape == reducer_1_output_shape &&
      reducer_0_reduce_dim == reducer_1_reduce_dim) {
    auto shared_size = 0;
    for (auto& fusion_group : {first, second}) {
      for (auto* master : fusion_group->master_nodes) {
        if (helper->GetOpKind(master) == framework::kReduction) {
          shared_size += helper->GetSharedSize(master);
        }
      }
    }

#define MAX_AVAILABLE_SHREAD 32 * 1024
    if (shared_size > MAX_AVAILABLE_SHREAD) {
      return false;
    }
#undef MAX_AVAILABLE_SHREAD
    return true;
  }

  return false;
}

#undef CONDITION_FUNC

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
