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

#define CONDITION_FUNC(func)                       \
  inline bool func(const FusionHelperBase* helper, \
                   const Node* producer,           \
                   const std::shared_ptr<Graph::Group>& consumer)

CONDITION_FUNC(always_fuse) { return true; }

CONDITION_FUNC(no_fuse) { return false; }

CONDITION_FUNC(is_same_shape) {
  auto master_node = consumer->master_nodes.begin();
  return helper->GetNodeDataShape(producer) ==
         helper->GetNodeDataShape(*master_node);
}

CONDITION_FUNC(is_same_size) {
  auto master_node = consumer->master_nodes.begin();
  auto producer_shape = helper->GetNodeDataShape(producer);
  auto consumer_shape = helper->GetNodeDataShape(*master_node);
  if (producer_shape == consumer_shape) {
    return true;
  }
  auto psize = std::accumulate(
      producer_shape.begin(), producer_shape.end(), 1, std::multiplies<int>());
  auto csize = std::accumulate(
      consumer_shape.begin(), consumer_shape.end(), 1, std::multiplies<int>());
  return psize == csize;
}

CONDITION_FUNC(without_last_dimension_in_reduce) {
  auto in_shape =
      helper->shape_dict_.at(producer->inlinks_in_order()[0]->source()->id());
  auto reduce_axes =
      absl::get<std::vector<int>>(producer->attrs.attr_store.at("dim"));
  return helper->WithoutLastDimInReduce(in_shape, reduce_axes);
}

CONDITION_FUNC(reduce_fuse_reduce) {
  Node* reducer = NULL;
  for (auto* master : consumer->master_nodes) {
    if (helper->GetOpKind(master) == framework::kReduction) {
      reducer = master;
      break;
    }
  }
  // check reduce has same input shape and output shape
  auto producer_input_shape =
      helper->shape_dict_.at(producer->inlinks_in_order()[0]->source()->id());
  auto producer_output_shape =
      helper->shape_dict_.at(producer->outlinks_in_order()[0]->sink()->id());

  auto reducer_input_shape =
      helper->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
  auto reducer_output_shape =
      helper->shape_dict_.at(reducer->outlinks_in_order()[0]->sink()->id());

  auto producer_reduce_dim =
      absl::get<std::vector<int>>(producer->attrs.attr_store.at("dim"));
  auto reducer_reduce_dim =
      absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));

  for (auto& dim : producer_reduce_dim) {
    // if dim = -1, set as shape.size() - 1
    if (dim < 0) {
      dim += producer_input_shape.size();
    }
  }

  for (auto& dim : reducer_reduce_dim) {
    // if dim = -1,  set as shape.size() - 1
    if (dim < 0) {
      dim += reducer_input_shape.size();
    }
  }

  if (producer_output_shape == reducer_output_shape &&
      producer_reduce_dim == reducer_reduce_dim) {
    bool input_shape_same = producer_input_shape == reducer_input_shape;
    bool without_last_dim =
        helper->WithoutLastDimInReduce(producer_input_shape,
                                       producer_reduce_dim) &&
        helper->WithoutLastDimInReduce(reducer_input_shape, reducer_reduce_dim);
    // check shape is same
    if (input_shape_same || without_last_dim) {
      auto shared_size = helper->GetSharedSize(producer);
      for (auto* master : consumer->master_nodes) {
        if (helper->GetOpKind(master) == framework::kReduction) {
          shared_size += helper->GetSharedSize(master);
        }
      }

      constexpr int MAX_AVAILABLE_SHREAD = 32 * 1024;
      if (shared_size > MAX_AVAILABLE_SHREAD) {
        return false;
      }
      return true;
    }
  }

  return false;
}

CONDITION_FUNC(is_horizontal_relation) {
  auto check_depency = [&](const Node* node) {
    std::queue<const Node*> candidates;
    std::unordered_set<const Node*> visited_set;
    candidates.push(node);

    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      // visit all producer node
      for (auto tmp_node : helper->GetProducerNode(candidate)) {
        // check depency.
        if (producer == tmp_node) {
          return true;
        }
        // check node is in region.
        if (!consumer->nodes_set.count(tmp_node)) {
          continue;
        }
        // recored visited node.
        if (!visited_set.count(tmp_node)) {
          visited_set.insert(tmp_node);
          candidates.push(tmp_node);
        }
      }
    }

    return false;
  };

  for (auto node : consumer->nodes_set) {
    if (helper->GetOpKind(node) != consumer->op_pattern_kind) {
      continue;
    }
    if (check_depency(node)) {
      return false;
    }
  }

  return true;
};

CONDITION_FUNC(horizontal_or_vertical_reduce_relation) {
  // check is same shape with horizontal relation.
  if (is_same_size(helper, producer, consumer)) {
    return true;
  }

  // reducer node in fusion op.
  Node* reducer = NULL;
  for (auto* master : consumer->master_nodes) {
    if (helper->GetOpKind(master) == framework::kReduction) {
      reducer = master;
      break;
    }
  }

  // check producer has same shape with reducer node.
  auto reduce_shape =
      helper->shape_dict_.at(helper->GetProducerNodeData(reducer)[0]->id());
  auto reduce_axes =
      absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  for (auto& axis : reduce_axes) {
    // if axis = -1, set as shape.size() - 1
    if (axis < 0) {
      axis += reduce_shape.size();
    }
  }

  auto node_shape = helper->GetNodeDataShape(producer);
  auto node_size = std::accumulate(
      node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());
  auto reduce_size = std::accumulate(
      reduce_shape.begin(), reduce_shape.end(), 1, std::multiplies<int>());

  // is not same size with reduce size.
  if (node_size != reduce_size) {
    return false;
  }
  // check without last axis in reduce.
  if (helper->WithoutLastDimInReduce(reduce_shape, reduce_axes)) {
    return false;
  }

  int succesive_reduce_dimension = reduce_shape.at(reduce_axes.back());
  for (int idx = reduce_axes.size() - 2; idx >= 0; --idx) {
    if (reduce_axes[idx] == reduce_axes[idx + 1] - 1) {
      succesive_reduce_dimension *= reduce_shape[reduce_axes[idx]];
      continue;
    }
    break;
  }

  return helper->target_ == common::DefaultNVGPUTarget()
             ? (succesive_reduce_dimension <= helper->target_.max_num_threads()
                    ? true
                    : false)
             : true;
}

CONDITION_FUNC(horizontal_or_can_inline) {
  // horizontal relation.
  if (is_horizontal_relation(helper, producer, consumer)) {
    if (is_same_size(helper, producer, consumer)) {
      return true;
    } else {
      // if do broadcast, check can compute inline.
      return helper->output_nodes_set_.count(producer) == 0;
    }
  }
  // vertical relation: 1.can compute inline
  if (helper->GetNodeData(producer)->outlinks().size() == 1 &&
      helper->output_nodes_set_.count(producer) == 0) {
    return true;
  }

  // link to same node.
  auto& out_links = helper->GetNodeData(producer)->outlinks();
  for (auto link : out_links) {
    if ((*out_links.begin())->sink() != link->sink()) {
      return false;
    }
  }

  return helper->output_nodes_set_.count(producer) == 0;
}

CONDITION_FUNC(horizontal_with_same_size) {
  return is_horizontal_relation(helper, producer, consumer) &&
         is_same_size(helper, producer, consumer);
}

CONDITION_FUNC(reduce_fuse_broadcast) {
  if (is_horizontal_relation(helper, producer, consumer)) {
    if (is_same_size(helper, producer, consumer)) {
      return true;
    }
    return false;
  }

  if (helper->target_ != common::DefaultNVGPUTarget()) {
    return true;
  }

  auto rinput_shape = helper->GetNodeInputShape(producer);
  auto reduce_axes =
      absl::get<std::vector<int>>(producer->attrs.attr_store.at("dim"));
  auto keep_dim = absl::get<bool>(producer->attrs.attr_store.at("keep_dim"));
  for (auto& axis : reduce_axes) {
    if (axis < 0) {
      axis += rinput_shape.size();
    }
  }

  int reduce_size = rinput_shape.back();
  for (auto idx = reduce_axes.size() - 1; idx >= 1; --idx) {
    if (reduce_axes[idx] != reduce_axes[idx - 1] + 1) {
      return false;
    }
    reduce_size *= rinput_shape[idx - 1];
  }

  if (reduce_size > helper->target_.max_num_threads()) {
    return false;
  }

  auto routput_shape = helper->GetNodeDataShape(producer);
  auto find_reducer = [&](const Node* node,
                          const Node* reducer,
                          const std::unordered_set<Node*>& nodes_set) {
    std::queue<const Node*> candidates;
    candidates.push(node);

    while (!candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto producer : helper->GetProducerNode(candidate)) {
        if (producer == reducer) {
          return true;
        }

        if (nodes_set.count(producer)) {
          candidates.push(producer);
        }
      }
    }

    return false;
  };

  for (auto node : consumer->nodes_set) {
    if (helper->GetOpKind(node) != framework::kBroadcast) {
      continue;
    }

    if (!find_reducer(node, producer, consumer->nodes_set)) {
      continue;
    }

    auto broadcast_shape =
        absl::get<std::vector<int>>(node->attrs.attr_store.at("out_shape"));
    auto broadcast_axes = absl::get<std::vector<int>>(
        node->attrs.attr_store.at("broadcast_axes"));
    for (auto& axis : broadcast_axes) {
      if (axis < 0) {
        axis += broadcast_shape.size();
      }
    }

    if (rinput_shape != broadcast_shape) {
      return false;
    }
    // if keep dim = true.
    if (keep_dim) {
      continue;
    } else {
      // if routput_shape = [1]
      if (routput_shape.size() == 1 && routput_shape[0] == 1) {
        continue;
      }
      // check [reduce_axes, axes] = {0, 1, 2, 3, 4, 5, 6, ...}
      for (int idx = 0; idx < rinput_shape.size(); ++idx) {
        // note: !x ^ y == (!x) ^ y == !(x ^ y)
        if ((std::find(broadcast_axes.begin(), broadcast_axes.end(), idx) !=
             broadcast_axes.end()) ^
            std::find(reduce_axes.begin(), reduce_axes.end(), idx) ==
                reduce_axes.end()) {
          return false;
        }
      }
      continue;
    }
    return false;
  }
  return true;
}

#undef CONDITION_FUNC

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
