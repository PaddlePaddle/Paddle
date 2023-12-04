// Copyright (c) 20223 CINN Authors. All Rights Reserved.
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

#include <limits.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"

namespace cinn {
namespace dialect {
namespace ir {

const std::set<std::string> ConstantOps = {
    "const_scalar", "fill_constant", "arange"};

// limit the group args number to less equal 512, as args stack size is 4K.
inline bool limit_args(const std::shared_ptr<ir::Group>& first,
                       const std::shared_ptr<ir::Group>& second) {
  std::unordered_set<const ::pir::Operation*> args;
  for (auto& group : {first, second}) {
    for (auto iter : group->input_ops) {
      args.insert(iter.first);
    }
    for (auto op : group->output_ops) {
      args.insert(op);
    }
  }

  if (args.size() > 512) {
    return false;
  } else {
    return true;
  }
}

inline bool always_fuse(const std::shared_ptr<ir::Group>& first,
                        const std::shared_ptr<ir::Group>& second) {
  return true;
}

inline bool is_same_shape(const std::shared_ptr<ir::Group>& first,
                          const std::shared_ptr<ir::Group>& second) {
  if (!limit_args(first, second)) {
    return false;
  }

  auto output_var_0 = GetValueShape((*first->master_ops.begin())->result(0));
  auto output_var_1 = GetValueShape((*second->master_ops.begin())->result(0));
  return output_var_0 == output_var_1;
}

inline bool is_same_size(const std::shared_ptr<ir::Group>& first,
                         const std::shared_ptr<ir::Group>& second) {
  if (!limit_args(first, second)) {
    return false;
  }

  auto output_var_0 = GetValueShape((*first->master_ops.begin())->result(0));
  auto output_var_1 = GetValueShape((*second->master_ops.begin())->result(0));
  if (output_var_0 == output_var_1) {
    return true;
  }

  auto size_0 = ::common::product(output_var_0);
  auto size_1 = ::common::product(output_var_1);
  return size_0 == size_1;
}

inline bool is_const_group(const std::shared_ptr<ir::Group>& group) {
  return group->CollectOps().size() == 1 &&
         ConstantOps.count(group->CollectOps()[0]->name());
}

inline bool elementwise_fuse_broadcast(
    const std::shared_ptr<ir::Group>& first,
    const std::shared_ptr<ir::Group>& second) {
  // if producer just include const op.
  if (is_const_group(first)) {
    return true;
  }
  // if same shape with horizontal relation
  if (is_same_size(first, second)) {
    return true;
  }
  // if first's output is not all in second's input
  for (auto output : first->output_ops) {
    return true;
    if (!second->input_ops.count(output)) {
      return false;
    }

    // TODO(phlrain): support output set here
    // if (helper->output_nodes_set_.count(output)) {
    //   return false;
    // }

    return true;
  }
  // 1.compute io-size
  // 2.compute computation-size
  // 3.compute recompute-times
  // 4.compute cost
  // TODO(sunli) : cost-model.
  return true;
}

inline bool honrizontal_elementwise_fuse_reduce(
    const std::shared_ptr<ir::Group>& first,
    const std::shared_ptr<ir::Group>& second) {
  std::shared_ptr<ir::Group> ele_group, reduce_group;
  if (first->op_pattern_kind == OpPatternKind::kReduction) {
    ele_group = second;
    reduce_group = first;
  } else {
    ele_group = first;
    reduce_group = second;
  }
  // if same shape with horizontal relation
  if (is_same_size(first, second)) {
    return true;
  }

  auto ele_node_shape =
      GetValueShape((*ele_group->master_ops.begin())->result(0));
  int32_t size_ele = ::common::product(ele_node_shape);
  // TODO(phlrain): seems extrame danger herem, why compare multi Master Node?
  for (auto* master : reduce_group->master_ops) {
    auto master_node_shape = GetValueShape(master->result(0));
    int32_t size_master = ::common::product(master_node_shape);
    if (size_ele == size_master) {
      return true;
    }
  }

  return false;
}

inline bool elementwise_fuse_reduce(const std::shared_ptr<ir::Group>& first,
                                    const std::shared_ptr<ir::Group>& second) {
  // if (helper->target_ == cinn::common::DefaultHostTarget()) {
  //   return true;
  // }
  // if same shape with horizontal relation
  if (is_same_size(first, second)) {
    return true;
  }

  // if reduce nodes not in consumers of first group
  std::queue<::pir::Operation*> candidates;
  std::unordered_set<::pir::Operation*> first_node_set = first->OpSet();
  std::unordered_set<::pir::Operation*> second_node_set = second->OpSet();
  for (const auto& pair : second->input_ops) {
    if (first_node_set.find(pair.first) != first_node_set.end()) {
      candidates.push(pair.first);
    }
  }
  std::unordered_set<::pir::Operation*> visited;
  std::unordered_set<::pir::Operation*> masters_in_consumers;

  while (!candidates.empty()) {
    ::pir::Operation* candidate = candidates.front();
    candidates.pop();

    // TODO(phlrain) : why only deal with first output
    auto first_output = candidate->result(0);
    for (auto it = first_output.use_begin(); it != first_output.use_end();
         ++it) {
      auto consumer = (*it).owner();
      if (visited.count(consumer)) {
        continue;
      }
      if (second_node_set.find(consumer) != second_node_set.end()) {
        visited.insert(consumer);
        candidates.push(consumer);
      }
      if (second->master_ops.count(consumer)) {
        masters_in_consumers.insert(consumer);
      }
    }
  }
  if (!masters_in_consumers.empty()) {
    bool flag = true;
    auto first_node_shape =
        GetValueShape((*first->master_ops.begin())->result(0));
    int32_t size_first = ::common::product(first_node_shape);

    for (::pir::Operation* master : masters_in_consumers) {
      auto second_node_shape = GetValueShape(master->result(0));
      int32_t size_second = ::common::product(second_node_shape);
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
  ::pir::Operation* reducer = nullptr;
  for (auto& node : second->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*node) ==
        OpPatternKind::kReduction) {
      reducer = node;
      break;
    }
  }
  // CHECK(reducer) << "Can't find reduce op in group " << second->group_id;

  // If the elementwise's output should be fetched, the output var cannot be
  // computed inline into reduce's loop, in other words, the elementwise's
  // cannot fused into reduce's loop Like: group1 = {cast_0},
  // group2={broadcast_0 -> elementwise_0 -> cast_1 -> reduce_max_0}

  // TODO(phlrain) : pass output node set
  // if (helper->output_nodes_set_.count(*first->master_nodes.begin())) {
  //   return false;
  // }

  auto input_shape = GetValueShape(reducer->operand_source(0));
  auto reduce_axes = GetVectorAttr(reducer, "dim");

  // int max_num_threads = helper->target_.max_num_threads();
  int max_num_threads = 1000;
  // if without last dimension in reduce.
  int lane = 1;
  if (WithoutLastDimInReduce(input_shape, reduce_axes)) {
    for (int idx = reduce_axes.back() + 1; idx < input_shape.size(); ++idx) {
      lane *= input_shape[idx];
    }
    if (lane > max_num_threads / 2) {
      return true;
    }
  }

  int index = reduce_axes.size() - 1;
  for (; index >= 0; --index) {
    if (static_cast<size_t>(index + 1) < reduce_axes.size() &&
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

inline bool broadcast_fuse_reduce(const std::shared_ptr<ir::Group>& first,
                                  const std::shared_ptr<ir::Group>& second) {
  // if same shape with horizontal relation
  if (is_same_size(first, second)) {
    return true;
  }
  ::pir::Operation* reducer = nullptr;
  for (auto& node : second->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*node) ==
        OpPatternKind::kReduction) {
      reducer = node;
      break;
    }
  }
  // CHECK(reducer) << "Can't find reduce op in group " << second->group_id;

  auto input_shape = GetValueShape(reducer->operand_source(0));
  auto input_size = ::common::product(input_shape);

  auto output_shape = GetValueShape((*first->master_ops.begin())->result(0));
  auto output_size = ::common::product(output_shape);

  if (input_size == output_size) {
    return elementwise_fuse_reduce(first, second);
  }
  return false;
}

inline bool reduce_fuse_elementwise(const std::shared_ptr<ir::Group>& first,
                                    const std::shared_ptr<ir::Group>& second) {
  if (!is_same_size(first, second)) {
    return false;
  }
  // if with last axis in reduce, fuse will waste computation resource.
  // so use a simple model evaluate the cost.
  // TODO(sunli) : cost-model.
  return true;
}

inline bool horizontal_relation(const std::shared_ptr<ir::Group>& first,
                                const std::shared_ptr<ir::Group>& second,
                                const OpPatternKind op_pattern_kind) {
  // merge injective
  auto merge_nodes_set = [](const std::shared_ptr<ir::Group>& group) {
    std::unordered_set<::pir::Operation*> nodes_set = group->ops_set;
    for (auto& sub_group : group->fused_sub_groups) {
      nodes_set.insert(sub_group->ops_set.begin(), sub_group->ops_set.end());
    }
    return nodes_set;
  };
  auto first_set = merge_nodes_set(first);
  auto second_set = merge_nodes_set(second);

  auto select_node_set = [](const std::unordered_set<::pir::Operation*>& nodes,
                            OpPatternKind kind) {
    std::unordered_set<::pir::Operation*> selected;
    for (auto node : nodes) {
      if (hlir::framework::pir::CompatibleInfo::OpKind(*node) == kind) {
        selected.insert(node);
      }
    }
    return selected;
  };
  auto selected_nodes = select_node_set(second_set, op_pattern_kind);

  auto check_depency = [&](::pir::Operation* node) {
    std::queue<::pir::Operation*> candidates;
    std::unordered_set<::pir::Operation*> visited_set;
    candidates.push(node);

    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      // visit all producer node
      // Get all the input Op
      for (size_t i = 0; i < candidate->num_operands(); ++i) {
        auto producer =
            candidate->operand_source(i).dyn_cast<pir::OpResult>().owner();
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

inline bool horizontal_with_injective(
    const std::shared_ptr<ir::Group>& first,
    const std::shared_ptr<ir::Group>& second) {
  if (is_const_group(first)) {
    return true;
  }

  if (!is_same_size(first, second)) {
    return false;
  }
  return horizontal_relation(first, second, OpPatternKind::kInjective);
}

inline bool injective_horizontal_with_reduce(
    const std::shared_ptr<ir::Group>& first,
    const std::shared_ptr<ir::Group>& second) {
  // check injective with injective.
  if (!horizontal_relation(first, second, OpPatternKind::kInjective)) {
    return false;
  }
  return elementwise_fuse_reduce(first, second);
}

inline bool reduce_fuse_broadcast(const std::shared_ptr<ir::Group>& first,
                                  const std::shared_ptr<ir::Group>& second) {
  // if same shape with horizontal relation
  if (is_same_size(first, second)) {
    return true;
  }

  // Traversing all reducers in all producers requires two types of conditions
  // to be met. The first type is the condition that the reducer itself needs to
  // meet, and the second type is the condition that the relationship between
  // each reducer and its consumers with type of Broadcast needs to meet. It is
  // required that each consumer of type Broadcast meet the same shape after
  // broadcast as before reduce.
  for (auto& node_in_master : first->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*node_in_master) !=
        OpPatternKind::kReduction) {
      continue;
    }
    ::pir::Operation* reducer = node_in_master;
    // First type conditions
    // Get some reduce information
    auto reducer_input_shape =
        ::common::vectorize(GetValueShape(reducer->operand_source(0)));
    auto reducer_output_shape =
        ::common::vectorize(GetValueShape(reducer->result(0)));
    std::vector<int64_t> reduce_axes = GetVectorAttr(reducer, "dim");

    auto keep_dim =
        reducer->attribute("keep_dim").dyn_cast<pir::BoolAttribute>().data();
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
    // if (helper->target_ == cinn::common::DefaultNVGPUTarget() &&
    //     reduce_size > helper->target_.max_num_threads()) {
    //   return false;
    // }

    // Second type conditions
    // Find directly or indirectly consumers with type of Broadcast in the
    // second group
    auto find_broadcasters_in_descendants = [&](::pir::Operation* producer)
        -> std::unordered_set<::pir::Operation*> {
      std::queue<::pir::Operation*> candidates;
      std::unordered_set<::pir::Operation*> visited_set;
      std::unordered_set<::pir::Operation*> broadcasters;
      candidates.push(producer);

      while (!candidates.empty()) {
        auto candidate = candidates.front();
        candidates.pop();

        if (candidate->num_results() == 0) {
          continue;
        }
        // TODO(phlrain) : why only deal with first output
        auto first_output = candidate->result(0);
        for (auto it = first_output.use_begin(); it != first_output.use_end();
             ++it) {
          auto consumer = (*it).owner();

          if (consumer->isa<pir::YieldOp>()) {
            continue;
          }
          if (!visited_set.count(consumer)) {
            visited_set.insert(consumer);
            candidates.push(consumer);
          }
          if (hlir::framework::pir::CompatibleInfo::OpKind(*consumer) ==
                  OpPatternKind::kBroadcast &&
              second->OpSet().find(consumer) != second->OpSet().end()) {
            broadcasters.insert(consumer);
          }
        }
      }

      return broadcasters;
    };

    // Check if each broadcast node meets the conditions
    std::unordered_set<::pir::Operation*> broadcasters_in_consumers =
        find_broadcasters_in_descendants(reducer);
    for (auto broadcaster : broadcasters_in_consumers) {
      // auto  = absl::get<std::vector<int>>(
      //     broadcaster->attrs.attr_store.at("out_shape"));

      // auto broadcast_axes = absl::get<std::vector<int>>(
      //     broadcaster->attrs.attr_store.at("broadcast_axes"));
      // TODO(phlrain) : suport here
      std::vector<int64_t> broadcaster_output_shape =
          GetVectorAttr(broadcaster, "out_shape");
      std::vector<int64_t> broadcast_axes =
          GetVectorAttr(broadcaster, "broadcast_axes");
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
        for (size_t idx = 0; idx < reducer_input_shape.size(); ++idx) {
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

inline bool reduce_fuse_reduce(const std::shared_ptr<ir::Group>& first,
                               const std::shared_ptr<ir::Group>& second) {
  if (!limit_args(first, second)) {
    return false;
  }
  ::pir::Operation* reducer_0 = nullptr;
  for (auto& reducer : first->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*reducer) ==
        OpPatternKind::kReduction) {
      reducer_0 = reducer;
      break;
    }
  }
  // CHECK(reducer_0) << "Can't find reduce op in group " << first->group_id;

  ::pir::Operation* reducer_1 = nullptr;
  for (auto& reducer : second->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*reducer) ==
        OpPatternKind::kReduction) {
      reducer_1 = reducer;
      break;
    }
  }
  CHECK(reducer_1) << "Can't find reduce op in group " << second->group_id;
  // check reduce has same input shape and output shape
  auto reducer_0_input_shape = GetValueShape(reducer_0->operand_source(0));
  auto reducer_0_output_shape = GetValueShape(reducer_0->result(0));

  auto reducer_1_input_shape = GetValueShape(reducer_1->operand_source(0));
  auto reducer_1_output_shape = GetValueShape(reducer_1->result(0));

  auto reducer_0_reduce_dim = GetVectorAttr(reducer_0, "dim");
  auto reducer_1_reduce_dim = GetVectorAttr(reducer_1, "dim");

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
      for (auto* master : fusion_group->master_ops) {
        if (hlir::framework::pir::CompatibleInfo::OpKind(*master) ==
            OpPatternKind::kReduction) {
          shared_size += GetSharedSize(master);
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

  if (WithoutLastDimInReduce(reducer_0_input_shape, reducer_0_reduce_dim) &&
      WithoutLastDimInReduce(reducer_1_input_shape, reducer_1_reduce_dim) &&
      reducer_0_output_shape == reducer_1_output_shape &&
      reducer_0_reduce_dim == reducer_1_reduce_dim) {
    auto shared_size = 0;
    for (auto& fusion_group : {first, second}) {
      for (auto* master : fusion_group->master_ops) {
        if (hlir::framework::pir::CompatibleInfo::OpKind(*master) ==
            OpPatternKind::kReduction) {
          shared_size += GetSharedSize(master);
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

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
