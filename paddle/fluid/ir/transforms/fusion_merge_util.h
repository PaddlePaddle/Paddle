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

#include <limits.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

namespace ir {

enum OpPatternKind {
  // The relation between input tensor index and output tensor index is
  // one-to-one correspondence.
  // for example :code:`out[i, j] = input[i, j] + 1`.
  // Note that the axis need to be in order.
  kElementWise = 0,
  // The relation between input tensor index and output tensor index is
  // one-to-many correspondence.
  // for example :code:`out[i, j, k] = input[i, j]`.
  // Note that the axis need to be in order.
  kBroadcast = 1,
  // Injective operator, we can always injectively map a output axis to a input
  // axis.
  // for example :code:`out[i, j] = input[j, i]`.
  kInjective = 2,
  // The relation between input tensor index and output tensor index is
  // many-to-one correspondence.
  // for example :code:`out[i, j] = sum(input[i, j, k]) along k`.
  kReduction = 3,
  // Complex operation, can still fuse one-to-one operations into its output.
  kOutFusible = 4,
  // Operation that cannot fuse anything.
  kNonFusible = 8
};

OpPatternKind GetOpKind(const std::string& op_name);

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::ir::Operation* op,
                             const std::string& name) {
  auto& attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  auto& val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<ir::ArrayAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<ir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<ir::Int64Attribute>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<ir::Int64Attribute>().data());
    }
  }
  return vec_res;
}

struct Group {
  Group() = default;

  // distance to last group.
  int depth{0};
  int max_depth{0};
  int min_depth{INT_MAX};
  // group id, consisted of node's id.
  std::string group_id{""};
  // global unique id.
  std::string unique_id{"uniq"};
  // node in this group
  std::vector<Operation*> nodes;
  std::unordered_set<Operation*> nodes_set;
  // input nodes of the group.
  std::unordered_map<Operation*, int> input_nodes;
  // output nodes of the group.
  std::unordered_set<Operation*> output_nodes;
  // op pattern kind.
  OpPatternKind op_pattern_kind{kElementWise};
  // internal node, the output is used by multi-node.
  // internal node can't use compute inline, should use buffer.
  std::unordered_set<Operation*> internal_nodes;
  // master node for schedule
  std::unordered_set<Operation*> master_nodes;

  // fused sub-groups, used for fusion merge pass
  std::vector<std::shared_ptr<Group>> fused_sub_groups;
  // if as sub-group, used for belong groups.
  std::unordered_set<std::shared_ptr<Group>> belong_groups;

  // for op lowering.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  struct SharedGroupHasher {
    size_t operator()(const std::shared_ptr<Group>& group) const noexcept {
      return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(group.get()));
    }
  };
  struct SharedGroupComparator {
    bool operator()(const std::shared_ptr<Group>& first,
                    const std::shared_ptr<Group>& second) const noexcept {
      return first.get() == second.get();
    }
  };

  std::vector<Operation*> CollectNodes() {
    if (fused_sub_groups.size()) {
      std::vector<Operation*> tmp_nodes;
      for (auto& group : fused_sub_groups) {
        tmp_nodes.insert(
            tmp_nodes.end(), group->nodes.begin(), group->nodes.end());
      }
      return tmp_nodes;
    } else {
      return nodes;
    }
  }

  void WalkNodes(const std::function<void(const Operation*)>& VisitNode) const {
    if (fused_sub_groups.size()) {
      for (auto& group : fused_sub_groups) {
        for (const auto& node : group->nodes) {
          VisitNode(node);
        }
      }
    } else {
      for (const auto& node : nodes) {
        VisitNode(node);
      }
    }
  }

  std::unordered_set<Operation*> NodeSet() {
    std::unordered_set<Operation*> node_set;
    for (auto node : CollectNodes()) {
      node_set.insert(node);
    }
    return node_set;
  }

  std::unordered_set<Value> GetInputNodeDatas() { return {}; }
  std::unordered_set<Value> GetOutputNodeDatas() { return {}; }

  std::string GetFuncName() { return "fn_" + group_id + unique_id; }

 public:
  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  producer_groups() const {
    return producer_groups_;
  }

  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  consumer_groups() const {
    return consumer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_producer_groups() {
    return &producer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_consumer_groups() {
    return &consumer_groups_;
  }

  OpPatternKind kind() const { return op_pattern_kind; }

 private:
  // input groups
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      producer_groups_;
  // output grous
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      consumer_groups_;
};

phi::DDim GetFirstInputShape(const Operation* op);

phi::DDim GetValueShape(const Value value);

bool WithoutLastDimInReduce(const std::vector<int64_t>& inshape,
                            const std::vector<int64_t>& axes);

int GetSharedSize(const Operation* node);

inline bool always_fuse(const Operation* producer,
                        const std::shared_ptr<Group>& consumer) {
  return true;
}

inline bool no_fuse(const Operation* producer,
                    const std::shared_ptr<Group>& consumer) {
  return false;
}

inline bool is_same_shape(const Operation* producer,
                          const std::shared_ptr<Group>& consumer) {
  auto master_node = consumer->master_nodes.begin();
  return GetValueShape(producer->result(0)) ==
         GetValueShape((*master_node)->result(0));
}

inline bool is_same_size(const Operation* producer,
                         const std::shared_ptr<Group>& consumer) {
  auto master_node = consumer->master_nodes.begin();
  auto producer_shape = GetValueShape(producer->result(0));
  auto consumer_shape = GetValueShape((*master_node)->result(0));
  if (producer_shape == consumer_shape) {
    return true;
  }
  auto psize = phi::product(producer_shape);
  auto csize = phi::product(consumer_shape);
  return psize == csize;
}

inline bool without_last_dimension_in_reduce(
    const Operation* producer, const std::shared_ptr<Group>& consumer) {
  auto in_shape = phi::vectorize<int64_t>(GetFirstInputShape(producer));
  auto reduce_axes = GetVectorAttr(producer, "axis");
  return WithoutLastDimInReduce(in_shape, reduce_axes);
}

inline bool reduce_fuse_reduce(const Operation* producer,
                               const std::shared_ptr<Group>& consumer) {
  Operation* reducer = NULL;
  for (auto* master : consumer->master_nodes) {
    if (GetOpKind(master->name()) == kReduction) {
      reducer = master;
      break;
    }
  }
  // check reduce has same input shape and output shape
  auto producer_input_shape =
      phi::vectorize<int64_t>(GetValueShape(producer->operand(0)));
  auto producer_output_shape =
      phi::vectorize<int64_t>(GetValueShape(producer->result(0)));

  auto reducer_input_shape =
      phi::vectorize<int64_t>(GetValueShape(reducer->operand(0)));
  auto reducer_output_shape =
      phi::vectorize<int64_t>(GetValueShape(reducer->result(0)));

  auto producer_reduce_dim = GetVectorAttr(producer, "axis");
  auto reducer_reduce_dim = GetVectorAttr(reducer, "axis");

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
        WithoutLastDimInReduce(producer_input_shape, producer_reduce_dim) &&
        WithoutLastDimInReduce(reducer_input_shape, reducer_reduce_dim);
    // check shape is same
    if (input_shape_same || without_last_dim) {
      auto shared_size = GetSharedSize(producer);
      for (auto* master : consumer->master_nodes) {
        if (GetOpKind(master->name()) == kReduction) {
          shared_size += GetSharedSize(master);
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

inline bool is_horizontal_relation(const Operation* producer,
                                   const std::shared_ptr<Group>& consumer) {
  auto check_depency = [&](const Operation* node) {
    std::queue<const Operation*> candidates;
    std::unordered_set<const Operation*> visited_set;
    candidates.push(node);

    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      // visit all producer node
      for (size_t i = 0; i < candidate->num_operands(); ++i) {
        auto tmp_node = candidate->operand(i).GetDefiningOp();
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
    if (GetOpKind(node->name()) != consumer->op_pattern_kind) {
      continue;
    }
    if (check_depency(node)) {
      return false;
    }
  }

  return true;
}

inline bool horizontal_or_vertical_reduce_relation(
    const Operation* producer, const std::shared_ptr<Group>& consumer) {
  // check is same shape with horizontal relation.
  if (is_same_size(producer, consumer)) {
    return true;
  }

  // reducer node in fusion op.
  Operation* reducer = NULL;
  for (auto* master : consumer->master_nodes) {
    if (GetOpKind(master->name()) == kReduction) {
      reducer = master;
      break;
    }
  }

  // check producer has same shape with reducer node.
  auto reduce_shape = phi::vectorize(GetFirstInputShape(reducer));
  auto reduce_axes = GetVectorAttr(reducer, "axis");

  for (auto& axis : reduce_axes) {
    // if axis = -1, set as shape.size() - 1
    if (axis < 0) {
      axis += reduce_shape.size();
    }
  }

  auto node_shape = phi::vectorize<int64_t>(GetFirstInputShape(producer));
  auto node_size = std::accumulate(
      node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());
  auto reduce_size = std::accumulate(
      reduce_shape.begin(), reduce_shape.end(), 1, std::multiplies<int>());

  // is not same size with reduce size.
  if (node_size != reduce_size) {
    return false;
  }
  // check without last axis in reduce.
  if (WithoutLastDimInReduce(reduce_shape, reduce_axes)) {
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

  // helper->target_ == common::DefaultNVGPUTarget()
  // succesive_reduce_dimension <= helper->target_.max_num_threads()
  // TODO(phlrain): support is_gpu_target and max_thread
  bool is_gpu_target = true;
  int max_thread = 32 * 1024;
  return is_gpu_target
             ? (succesive_reduce_dimension <= max_thread ? true : false)
             : true;
}

inline bool horizontal_or_can_inline(const Operation* producer,
                                     const std::shared_ptr<Group>& consumer) {
  // horizontal relation.
  if (is_horizontal_relation(producer, consumer)) {
    if (is_same_size(producer, consumer)) {
      return true;
    } else {
      // if do broadcast, check can compute inline.
      // return helper->output_nodes_set_.count(producer) == 0;
      // TODO(phlrain): support output node set check
      return false;
    }
  }
  // vertical relation: 1.can compute inline
  // if (helper->GetNodeData(producer)->outlinks().size() == 1 &&
  //     helper->output_nodes_set_.count(producer) == 0) {
  //   return true;
  // }

  // link to same node.
  // auto& out_links = helper->GetNodeData(producer)->outlinks();
  // for (auto link : out_links) {
  //   if ((*out_links.begin())->sink() != link->sink()) {
  //     return false;
  //   }
  // }

  // return helper->output_nodes_set_.count(producer) == 0;

  return false;
}

inline bool horizontal_with_same_size(const Operation* producer,
                                      const std::shared_ptr<Group>& consumer) {
  return is_horizontal_relation(producer, consumer) &&
         is_same_size(producer, consumer);
}

inline bool reduce_fuse_broadcast(const Operation* producer,
                                  const std::shared_ptr<Group>& consumer) {
  if (is_horizontal_relation(producer, consumer)) {
    if (is_same_size(producer, consumer)) {
      return true;
    }
    return false;
  }

  // if (helper->target_ != common::DefaultNVGPUTarget()) {
  //   return true;
  // }

  auto rinput_shape = phi::vectorize<int64_t>(GetFirstInputShape(producer));
  auto reduce_axes = GetVectorAttr(producer, "axis");
  auto keep_dim = producer->attributes()
                      .at("keep_dim")
                      .dyn_cast<ir::BoolAttribute>()
                      .data();
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

  // if (reduce_size > helper->target_.max_num_threads()) {
  //   return false;
  // }

  auto routput_shape =
      phi::vectorize<int64_t>(GetValueShape(producer->result(0)));
  auto find_reducer = [&](const Operation* node,
                          const Operation* reducer,
                          const std::unordered_set<Operation*>& nodes_set) {
    std::queue<const Operation*> candidates;
    candidates.push(node);

    while (!candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (size_t i = 0; i < candidate->num_operands(); ++i) {
        auto producer = candidate->operand(i).GetDefiningOp();
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
    if (GetOpKind(node->name()) != kBroadcast) {
      continue;
    }

    if (!find_reducer(node, producer, consumer->nodes_set)) {
      continue;
    }

    auto broadcast_shape = GetVectorAttr(node, "out_shape");
    auto broadcast_axes = GetVectorAttr(node, "broadcast_axes");

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
      for (size_t idx = 0; idx < rinput_shape.size(); ++idx) {
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

}  // namespace ir
