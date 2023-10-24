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

#include "paddle/cinn/hlir/framework/op_lowering_util.h"

#include "paddle/cinn/hlir/pe/nn_util.h"
#include "paddle/cinn/utils/string.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#endif
#include <queue>

namespace cinn {
namespace hlir {
namespace framework {

namespace utils {
struct NodeCompare {
  bool operator()(Node* lhs, Node* rhs) const { return lhs->id() < rhs->id(); }
};
}  // namespace utils

std::vector<NodeData*> GetInputNodeData(const Node* node) {
  std::vector<NodeData*> producers;
  for (auto& link : node->inlinks_in_order()) {
    auto node_data = link->source()->safe_as<NodeData>();
    producers.push_back(node_data);
  }
  return producers;
}

ir::Tensor GetTensor(
    const NodeData* node_data,
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto dtype = type_dict.at(node_data->id());
  if (dtype.is_float(32)) {
    return lang::Placeholder<float>(node_data->id(),
                                    shape_dict.at(node_data->id()));
  } else if (dtype.is_float(64)) {
    return lang::Placeholder<double>(node_data->id(),
                                     shape_dict.at(node_data->id()));
  } else if (dtype.is_bfloat16()) {
    return lang::Placeholder<common::bfloat16>(node_data->id(),
                                               shape_dict.at(node_data->id()));
  } else if (dtype.is_float16()) {
    return lang::Placeholder<common::float16>(node_data->id(),
                                              shape_dict.at(node_data->id()));
  } else if (dtype.is_bool()) {
    return lang::Placeholder<bool>(node_data->id(),
                                   shape_dict.at(node_data->id()));
  } else if (dtype.is_int(8)) {
    return lang::Placeholder<int8_t>(node_data->id(),
                                     shape_dict.at(node_data->id()));
  } else if (dtype.is_int(16)) {
    return lang::Placeholder<int16_t>(node_data->id(),
                                      shape_dict.at(node_data->id()));
  } else if (dtype.is_int(32)) {
    return lang::Placeholder<int32_t>(node_data->id(),
                                      shape_dict.at(node_data->id()));
  } else if (dtype.is_int(64)) {
    return lang::Placeholder<int64_t>(node_data->id(),
                                      shape_dict.at(node_data->id()));
  } else if (dtype.is_uint(8)) {
    return lang::Placeholder<uint8_t>(node_data->id(),
                                      shape_dict.at(node_data->id()));
  } else if (dtype.is_uint(16)) {
    return lang::Placeholder<uint16_t>(node_data->id(),
                                       shape_dict.at(node_data->id()));
  } else if (dtype.is_uint(32)) {
    return lang::Placeholder<uint32_t>(node_data->id(),
                                       shape_dict.at(node_data->id()));
  } else if (dtype.is_uint(64)) {
    return lang::Placeholder<uint64_t>(node_data->id(),
                                       shape_dict.at(node_data->id()));
  } else {
    LOG(FATAL) << "Unsupport dtype: " << dtype;
  }
}

std::vector<ir::Tensor> CollectInputTensor(
    const Node* node,
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<std::string, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  // get all input nodes
  for (auto& node_data : GetInputNodeData(node)) {
    CHECK(node_data);
    auto tensor = GetTensor(node_data, type_dict, shape_dict);
    if (!tensor_map->count(node_data->id())) {
      (*tensor_map)[node_data->id()] = tensor;
      // record func input args
      func_args->push_back(tensor);
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

NodeData* GetNodeData(const Node* node) {
  auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
  CHECK(node_data);
  return node_data;
}

std::vector<NodeData*> GetAllNodeData(const Node* node) {
  std::vector<NodeData*> node_datas;
  for (auto& link : node->outlinks_in_order()) {
    auto node_data = link->sink()->safe_as<NodeData>();
    CHECK(node_data);
    node_datas.push_back(node_data);
  }

  return node_datas;
}

std::vector<Node*> GetConsumers(const Node* node) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks()) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    consumers.push_back(consumer);
  }
  return consumers;
}

std::vector<Node*> GetConsumersInSet(
    const Node* node, const std::unordered_set<Node*>& node_set) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks()) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    if (node_set.count(consumer)) {
      consumers.push_back(consumer);
    }
  }
  return consumers;
}

std::vector<Node*> GetProducers(const Node* node) {
  std::vector<Node*> producers;
  for (auto& link : node->inlinks_in_order()) {
    auto data = link->source()->safe_as<NodeData>();
    CHECK(data);
    if (data->source_node.get()) {
      producers.push_back(data->source_node.get());
    }
  }
  return producers;
}

std::vector<Node*> GetProducersInSet(
    const Node* node, const std::unordered_set<Node*>& node_set) {
  std::vector<Node*> producers;
  for (auto& link : node->inlinks_in_order()) {
    auto data = link->source()->safe_as<NodeData>();
    CHECK(data);
    if (data->source_node.get() && node_set.count(data->source_node.get())) {
      producers.push_back(data->source_node.get());
    }
  }
  return producers;
}

bool IsConstOp(const framework::Node* node) {
  static std::unordered_set<std::string> const_op_type = {
      "const_scalar", "fill_constant", "arange"};
  if (const_op_type.count(node->op()->name)) {
    return true;
  } else {
    return false;
  }
}

std::vector<int> GetInputShape(
    const Node* node,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  const auto& in_links = node->inlinks_in_order();
  CHECK(!in_links.empty()) << "Cannot get input shape from a no-input op \""
                           << node->id() << "\"";

  auto* producer_data = in_links.front()->source()->safe_as<NodeData>();
  CHECK_NOTNULL(producer_data);
  return shape_dict.at(producer_data->id());
}

std::vector<int> GetOutputShape(
    const Node* node,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto node_data = GetNodeData(node);
  return shape_dict.at(node_data->id());
}

Node* FindGlobalReducer(const std::vector<Node*>& nodes_in_order) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  for (auto iter = nodes_in_order.rbegin(); iter != nodes_in_order.rend();
       ++iter) {
    if (op_pattern_dict[(*iter)->op()] == framework::kReduction) {
      return *iter;
    }
  }

  return nullptr;
}

using Visitor = std::function<std::vector<Node*>(
    const Node*, const std::unordered_set<Node*>&)>;
Node* FindReducerInRoute(const Node* node,
                         const std::unordered_set<Node*>& nodes_set,
                         Visitor visitor) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  std::queue<const Node*> candidates;
  candidates.push(node);
  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    for (auto consumer : visitor(candidate, nodes_set)) {
      if (op_pattern_dict[consumer->op()] == framework::kReduction) {
        return consumer;
      }
      candidates.push(consumer);
    }
  }

  return nullptr;
}

Node* FindNearestReducer(const Node* node,
                         const std::unordered_set<Node*>& nodes_set) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // from consumers find reducer.
  auto reducer = FindReducerInRoute(node, nodes_set, GetConsumersInSet);
  if (reducer)
    return reducer;
  else
    return FindReducerInRoute(node, nodes_set, GetProducersInSet);
}

std::unordered_map<Node*, Node*> BuildVirtualConsumer(
    const GroupPtr& group,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  std::unordered_map<Node*, Node*> virtual_consumers;
  std::unordered_set<Node*> nodes_set = group->NodeSet();
  if (group->op_pattern_kind != framework::kReduction) {
    return virtual_consumers;
  }
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  Node* e_node = nullptr;
  Node* r_node = nullptr;
  for (auto t_node : group->master_nodes) {
    if (op_pattern_dict[t_node->op()] != framework::kReduction) {
      // producer exits reduce-sum and not consumers.
      if (!e_node && FindReducerInRoute(t_node, nodes_set, GetProducersInSet) &&
          GetConsumersInSet(t_node, nodes_set).size() == 0) {
        e_node = t_node;
      }
    } else if (!r_node) {
      r_node = t_node;
    }
  }

  // try to find reducer with different shape.
  for (auto t_node : group->output_nodes) {
    if (op_pattern_dict[t_node->op()] == framework::kReduction) {
      if (e_node) {
        virtual_consumers[t_node] = e_node;
      }
      continue;
    }
    if (FindNearestReducer(t_node, nodes_set)) {
      continue;
    }

    bool found = false;
    std::unordered_set<Node*> visited;
    std::queue<Node*> candidates;

    candidates.push(t_node);
    visited.insert(t_node);
    // from producers find reducer consumer.
    while (!found && !candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto producer : GetProducersInSet(candidate, nodes_set)) {
        if (visited.count(producer)) {
          continue;
        }

        auto reducer =
            FindReducerInRoute(producer, nodes_set, GetConsumersInSet);
        if (reducer) {
          virtual_consumers[t_node] = reducer;
          found = true;
          break;
        }
        candidates.push(producer);
        visited.insert(producer);
      }
    }

    auto output_shape = GetOutputShape(t_node, shape_dict);
    if (!found && t_node != e_node && e_node) {
      auto enode_output_shape = GetOutputShape(e_node, shape_dict);
      if (std::accumulate(output_shape.begin(),
                          output_shape.end(),
                          1,
                          std::multiplies<int>()) ==
          std::accumulate(enode_output_shape.begin(),
                          enode_output_shape.end(),
                          1,
                          std::multiplies<int>())) {
        virtual_consumers[t_node] = e_node;
        found = true;
      }
    }
    if (!found && r_node) {
      auto rnode_input_shape = GetInputShape(r_node, shape_dict);
      if (std::accumulate(output_shape.begin(),
                          output_shape.end(),
                          1,
                          std::multiplies<int>()) ==
          std::accumulate(rnode_input_shape.begin(),
                          rnode_input_shape.end(),
                          1,
                          std::multiplies<int>())) {
        virtual_consumers[t_node] = r_node;
        found = true;
      }
    }
  }
  // Establish virtual consumer relationships between output nodes with the same
  // shape. This allows the calculation of output nodes without affiliation to
  // be placed under the same loop.
  std::unordered_map<int, Node*> numel_consumers;
  for (auto out_node : group->output_nodes) {
    if (virtual_consumers.find(out_node) != virtual_consumers.end() ||
        !GetConsumersInSet(out_node, nodes_set).empty()) {
      continue;
    }
    auto shape = GetOutputShape(out_node, shape_dict);
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (numel_consumers.find(numel) == numel_consumers.end()) {
      numel_consumers.insert(std::make_pair(numel, out_node));
    } else {
      virtual_consumers[out_node] = numel_consumers[numel];
    }
  }

  return virtual_consumers;
}

std::vector<Node*> FindConsumers(
    Node* node,
    const std::unordered_set<Node*>& nodes_set,
    const std::unordered_map<Node*, Node*>& virtual_consumers) {
  auto consumers = GetConsumersInSet(node, nodes_set);
  if (virtual_consumers.count(node)) {
    consumers.push_back(virtual_consumers.find(node)->second);
  }
  return consumers;
}

std::vector<Node*> FindProducers(
    Node* node,
    const std::unordered_set<Node*>& nodes_set,
    const std::unordered_map<Node*, Node*>& virtual_consumers) {
  auto producers = GetProducersInSet(node, nodes_set);
  for (const auto& iter : virtual_consumers) {
    if (iter.second == node) {
      producers.push_back(iter.first);
    }
  }

  return producers;
}

std::vector<Node*> TopologicalOrder(
    const GroupPtr& group,
    const std::unordered_map<Node*, Node*>& virtual_consumers) {
  std::vector<Node*> nodes_in_order;
  std::unordered_set<Node*> nodes_set = group->NodeSet();

  while (!nodes_set.empty()) {
    std::set<Node*, utils::NodeCompare> tmp_node_set(nodes_set.begin(),
                                                     nodes_set.end());
    for (auto node : tmp_node_set) {
      auto consumers = FindConsumers(node, nodes_set, virtual_consumers);
      bool cant_be_erase = false;
      for (auto consumer : consumers) {
        if (nodes_set.count(consumer)) {
          cant_be_erase = true;
          break;
        }
      }

      if (cant_be_erase) continue;
      nodes_in_order.push_back(node);
      nodes_set.erase(node);
    }
  }

  return nodes_in_order;
}

std::vector<Node*> BFSTopologicalOrderWithPriority(
    const GroupPtr& group,
    const std::unordered_map<Node*, Node*>& virtual_consumers,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  struct NodeWithPriority {
    Node* node;
    int priority;
  };

  struct Comparator {
    bool operator()(const NodeWithPriority& lhs, const NodeWithPriority& rhs) {
      return lhs.priority > rhs.priority;
    }
  };

  std::vector<Node*> nodes_in_order;
  std::unordered_set<Node*> visited;
  std::unordered_set<Node*> nodes_set = group->NodeSet();
  std::unordered_map<Node*, int> degree_map;
  std::priority_queue<NodeWithPriority,
                      std::vector<NodeWithPriority>,
                      Comparator>
      priority_candidates;
  std::vector<int> visited_numel;

  // Calculate the priority of a node.
  // The smaller the value, the higher the priority.
  // Prioritize the same shape before considering OpPattern
  auto PriorityFunc = [&visited_numel, &shape_dict](const Node* node) -> int {
    auto node_shape = GetOutputShape(node, shape_dict);
    int numel = std::accumulate(
        node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());
    int index = -1;
    for (int i = 0; i < visited_numel.size(); ++i) {
      if (numel == visited_numel[i]) {
        index = i;
        break;
      }
    }
    if (index == -1) {
      index = visited_numel.size();
      visited_numel.push_back(numel);
    }
    auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
    return index * 10 + static_cast<int>(op_pattern_dict[node->op()]);
  };

  for (Node* node : nodes_set) {
    auto consumers = FindConsumers(node, nodes_set, virtual_consumers);
    // Some nodes may have multiple edges between them, resulting in duplicates
    // in the consumer. We only need to calculate once.
    std::unordered_set<Node*> consumers_without_duplicate(consumers.begin(),
                                                          consumers.end());
    degree_map[node] = consumers_without_duplicate.size();
    if (degree_map.at(node) == 0) {
      priority_candidates.push(NodeWithPriority{node, PriorityFunc(node)});
    }
  }

  // Nested BFS, outer layer traverses priority, inner layer performs BFS on
  // current priority.
  while (!priority_candidates.empty()) {
    Node* cur_priority_node = priority_candidates.top().node;
    priority_candidates.pop();

    std::queue<Node*> bfs_queue;
    bfs_queue.push(cur_priority_node);
    visited.insert(cur_priority_node);
    while (!bfs_queue.empty()) {
      Node* cur = bfs_queue.front();
      bfs_queue.pop();

      nodes_in_order.push_back(cur);
      auto producers = FindProducers(cur, nodes_set, virtual_consumers);
      std::unordered_set<Node*> producers_without_duplicate(producers.begin(),
                                                            producers.end());
      for (Node* node : producers_without_duplicate) {
        --degree_map[node];
        // Ensure that each node is accessed only once and maintain topological
        // order.
        if (visited.count(node) != 0 || degree_map[node] != 0) {
          continue;
        }
        // Perform BFS access to the current priority producers
        int node_priority = PriorityFunc(node);
        if (node_priority <= PriorityFunc(cur_priority_node)) {
          bfs_queue.push(node);
          visited.insert(node);
        } else {
          priority_candidates.push(NodeWithPriority{node, node_priority});
        }
      }
    }
  }

  return nodes_in_order;
}

bool WithoutLastDimInReduce(const std::vector<int>& shape,
                            const std::vector<int>& axes) {
  if (axes.empty()) {
    return false;
  }
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), shape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int sum_last_axes = 1;
  for (int idx = axes.back() + 1; idx < shape.size(); ++idx) {
    sum_last_axes *= shape[idx];
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

void LoopOrderAssignReduce(ir::IRSchedule& ir_sch,  // NOLINT
                           const std::string& block_name,
                           const std::vector<int>& axes,
                           const common::Target& target,
                           const bool just_reorder = false) {
  // reorder none-last reduce axis to last.
  // like: shape = [16,16,16,16,16],axes = [1,3] -> new order = [0, 2, 4, 1, 3].
  std::vector<int> order;
  int n_out_dims = ir_sch.GetLoops(block_name).size();
  for (int idx = 0; idx < n_out_dims; ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
      order.push_back(idx);
    }
  }
  for (auto axis : axes) {
    order.push_back(axis);
  }
  ir_sch.Reorder(ir_sch.GetBlock(block_name), order);

  if (just_reorder) {
    return;
  }
  // fuse others none-reduce axis.
  int last_dimension_num = n_out_dims - axes.back() - 1;
  int index = n_out_dims - last_dimension_num - axes.size();

  // fuse last_dimension_num - 1 times
  for (auto idx = index; idx < index + last_dimension_num - 1; ++idx) {
    ir_sch.Fuse(block_name, {index, index + 1});
  }

  auto loops = ir_sch.GetLoops(block_name);
  auto psize = ir::GetLoopExtent(loops[index]);

  if (psize > target.max_num_threads()) {
    for (int idx = target.max_num_threads(); idx > 0; --idx) {
      if (psize % idx == 0) {
        ir_sch.Split(loops[index], {-1, idx});
        break;
      }
      CHECK_GT(idx, 1);
    }
  }

  // fuse index - 1 times
  for (int idx = 0; idx < index - 1; ++idx) {
    ir_sch.Fuse(block_name, {0, 1});
  }
}

void LoopAssignReduceWithoutLast(ir::IRSchedule& ir_sch,  // NOLINT
                                 const std::string& block_name,
                                 const std::vector<int>& inshape,
                                 const std::vector<int>& axes,
                                 const common::Target& target) {
  int tail = 0;
  bool bound = true;
  auto shape = pe::GetFirstStepReduceShape(inshape, axes, bound, tail);
  CHECK(bound) << std::accumulate(inshape.begin(),
                                  inshape.end(),
                                  std::string(""),
                                  [](const std::string& left, const int right) {
                                    return left + std::to_string(right) + " ";
                                  });

  VLOG(4) << "LoopAssignReduceWithoutLast: THe input shape=["
          << cinn::utils::Join(inshape, ", ") << "], first step reduce shape=["
          << cinn::utils::Join(shape, ", ") << "]"
          << ", axes=[" << cinn::utils::Join(axes, ", ") << "], tail=" << tail;

  // remove loop size = 1 and remove axis in axes.
  std::vector<int> nshape, axes_shift_num(axes.size(), 0);
  for (int idx = 0; idx < shape.size(); ++idx) {
    if (shape[idx] == 1 && idx < axes.back()) {
      for (int j = 0; j < axes.size(); ++j) {
        if (axes[j] == idx) {
          // the loop size at axis is 1, need remove
          axes_shift_num[j] = -1;
        } else if (axes[j] > idx) {
          // the axies value need left shift
          axes_shift_num[j]++;
        }
      }
    } else {
      nshape.push_back(shape[idx]);
    }
  }

  // remove loop size - 1 axes
  std::vector<int> naxes;
  for (int i = 0; i < axes_shift_num.size(); ++i) {
    if (axes_shift_num[i] != -1) {
      // the axis do not need remove, but need left shift
      naxes.emplace_back(axes[i] - axes_shift_num[i]);
    }
  }

  // fuse tail for bind threadIdx.x
  int ptail = 1;
  int index = naxes.back() + 2;
  for (int idx = index; idx < nshape.size(); ++idx) {
    ptail *= nshape[idx];
  }
  nshape.resize(index);
  nshape.push_back(ptail);

  ir_sch.Split(block_name, 0, nshape);
  LoopOrderAssignReduce(ir_sch, block_name, naxes, target, true);

  // fuse loop for bind blockIdx.x
  auto loops = ir_sch.GetLoops(block_name);
  auto fsize = nshape.size() - (naxes.size() + 2);
  if (fsize > 1) {
    ir_sch.Fuse({loops.begin(), loops.begin() + fsize});
  }

  auto get_tile_size = [&](int idx) {
    auto range = GetLoopExtent(loops[idx - 1]);
    if (range > 32) {
      return 8;
    } else if (range > 16) {
      return 16;
    } else if (range > 4) {
      return 32;
    } else {
      return 64;
    }
  };

  std::vector<int> new_order;
  loops = ir_sch.GetLoops(block_name);
  if (fsize) {
    int tail_index = 2;
    auto tile_size = get_tile_size(tail_index);
    if (GetLoopExtent(loops[tail_index]) > tile_size) {
      // split index
      ir_sch.Split(loops[tail_index], {-1, tile_size});
      loops = ir_sch.GetLoops(block_name);
      // order
      new_order = {0, 2, 3, 1};
    } else {
      // order
      new_order = {0, 2, 1};
    }
  } else {
    int tail_index = 1;
    auto tile_size = get_tile_size(tail_index);
    if (GetLoopExtent(loops[tail_index]) > tile_size) {
      // split index
      ir_sch.Split(loops[tail_index], {-1, tile_size});
      loops = ir_sch.GetLoops(block_name);
      // order
      new_order = {1, 2, 0};
    } else {
      // order
      new_order = {1, 0};
    }
  }
  for (int idx = new_order.size(); idx < loops.size(); ++idx) {
    new_order.push_back(idx);
  }
  ir_sch.Reorder(block_name, new_order);
}

void LoopAssignReduceWithLast(ir::IRSchedule& ir_sch,  // NOLINT
                              const std::string& block_name,
                              const std::vector<int>& inshape,
                              const std::vector<int>& axes,
                              const common::Target& target) {
  // If the number of current device SM is smaller than the number of SM
  // required by Warp Reduce, the performance of Warp Reduce is better.
  // Otherwise, use Block Reduce.
  auto max_num_threads = common::DefaultNVGPUTarget().max_num_threads();
  int need_reduce_last_count = 1;
  for (int i = 0; i < inshape.size(); i++) {
    if (find(axes.begin(), axes.end(), i) == axes.end()) {
      need_reduce_last_count *= inshape[i];
    }
  }
  int warp_reduce_need_sm_count =
      ceil((need_reduce_last_count * 32) /
           static_cast<float>(target.get_max_threads_per_sm()));
  // Set Num_max_threads to 32 is Warp Reduce
  if (target.get_multi_processor_count() < warp_reduce_need_sm_count) {
    max_num_threads = 32;
  }
  // find first reduce and second reduce axis.
  int lane = 1;
  int index = static_cast<int>(axes.size()) - 1;

  for (; index >= 0; --index) {
    if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
      break;
    }
    lane *= inshape[axes[index]];
    if (index == 0 && lane <= max_num_threads) {
      LOG(FATAL)
          << "Error! lane is less equal than max_num_threads, Please check!";
    }
    if (lane >= max_num_threads / 2) {
      if (lane <= max_num_threads) {
        --index;
      }
      break;
    }
  }
  std::vector<int> first_axes(axes.begin(), axes.begin() + index + 1);
  if (lane > max_num_threads) {
    // last reduce axis size > 1024
    if (index == static_cast<int>(axes.size()) - 1) {
      int tail = max_num_threads;
      bool check_bound = true;
      for (; tail >= max_num_threads / 2; --tail) {
        if (lane % tail == 0) {
          check_bound = false;
          break;
        }
      }
      if (check_bound) {
        lane =
            ((lane + max_num_threads - 1) / max_num_threads) * max_num_threads;
        ir_sch.Split(block_name, axes[index], {lane});
      }
      int idx = max_num_threads;
      do {
        if (lane % idx == 0) {
          ir_sch.Split(block_name, axes[index], {-1, idx});
          break;
        }
        --idx;
      } while (idx >= max_num_threads / 2);
      // if can't be divide by(1024, 512), it's shouldn't be fused.
      CHECK_GE(idx, max_num_threads / 2) << "Check bounds exist, can't fuse!";
    } else {
      int axis = axes[index];
      int prefix = inshape[axis];
      int tail = lane / prefix;
      for (int idx = max_num_threads / tail; idx > (max_num_threads / 2) / tail;
           --idx) {
        if (prefix % idx == 0) {
          ir_sch.Split(block_name, axis, {-1, idx});
          break;
        }
        CHECK_GT(idx, (max_num_threads / 2) / tail)
            << "Error, it's shouldn't fuse!";
      }
    }
    LoopOrderAssignReduce(ir_sch, block_name, first_axes, target);
    // The current one-dimensional reduce does not make full use of SM.
    // This case is optimized into a two-dimensional.
    auto loops = ir_sch.GetLoops(block_name);
    auto block_dim_x = loops[1].As<ir::For>()->extent.as_int32();
    int block_dim_y = block_dim_x <= 32 ? 2 : 1;
    if (block_dim_y != 1) {
      ir_sch.Split(loops[0], {-1, block_dim_y});
    }
  } else {
    int fuse_times = axes.size() - (index + 1) - 1;
    for (int idx = 0; idx < fuse_times; ++idx) {
      ir_sch.Fuse(block_name, {axes[index + 1], axes[index + 1] + 1});
    }
    LoopOrderAssignReduce(ir_sch, block_name, first_axes, target, true);
    // fuse axis before reduce to bind blockidx.
    for (int idx = 0; idx < static_cast<int>(inshape.size() - axes.size()) - 1;
         ++idx) {
      ir_sch.Fuse(block_name, {0, 1});
    }
  }
}

bool CanbeInline(Node* node,
                 const std::vector<Node*> consumers,
                 const Node* reducer,
                 const std::unordered_set<Node*> masters,
                 const GroupPtr& group,
                 const std::unordered_set<Node*>& nodes_set,
                 const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  if (group->output_nodes.count(node)) {
    return false;
  }

  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  for (auto consumer : consumers) {
    if (op_pattern_dict[consumer->op()] == framework::kReduction) {
      return false;
    }
  }

  if (IsConstOp(node)) {
    return true;
  }

  if (op_pattern_dict[node->op()] == framework::kReduction) {
    return false;
  }

  if (consumers.size() == 1) {
    return true;
  }

  if (reducer) {
    // node is before reducer and node is not after reduce.
    if (FindReducerInRoute(node, nodes_set, GetConsumersInSet) &&
        !FindReducerInRoute(node, nodes_set, GetProducersInSet)) {
      auto node_shape = GetOutputShape(node, shape_dict);
      auto input_shape = GetInputShape(reducer, shape_dict);
      // check with same shape with reducer input.
      if (std::accumulate(node_shape.begin(),
                          node_shape.end(),
                          1,
                          std::multiplies<int>()) !=
          std::accumulate(input_shape.begin(),
                          input_shape.end(),
                          1,
                          std::multiplies<int>())) {
        return true;
      }
    }

    return false;
  } else {
    auto node_shape = GetOutputShape(node, shape_dict);
    auto node_size = std::accumulate(
        node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());

    for (auto master : masters) {
      auto master_shape = GetOutputShape(master, shape_dict);
      auto master_size = std::accumulate(
          master_shape.begin(), master_shape.end(), 1, std::multiplies<int>());
      if (node_size != master_size) {
        return true;
      }
    }

    return false;
  }
}

Node* GetMasterToComputeAt(
    Node* node,
    const std::vector<Node*>& nodes_in_order,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set,
    const std::unordered_map<Node*, Node*>& virtual_consumers,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // if node is reduction, try find horizontal to compute at.
  if (op_pattern_dict[node->op()] == framework::kReduction) {
    // find all reduce node has done schedule.
    std::unordered_set<Node*> done_schedule;
    for (auto tmp : nodes_in_order) {
      if (tmp == node) {
        break;
      }
      if (op_pattern_dict[tmp->op()] == framework::kReduction) {
        done_schedule.insert(tmp);
      }
    }
    // remove all consuemr reducer node of node from done_schedule.
    std::unordered_set<Node*> visited;
    std::queue<Node*> candidates;
    candidates.push(node);

    while (!candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto consumer : GetConsumersInSet(candidate, nodes_set)) {
        // remove reduction node from done_schedule.
        if (op_pattern_dict[consumer->op()] == framework::kReduction) {
          done_schedule.erase(consumer);
        }
        if (visited.count(consumer)) {
          continue;
        }
        candidates.push(consumer);
        visited.insert(consumer);
      }
    }

    if (done_schedule.size()) {
      auto shape = shape_dict.at(node->inlinks_in_order()[0]->source()->id());
      for (auto rnode : done_schedule) {
        auto rshape =
            shape_dict.at(rnode->inlinks_in_order()[0]->source()->id());
        if (shape == rshape) {
          return rnode;
        }
      }
      return *done_schedule.begin();
    }
  }

  // collect all consumers.
  std::unordered_set<Node*> visited, masters;
  std::queue<Node*> candidates;
  candidates.push(node);

  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    auto consumers = FindConsumers(candidate, nodes_set, virtual_consumers);
    for (auto consumer : consumers) {
      if (visited.count(consumer)) {
        continue;
      }
      if (nodes_inline.count(consumer)) {
        candidates.push(consumer);
        visited.insert(consumer);
      } else {
        masters.insert(consumer);
      }
    }
  }

  // nodes-in-order
  for (int idx = 0; idx < nodes_in_order.size(); ++idx) {
    if (nodes_in_order[idx] == node) {
      for (int idy = idx - 1; idy >= 0; --idy) {
        if (masters.count(nodes_in_order[idy])) {
          return nodes_in_order[idy];
        }
      }
      break;
    }
  }
  return nullptr;
}

void LoopAssignReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    const Node* node,
    const Node* reducer,
    const Target& target,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // if node is reducer, return.
  if (op_pattern_dict[node->op()] == framework::kReduction) {
    return;
  }
  auto node_data = GetNodeData(node);
  auto reducer_data = GetNodeData(reducer);

  // flatten loops.
  auto loops = ir_sch.GetLoops(node_data->id());
  // do loop flatten.
  if (op_pattern_dict[node->op()] == framework::kElementWise) {
    ir_sch.FlattenLoops(loops, true);
  } else {
    ir_sch.FlattenLoops(loops, false);
  }

  // shape and axis.
  CHECK(shape_dict.count(reducer->inlinks_in_order()[0]->source()->id()));
  auto shape = shape_dict.at(reducer->inlinks_in_order()[0]->source()->id());
  auto axes = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }

  auto copy_loop_info = [](std::vector<ir::Expr>& loops,
                           std::vector<ir::Expr>& rloops) {
    for (int idx = 0; idx < std::min(rloops.size(), loops.size()); ++idx) {
      auto l0 = rloops[idx].As<ir::For>();
      auto l1 = loops[idx].As<ir::For>();
      l1->set_for_type(l0->for_type());
      l1->set_bind_info(l0->bind_info());
    }
  };

  auto node_shape = shape_dict.at(node_data->id());
  // The output shape of node is different from that of reduce node
  if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) !=
      std::accumulate(
          node_shape.begin(), node_shape.end(), 1, std::multiplies<int>())) {
    // get loop factors of reduce node
    int extend = 1;
    std::vector<int> factors;
    loops = ir_sch.GetLoops(node_data->id());
    auto rloops = ir_sch.GetLoops(reducer_data->id());

    for (auto& loop : rloops) {
      if (extend >= loops.back().As<ir::For>()->extent.as_int32() &&
          factors.size() && loop.As<ir::For>()->extent.as_int32() > 1) {
        break;
      }
      extend *= loop.As<ir::For>()->extent.as_int32();
      factors.push_back(loop.As<ir::For>()->extent.as_int32());
    }

    // If there are IfThenElse stmt in loop, we need to find out the indices in
    // condition, and special treatment should be applied to loops with these
    // indices. We apply two step split on loop of src node to align the loop of
    // reduce node.
    std::unordered_set<int> loop_index_in_if;
    auto first_reduce_loop = rloops.front();
    // collect if
    auto if_checker = [](const Expr* x) { return x->As<ir::IfThenElse>(); };
    auto if_set = ir::ir_utils::CollectIRNodesWithoutTensor(
        first_reduce_loop.As<ir::For>()->body, if_checker);
    std::string reduce_block_name = reducer_data->id();
    for (auto if_expr : if_set) {
      auto checker = [reduce_block_name](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() &&
               x->As<ir::ScheduleBlockRealize>()
                       ->schedule_block.As<ir::ScheduleBlock>()
                       ->name == reduce_block_name;
      };
      auto blocks_in_if =
          ir::ir_utils::CollectIRNodesWithoutTensor(if_expr, checker);
      if (!blocks_in_if.empty()) {
        ir::Expr condition = if_expr.As<ir::IfThenElse>()->condition;
        auto indices_in_if = ir::ir_utils::CollectIRNodesWithoutTensor(
            condition, [](const Expr* x) { return x->As<ir::_Var_>(); });
        for (int i = 0; i < rloops.size(); ++i) {
          std::string var_name = rloops[i].As<ir::For>()->loop_var->name;
          auto find_var_iter =
              std::find_if(indices_in_if.begin(),
                           indices_in_if.end(),
                           [&var_name](const ir::Expr& x) {
                             return x.As<ir::_Var_>()->name == var_name;
                           });
          if (find_var_iter != indices_in_if.end()) {
            loop_index_in_if.insert(i);
          }
        }
        break;
      }
    }

    // prepare factors of two step split
    std::vector<int> first_step_factors;
    std::vector<int> second_step_factors;
    int second_start_loop_index;
    for (int i = 0; i < factors.size(); ++i) {
      if (loop_index_in_if.count(i) == 0) {
        first_step_factors.push_back(factors[i]);
      } else if (loop_index_in_if.count(i) != 0 &&
                 second_step_factors.empty()) {
        first_step_factors.push_back(-1);
        second_step_factors.push_back(factors[i]);
        second_start_loop_index = i;
      } else if (loop_index_in_if.count(i) != 0 &&
                 !second_step_factors.empty()) {
        second_step_factors.push_back(factors[i]);
      }
    }
    // do two step split
    if (!first_step_factors.empty()) {
      ir_sch.Split(loops.back(), first_step_factors);
      loops = ir_sch.GetLoops(node_data->id());
    }
    if (!second_step_factors.empty()) {
      ir_sch.Split(loops.at(second_start_loop_index), second_step_factors);
      loops = ir_sch.GetLoops(node_data->id());
    }

    // copy loop info form rloops.
    copy_loop_info(loops, rloops);
    return;
  }

  // node output is same shape with reduce input.
  if (WithoutLastDimInReduce(shape, axes)) {
    // if using two strep reduce.
    if (tensor_map.count(reducer_data->id() + "_1")) {
      VLOG(4) << "Try assign loop of " << node_data->id()
              << " into two strep reduce loop of " << reducer_data->id();
      LoopAssignReduceWithoutLast(ir_sch, node_data->id(), shape, axes, target);
      auto nloops = ir_sch.GetLoops(node_data->id());
      auto rloops = ir_sch.GetLoops(
          tensor_map.find(reducer_data->id() + "_0")->second->name);

      VLOG(4) << node_data->id() << "'s loop level is " << nloops.size()
              << ", and " << reducer_data->id() << "'s loop level is "
              << rloops.size();
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(node_data->id());
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else {
      VLOG(4) << "Try assign loop of " << node_data->id()
              << " into reduce loop of " << reducer_data->id();

      auto nloops = ir_sch.GetLoops(node_data->id());
      ir_sch.Split(nloops.back(), shape);
      LoopOrderAssignReduce(ir_sch, node_data->id(), axes, target);
      nloops = ir_sch.GetLoops(node_data->id());
      auto rloops =
          ir_sch.GetLoops(tensor_map.find(reducer_data->id())->second->name);
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(node_data->id());
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    }
  } else {
    if (tensor_map.count(reducer_data->id() + "_1")) {
      {
        auto nloops = ir_sch.GetLoops(node_data->id());
        ir_sch.Split(nloops.back(), shape);
      }
      LoopAssignReduceWithLast(ir_sch, node_data->id(), shape, axes, target);

      auto nloops = ir_sch.GetLoops(node_data->id());
      auto rloops = ir_sch.GetLoops(
          tensor_map.find(reducer_data->id() + "_1")->second->name);
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(node_data->id());
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else if (tensor_map.count(reducer_data->id() + "_0")) {
      auto tensor = tensor_map.find(reducer_data->id() + "_0")->second;
      auto rloops = ir_sch.GetLoops(tensor->name);
      std::vector<int> factors;
      for (auto& loop : rloops) {
        factors.push_back(loop.As<ir::For>()->extent.as_int32());
      }
      auto nloops = ir_sch.GetLoops(node_data->id());
      ir_sch.Split(nloops.back(), factors);

      nloops = ir_sch.GetLoops(node_data->id());
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else {
      LOG(FATAL) << "Error! Unkown Reduce Type!";
    }
  }
}

// The struct used to remove the original block in ComputeAt.
class RemoveExpr : public ir::IRMutator<> {
 public:
  explicit RemoveExpr(const Expr& target) : target_(target) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();
    auto iter = std::find(node->stmts.begin(), node->stmts.end(), target_);
    if (iter != node->stmts.end()) {
      node->stmts.erase(iter);
    } else {
      for (auto stmt : node->stmts) {
        IRMutator::Visit(&stmt, &stmt);
      }
    }
  }

 private:
  const Expr& target_;
};

void MergeLoops(ir::Expr root,
                std::vector<ir::Expr>& src,  // NOLINT
                std::vector<ir::Expr>& dst,  // NOLINT
                int index) {
  if (index < 0) {
    return;
  }
  CHECK_GT(src.size(), index) << "\nindex -> " << index << "\n" << src[0];
  CHECK_GT(dst.size(), index) << "\nindex -> " << index << "\n" << dst[0];

  if (src[0] == dst[0]) {
    return;
  }

  std::vector<ir::Var> src_vars;
  std::vector<ir::Expr> dst_vars;
  for (int idx = 0; idx <= index; ++idx) {
    src_vars.push_back(src[idx].As<ir::For>()->loop_var);
    dst_vars.push_back(ir::Expr(dst[idx].As<ir::For>()->loop_var));
  }

  auto src_body = src[index].As<ir::For>()->body;
  ReplaceExpr(&src_body, src_vars, dst_vars);
  dst[index].As<ir::For>()->body =
      ir::Block::Make({src_body, dst[index].As<ir::For>()->body});

  RemoveExpr remove_expr(src[0]);
  remove_expr(&root);
}

void InsertSyncThread(
    ir::IRSchedule& ir_sch,  // NOLINT
    const Node* node,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  CHECK(shape_dict.count(node->inlinks_in_order()[0]->source()->id()));
  auto shape = shape_dict.at(node->inlinks_in_order()[0]->source()->id());
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("dim"));
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }
  if (!WithoutLastDimInReduce(shape, axes)) {
    return;
  }

  auto node_data = GetNodeData(node);
  std::string post = "";
  for (int idx = 0;; ++idx) {
    if (!tensor_map.count(node_data->id() + post)) {
      break;
    }
    auto tensor = tensor_map.find(node_data->id() + post)->second;
    if (!ir_sch.HasBlock(tensor->name)) {
      break;
    }

    post = "_" + std::to_string(idx);
    if (idx > 0) {
      // insert syncthreads.
      auto loops = ir_sch.GetLoops(node_data->id());
      ir_sch.SyncThreads(loops[loops.size() - 2], false);
      return;
    }
  }
}

// The struct used to remove the original block in ComputeAt.
class InsertExpr : public ir::IRMutator<> {
 public:
  InsertExpr(Expr& target, Expr& anchor) : target_(target), anchor_(anchor) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();
    auto iter = std::find(node->stmts.begin(), node->stmts.end(), anchor_);
    if (iter != node->stmts.end()) {
      node->stmts.insert(iter, target_);
    } else {
      for (auto stmt : node->stmts) {
        IRMutator::Visit(&stmt, &stmt);
      }
    }
  }

 private:
  Expr target_;
  Expr anchor_;
};

void MergeReduceToReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    const Node* node,
    const Node* master,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  auto node_data = GetNodeData(node);
  auto master_data = GetNodeData(master);

  CHECK(shape_dict.count(node->inlinks_in_order()[0]->source()->id()));
  auto shape = shape_dict.at(node->inlinks_in_order()[0]->source()->id());
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("dim"));
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }
  if (WithoutLastDimInReduce(shape, axes)) {
    auto mshape = shape_dict.at(master->inlinks_in_order()[0]->source()->id());
    if (tensor_map.count(node_data->id() + "_1")) {
      if (shape == mshape) {
        // second step reduce
        {
          auto block = ir_sch.GetBlock(node_data->id());
          auto loops = ir_sch.GetLoops(master_data->id());
          ir_sch.SimpleComputeAt(block, loops.back());
          // reduce init
          {
            auto block = ir_sch.GetBlock(node_data->id() + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_data->id() + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
        // first step reduce
        {
          auto n_tensor = tensor_map.find(node_data->id() + "_0")->second;
          auto m_tensor = tensor_map.find(master_data->id() + "_0")->second;

          auto block = ir_sch.GetBlock(n_tensor->name);
          auto loops = ir_sch.GetLoops(m_tensor->name);
          ir_sch.SimpleComputeAt(block, loops.back());
          // reduce init
          {
            auto block = ir_sch.GetBlock(n_tensor->name + "__reduce_init");
            auto loops = ir_sch.GetLoops(m_tensor->name + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
      } else {
        auto n_tensor = tensor_map.find(node_data->id() + "_0")->second;
        auto m_tensor = tensor_map.find(master_data->id() + "_0")->second;
        if (n_tensor->shape == m_tensor->shape) {
          // second step reduce
          {
            auto block = ir_sch.GetBlock(node_data->id());
            auto loops = ir_sch.GetLoops(master_data->id());
            ir_sch.SimpleComputeAt(block, loops.back());
            // reduce init
            {
              auto block = ir_sch.GetBlock(node_data->id() + "__reduce_init");
              auto loops = ir_sch.GetLoops(master_data->id() + "__reduce_init");
              ir_sch.SimpleComputeAt(block, loops.back());
            }
          }
          // first step reduce
          {
            auto n_tensor = tensor_map.find(node_data->id() + "_0")->second;
            auto m_tensor = tensor_map.find(master_data->id() + "_0")->second;

            auto n_loops = ir_sch.GetLoops(n_tensor->name + "__reduce_init");
            auto m_loops = ir_sch.GetLoops(m_tensor->name + "__reduce_init");

            CHECK_EQ(n_loops.size(), m_loops.size());
            MergeLoops(ir_sch.GetModule().GetExprs().at(0),
                       n_loops,
                       m_loops,
                       n_loops.size() - 1);
          }
        } else {
          LOG(FATAL) << "not support this type fusion!";
        }
      }
    } else {
      if (shape == mshape) {
        // reduce loop
        {
          auto block = ir_sch.GetBlock(node_data->id());
          auto loops = ir_sch.GetLoops(master_data->id());
          ir_sch.SimpleComputeAt(block, loops.back());
          // reduce init
          {
            auto block = ir_sch.GetBlock(node_data->id() + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_data->id() + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
      } else {
        // reduce loop
        {
          auto block = ir_sch.GetBlock(node_data->id());
          auto nloops = ir_sch.GetLoops(node_data->id());
          auto mloops = ir_sch.GetLoops(master_data->id());
          for (int idx = 0; idx < mloops.size(); ++idx) {
            if (GetLoopExtent(nloops[idx]) != GetLoopExtent(mloops[idx])) {
              ir_sch.SimpleComputeAt(block, mloops[idx - 1]);
              break;
            }
          }
          // reduce init
          {
            auto block = ir_sch.GetBlock(node_data->id() + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_data->id() + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
      }
    }
  } else {
    if (tensor_map.count(node_data->id() + "_1")) {
      // identity
      {
        auto block = ir_sch.GetBlock(node_data->id());
        auto loops = ir_sch.GetLoops(master_data->id());
        ir_sch.SimpleComputeAt(block, loops.back());
      }
      // reduce
      {
        auto n_tensor = tensor_map.find(node_data->id() + "_1")->second;
        auto m_tensor = tensor_map.find(master_data->id() + "_1")->second;

        auto block = ir_sch.GetBlock(n_tensor->name);
        auto loops = ir_sch.GetLoops(m_tensor->name);
        ir_sch.SimpleComputeAt(block, loops.back());
        // reduce init
        {
          auto block = ir_sch.GetBlock(n_tensor->name + "__reduce_init");
          auto loops = ir_sch.GetLoops(m_tensor->name + "__reduce_init");
          ir_sch.SimpleComputeAt(block, loops.back());
        }
      }
      // block shuffle
      {
        auto n_tensor = tensor_map.find(node_data->id() + "_0")->second;
        auto m_tensor = tensor_map.find(master_data->id() + "_0")->second;

        auto n_block = ir_sch.GetBlock(n_tensor->name);
        auto m_block = ir_sch.GetBlock(m_tensor->name);

        auto n_loops = ir_sch.GetLoops(n_tensor->name);
        auto m_loops = ir_sch.GetLoops(m_tensor->name);
        CHECK_EQ(n_loops.size(), m_loops.size());

        std::vector<ir::Var> src_vars;
        std::vector<ir::Expr> dst_vars;
        for (int idx = 0; idx < m_loops.size(); ++idx) {
          src_vars.push_back(n_loops[idx].As<ir::For>()->loop_var);
          dst_vars.push_back(ir::Expr(m_loops[idx].As<ir::For>()->loop_var));
        }
        ReplaceExpr(&n_block, src_vars, dst_vars);

        InsertExpr insert_expr(n_block, m_block);
        insert_expr(&m_loops.back());

        RemoveExpr remove_expr(n_loops[0]);
        remove_expr(&ir_sch.GetModule().GetExprs().at(0));
      }
    } else if (tensor_map.count(node_data->id() + "_0")) {
      // identity
      {
        auto block = ir_sch.GetBlock(node_data->id());
        auto loops = ir_sch.GetLoops(master_data->id());
        ir_sch.SimpleComputeAt(block, loops.back());
      }
      // shuffle reduce
      {
        auto n_tensor = tensor_map.find(node_data->id() + "_0")->second;
        auto m_tensor = tensor_map.find(master_data->id() + "_0")->second;

        auto block = ir_sch.GetBlock(n_tensor->name);
        auto loops = ir_sch.GetLoops(m_tensor->name);
        ir_sch.SimpleComputeAt(block, loops.back());
      }
    } else {
      LOG(FATAL) << "Error! Unkown Reduce Type, Please Check!";
    }
  }
}

void MergeReduceLoop(
    ir::IRSchedule& ir_sch,  // NOLINT
    Node* node,
    const Node* master,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  if (op_pattern_dict[master->op()] == kReduction && node != master) {
    MergeReduceToReduce(ir_sch, node, master, shape_dict, tensor_map);
    return;
  }

  auto node_data = GetNodeData(node);
  auto master_data = GetNodeData(master);

  int min_index_loop = INT_MAX;
  std::string post_ = "", post__ = "_0";
  for (int idx = 0;; ++idx) {
    if (!tensor_map.count(node_data->id() + post__)) {
      break;
    }
    auto tensor_ = tensor_map.find(node_data->id() + post_)->second;
    auto tensor__ = tensor_map.find(node_data->id() + post__)->second;
    if (!ir_sch.HasBlock(tensor__->name)) {
      break;
    }

    auto dst_loops = ir_sch.GetLoops(tensor_->name);
    auto src_loops = ir_sch.GetLoops(tensor__->name);
    int index = -1;
    while (src_loops[index + 1].As<ir::For>()->extent.as_int32() ==
           dst_loops[index + 1].As<ir::For>()->extent.as_int32()) {
      ++index;
      if (src_loops.size() == index + 1 || dst_loops.size() == index + 1) {
        break;
      }
    }
    min_index_loop = std::min(min_index_loop, index);
    MergeLoops(
        ir_sch.GetModule().GetExprs().at(0), src_loops, dst_loops, index);

    post_ = "_" + std::to_string(idx);
    post__ = "_" + std::to_string(idx + 1);
  }
  InsertSyncThread(ir_sch, node, shape_dict, tensor_map);

  if (node == master) return;
  auto node_loops = ir_sch.GetLoops(node_data->id());
  auto master_loops = ir_sch.GetLoops(master_data->id());

  int index = std::min(node_loops.size(), master_loops.size()) - 1;
  do {
    // if loop range is not equal.
    if (node_loops[index].As<ir::For>()->extent.as_int32() !=
        master_loops[index].As<ir::For>()->extent.as_int32()) {
      continue;
    }

    MergeLoops(ir_sch.GetModule().GetExprs().at(0),
               node_loops,
               master_loops,
               std::min(index, min_index_loop));
    if (index > min_index_loop) {
      auto block = ir_sch.GetBlock(node_data->id());
      auto loops = ir_sch.GetLoops(master_data->id());
      ir_sch.SimpleComputeAt(block, loops.back());

      if (ir_sch.HasBlock(node_data->id() + "__reduce_init")) {
        auto block = ir_sch.GetBlock(node_data->id() + "__reduce_init");
        auto loops = ir_sch.GetLoops(master_data->id());
        ir_sch.SimpleComputeAt(block, loops.back());
      }
    }

    break;
  } while (--index >= 0);
}

// The struct used to find all ir::For or ScheduleBlock in given block.
class FindExprInBlock : public ir::IRMutator<> {
 public:
  FindExprInBlock() {}

  std::vector<ir::Expr> operator()(Expr* expr) {
    IRMutator::Visit(expr, expr);
    return exprs_;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* expr, Expr* op) override {
    exprs_.push_back(*op);
  }

  void Visit(const ir::For* expr, Expr* op) override { exprs_.push_back(*op); }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto node = op->As<ir::Block>();
    for (auto stmt : node->stmts) {
      IRMutator::Visit(&stmt, &stmt);
    }
  }

  std::vector<ir::Expr> exprs_;
};

void LoopComputeAt(
    ir::IRSchedule& ir_sch,  // NOLINT
    Node* node,
    const Node* master,
    const GroupPtr& group,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  if (!group->output_nodes.count(node)) {
    auto block = ir_sch.GetBlock(GetNodeData(node)->id());
    ir_sch.SetBuffer(block, "local");
  }

  if (op_pattern_dict[node->op()] == framework::kReduction) {
    MergeReduceLoop(ir_sch, node, master, shape_dict, tensor_map);
    return;
  }

  if (node == master) return;

  auto node_data = GetNodeData(node);
  auto master_data = GetNodeData(master);

  auto node_loops = ir_sch.GetLoops(node_data->id());
  auto master_loops = ir_sch.GetLoops(master_data->id());

  if (op_pattern_dict[master->op()] == framework::kReduction) {
    // find real master loops.
    std::string prefix = "", post = "";
    for (int idx = 0;; ++idx) {
      if (!tensor_map.count(master_data->id() + post)) {
        break;
      }
      auto tensor = tensor_map.find(master_data->id() + post)->second;
      if (!ir_sch.HasBlock(tensor->name)) {
        break;
      }

      prefix = post;
      post = "_" + std::to_string(idx);
    }

    auto tensor = tensor_map.find(master_data->id() + prefix)->second;
    master_loops = ir_sch.GetLoops(tensor->name);
  }

  int index = std::min(node_loops.size(), master_loops.size()) - 1;
  do {
    // if loop range is not equal.
    if (node_loops[index].As<ir::For>()->extent.as_int32() !=
        master_loops[index].As<ir::For>()->extent.as_int32()) {
      continue;
    }
    MergeLoops(
        ir_sch.GetModule().GetExprs().at(0), node_loops, master_loops, index);

    break;
  } while (--index >= 0);
}

std::unordered_map<std::string, NodeData*> GetNodeDataSet(
    const std::unordered_set<Node*>& nodes_set) {
  std::unordered_map<std::string, NodeData*> node_data_set;
  for (auto node : nodes_set) {
    auto node_data = GetNodeData(node);
    node_data_set[node_data->id()] = node_data;
  }
  return node_data_set;
}

std::unordered_set<Node*> GetMasters(
    Node* node,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set) {
  // find consumer
  std::unordered_set<Node*> visited;
  std::queue<Node*> candidates;
  candidates.push(node);
  std::unordered_set<Node*> masters;

  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    auto consumers = GetConsumersInSet(candidate, nodes_set);
    for (auto consumer : consumers) {
      if (visited.count(consumer)) {
        continue;
      }
      if (nodes_inline.count(consumer)) {
        candidates.push(consumer);
        visited.insert(consumer);
      } else {
        masters.insert(consumer);
      }
    }
  }

  return masters;
}

void SyncThreadWithShared(
    ir::IRSchedule& ir_sch,  // NOLINT
    const GroupPtr& group,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  auto exprs_inorder = ir_sch.GetAllBlocks();
  auto node_data_set = GetNodeDataSet(nodes_set);
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  std::unordered_set<std::string> sync_mark;
  auto check_sync_mark = [&](const int start, const std::string& m_id) {
    for (int idx = start + 1; exprs_inorder.size(); ++idx) {
      auto expr = exprs_inorder[idx];
      CHECK(expr.As<ir::ScheduleBlockRealize>());
      CHECK(expr.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>());
      auto block = expr.As<ir::ScheduleBlockRealize>()
                       ->schedule_block.As<ir::ScheduleBlock>();

      if (sync_mark.count(block->name)) {
        return false;
      }

      if (block->name == m_id) {
        return true;
      }
    }
    return false;
  };

  for (int idx = 0; idx < exprs_inorder.size() - 1; ++idx) {
    auto expr = exprs_inorder[idx];
    CHECK(expr.As<ir::ScheduleBlockRealize>());
    CHECK(expr.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>());
    auto block = expr.As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();

    if (!node_data_set.count(block->name)) {
      continue;
    }
    auto node_data = node_data_set.find(block->name)->second;
    auto node = node_data->source_node.get();
    auto node_shape = shape_dict.at(node_data->id());

    auto masters = GetMasters(node, nodes_inline, nodes_set);
    if (masters.empty()) {
      continue;
    }

    bool do_set_buffer_to_shared = false;
    for (auto master : masters) {
      auto master_data = GetNodeData(master);
      auto master_shape = shape_dict.at(master_data->id());
      if (op_pattern_dict[master->op()] == framework::kReduction) {
        master_shape =
            shape_dict.at(master->inlinks_in_order()[0]->source()->id());
      }

      auto node_size = std::accumulate(
          node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());
      auto master_size = std::accumulate(
          master_shape.begin(), master_shape.end(), 1, std::multiplies<int>());

      if (node_size != master_size) {
        if (check_sync_mark(idx, master_data->id())) {
          auto loops = ir_sch.GetLoops(master_data->id());
          ir_sch.SyncThreads(loops.back(), false);
          sync_mark.insert(master_data->id());
        }
        do_set_buffer_to_shared = true;
      }
    }
    if (do_set_buffer_to_shared &&
        group->output_nodes.find(node) == group->output_nodes.end()) {
      auto block = ir_sch.GetBlock(node_data->id());
      ir_sch.SetBuffer(block, "shared");
    }
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
