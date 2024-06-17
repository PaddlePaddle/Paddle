// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"

#include <algorithm>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/nn_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

::pir::Operation* FindGlobalReducer(
    const std::vector<::pir::Operation*>& ops_in_order) {
  for (auto& op : ops_in_order) {
    if (CompatibleInfo::OpKind(*op) == framework::kReduction) {
      return op;
    }
  }
  return nullptr;
}

std::vector<::pir::Operation*> GetConsumersInSet(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set) {
  std::vector<::pir::Operation*> consumers;
  for (auto& out : op->results()) {
    for (auto use_iter = out.use_begin(); use_iter != out.use_end();
         ++use_iter) {
      ::pir::Operation* consumer = use_iter->owner();
      CHECK(consumer);
      if (ops_set.count(consumer)) {
        consumers.push_back(consumer);
      }
    }
  }
  return consumers;
}

std::vector<::pir::Operation*> GetProducers(::pir::Operation* op) {
  std::vector<::pir::Operation*> producers;
  for (auto& source : op->operands_source()) {
    auto* producer_op = source.defining_op();
    CHECK(producer_op);
    producers.push_back(producer_op);
  }
  return producers;
}

std::vector<::pir::Operation*> GetProducersInSet(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set) {
  std::vector<::pir::Operation*> producers;
  for (auto& producer_op : GetProducers(op)) {
    CHECK(producer_op);
    if (ops_set.count(producer_op)) {
      producers.push_back(producer_op);
    }
  }
  return producers;
}

std::vector<::pir::Operation*> FindConsumers(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers) {
  auto consumers = GetConsumersInSet(op, ops_set);
  if (virtual_consumers.count(op)) {
    consumers.push_back(virtual_consumers.find(op)->second);
  }
  return consumers;
}

std::vector<::pir::Operation*> FindProducers(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers) {
  auto producers = GetProducersInSet(op, ops_set);
  for (const auto& iter : virtual_consumers) {
    if (iter.second == op) {
      producers.push_back(iter.first);
    }
  }

  return producers;
}

using Visitor = std::function<std::vector<::pir::Operation*>(
    ::pir::Operation*, const std::unordered_set<::pir::Operation*>&)>;
::pir::Operation* FindReducerInRoute(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set,
    Visitor visitor) {
  std::queue<::pir::Operation*> candidates;
  candidates.push(op);
  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    for (auto consumer : visitor(candidate, ops_set)) {
      if (CompatibleInfo::OpKind(*consumer) == framework::kReduction) {
        return consumer;
      }
      candidates.push(consumer);
    }
  }

  return nullptr;
}

::pir::Operation* FindNearestReducer(
    ::pir::Operation* op,
    const std::unordered_set<::pir::Operation*>& ops_set) {
  // from consumers find reducer.
  if (auto reducer = FindReducerInRoute(op, ops_set, GetConsumersInSet)) {
    return reducer;
  } else {
    return FindReducerInRoute(op, ops_set, GetProducersInSet);
  }
}

std::unordered_map<::pir::Operation*, ::pir::Operation*> BuildVirtualConsumer(
    const GroupPtr& group) {
  std::unordered_map<::pir::Operation*, ::pir::Operation*> virtual_consumers;
  std::unordered_set<::pir::Operation*> ops_set(group->ops.begin(),
                                                group->ops.end());
  if (group->op_pattern_kind != framework::kReduction) {
    return virtual_consumers;
  }

  ::pir::Operation* e_op = nullptr;
  ::pir::Operation* r_op = nullptr;
  for (auto master_op : group->master_ops) {
    if (CompatibleInfo::OpKind(*master_op) != framework::kReduction) {
      // producer exits reduce-sum and not consumers.
      if (!e_op && FindReducerInRoute(master_op, ops_set, GetProducersInSet) &&
          GetConsumersInSet(master_op, ops_set).size() == 0) {
        e_op = master_op;
      }
    } else if (!r_op) {
      r_op = master_op;
    }
  }

  // try to find reducer with different shape.
  for (auto output_op : group->output_ops) {
    if (CompatibleInfo::OpKind(*output_op) == framework::kReduction) {
      if (isl_ast_expr_op_and_then) {
        virtual_consumers[output_op] = e_op;
      }
      continue;
    }
    if (FindNearestReducer(output_op, ops_set)) {
      continue;
    }

    bool found = false;
    std::unordered_set<::pir::Operation*> visited;
    std::queue<::pir::Operation*> candidates;

    candidates.push(output_op);
    visited.insert(output_op);
    // from producers find reducer consumer.
    while (!found && !candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto producer : GetProducersInSet(candidate, ops_set)) {
        if (visited.count(producer)) {
          continue;
        }

        auto reducer = FindReducerInRoute(producer, ops_set, GetConsumersInSet);
        if (reducer) {
          virtual_consumers[output_op] = reducer;
          found = true;
          break;
        }
        candidates.push(producer);
        visited.insert(producer);
      }
    }

    auto output_shape = CompatibleInfo::ValueShape(output_op->result(0));
    if (!found && output_op != e_op && e_op) {
      auto e_output_shape = CompatibleInfo::ValueShape(e_op->result(0));
      if (CompatibleInfo::ShapeProduct(output_shape) ==
          CompatibleInfo::ShapeProduct(e_output_shape)) {
        virtual_consumers[output_op] = e_op;
        found = true;
      }
    }
    if (!found && r_op) {
      auto r_input_shape = CompatibleInfo::ValueShape(r_op->operand_source(0));
      if (CompatibleInfo::ShapeProduct(output_shape) ==
          CompatibleInfo::ShapeProduct(r_input_shape)) {
        virtual_consumers[output_op] = r_op;
        found = true;
      }
    }
  }
  // Establish virtual consumer relationships between output nodes with the same
  // shape. This allows the calculation of output nodes without affiliation to
  // be placed under the same loop.
  std::unordered_map<int, ::pir::Operation*> numel_consumers;
  for (auto out_op : group->output_ops) {
    if (virtual_consumers.find(out_op) != virtual_consumers.end() ||
        !GetConsumersInSet(out_op, ops_set).empty()) {
      continue;
    }
    auto shape = CompatibleInfo::ValueShape(out_op->result(0));
    int numel = CompatibleInfo::ShapeProduct(shape);
    if (numel_consumers.find(numel) == numel_consumers.end()) {
      numel_consumers.insert(std::make_pair(numel, out_op));
    } else {
      virtual_consumers[out_op] = numel_consumers[numel];
    }
  }
  return virtual_consumers;
}

std::vector<::pir::Operation*> BFSTopologicalOrderWithPriority(
    const GroupPtr& group,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers) {
  struct OpWithPriority {
    ::pir::Operation* op;
    int priority;
  };

  struct Comparator {
    bool operator()(const OpWithPriority& lhs, const OpWithPriority& rhs) {
      return lhs.priority > rhs.priority;
    }
  };

  std::vector<::pir::Operation*> ops_in_order;
  std::unordered_set<::pir::Operation*> visited;
  std::unordered_set<::pir::Operation*> ops_set(group->ops.begin(),
                                                group->ops.end());
  std::unordered_map<::pir::Operation*, int> degree_map;
  std::priority_queue<OpWithPriority, std::vector<OpWithPriority>, Comparator>
      priority_candidates;
  std::vector<int> visited_numel;

  // Calculate the priority of a node.
  // The smaller the value, the higher the priority.
  // Prioritize the same shape before considering OpPattern
  auto PriorityFunc = [&visited_numel](::pir::Operation* op) -> int {
    auto op_shape = CompatibleInfo::ValueShape(op->result(0));
    int numel = CompatibleInfo::ShapeProduct(op_shape);
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
    return index * 10 + static_cast<int>(CompatibleInfo::OpKind(*op));
  };

  for (::pir::Operation* op : ops_set) {
    auto consumers = FindConsumers(op, ops_set, virtual_consumers);
    // Some nodes may have multiple edges between them, resulting in duplicates
    // in the consumer. We only need to calculate once.
    std::unordered_set<::pir::Operation*> consumers_without_duplicate(
        consumers.begin(), consumers.end());
    degree_map[op] = consumers_without_duplicate.size();
    if (degree_map.at(op) == 0) {
      priority_candidates.push(OpWithPriority{op, PriorityFunc(op)});
    }
  }

  // Nested BFS, outer layer traverses priority, inner layer performs BFS on
  // current priority.
  while (!priority_candidates.empty()) {
    ::pir::Operation* cur_priority_op = priority_candidates.top().op;
    priority_candidates.pop();

    std::queue<::pir::Operation*> bfs_queue;
    bfs_queue.push(cur_priority_op);
    visited.insert(cur_priority_op);
    while (!bfs_queue.empty()) {
      ::pir::Operation* cur = bfs_queue.front();
      bfs_queue.pop();

      ops_in_order.push_back(cur);
      auto producers = FindProducers(cur, ops_set, virtual_consumers);
      std::unordered_set<::pir::Operation*> producers_without_duplicate(
          producers.begin(), producers.end());
      for (::pir::Operation* op : producers_without_duplicate) {
        --degree_map[op];
        // Ensure that each node is accessed only once and maintain topological
        // order.
        if (visited.count(op) != 0 || degree_map[op] != 0) {
          continue;
        }
        // Perform BFS access to the current priority producers
        int op_priority = PriorityFunc(op);
        if (op_priority <= PriorityFunc(cur_priority_op)) {
          bfs_queue.push(op);
          visited.insert(op);
        } else {
          priority_candidates.push(OpWithPriority{op, op_priority});
        }
      }
    }
  }
  return ops_in_order;
}

std::unordered_set<::pir::Operation*> GetMasters(
    ::pir::Operation* op,
    PrettyNamer* pretty_name,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set) {
  // find consumer
  std::unordered_set<::pir::Operation*> visited;
  std::queue<::pir::Operation*> candidates;
  candidates.push(op);
  std::unordered_set<::pir::Operation*> masters;

  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    auto consumers = GetConsumersInSet(candidate, ops_set);
    for (auto consumer : consumers) {
      if (visited.count(consumer)) {
        continue;
      }
      if (ops_inline.count(consumer)) {
        candidates.push(consumer);
        visited.insert(consumer);
      } else {
        masters.insert(consumer);
      }
    }
  }

  return masters;
}

bool IsConstOp(const ::pir::Operation* op) {
  static std::unordered_set<std::string> const_op_type = {
      "const_scalar", "fill_constant", "arange"};
  return const_op_type.count(CompatibleInfo::OpName(*op));
}

bool CanbeInline(::pir::Operation* op,
                 ::pir::Operation* reducer,
                 PrettyNamer* pretty_name,
                 const std::vector<::pir::Operation*> consumers,
                 const std::unordered_set<::pir::Operation*> masters,
                 const GroupPtr& group,
                 const std::unordered_set<::pir::Operation*>& ops_set) {
  if (group->output_ops.count(op)) {
    return false;
  }
  for (auto consumer : consumers) {
    if (CompatibleInfo::OpKind(*consumer) == framework::kReduction) {
      return false;
    }
  }

  if (IsConstOp(op)) {
    return true;
  }
  if (CompatibleInfo::OpKind(*op) == framework::kReduction) {
    return false;
  }

  if (consumers.size() == 1) {
    return true;
  }
  auto op_shape = CompatibleInfo::ValueShape(op->result(0));
  if (reducer) {
    // node is before reducer and node is not after reduce.
    if (FindReducerInRoute(op, ops_set, GetConsumersInSet) &&
        !FindReducerInRoute(op, ops_set, GetProducersInSet)) {
      auto input_shape = CompatibleInfo::ValueShape(reducer->result(0));
      // check with same shape with reducer input.
      if (CompatibleInfo::ShapeProduct(op_shape) !=
          CompatibleInfo::ShapeProduct(input_shape)) {
        return true;
      }
    }

    return false;
  } else {
    auto op_shape_size = CompatibleInfo::ShapeProduct(op_shape);
    for (auto master : masters) {
      auto master_shape = CompatibleInfo::ValueShape(master->result(0));
      auto master_size = CompatibleInfo::ShapeProduct(master_shape);
      if (op_shape_size != master_size) {
        return true;
      }
    }

    return false;
  }
}

::pir::Operation* GetMasterToComputeAt(
    ::pir::Operation* op,
    PrettyNamer* pretty_name,
    const std::vector<::pir::Operation*>& ops_in_order,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers) {
  // if node is reduction, try find horizontal to compute at.
  if (CompatibleInfo::OpKind(*op) == framework::kReduction) {
    // find all reduce node has done schedule.
    std::unordered_set<::pir::Operation*> done_schedule;
    for (auto tmp : ops_in_order) {
      if (tmp == op) {
        break;
      }
      if (CompatibleInfo::OpKind(*tmp) == framework::kReduction) {
        done_schedule.insert(tmp);
      }
    }
    // remove all consumer reducer node of node from done_schedule.
    std::unordered_set<::pir::Operation*> visited;
    std::queue<::pir::Operation*> candidates;
    candidates.push(op);

    while (!candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto consumer : GetConsumersInSet(candidate, ops_set)) {
        // remove reduction node from done_schedule.
        if (CompatibleInfo::OpKind(*consumer) == framework::kReduction) {
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
      auto shape = CompatibleInfo::ValueShape(op->operand_source(0));
      for (auto r_op : done_schedule) {
        auto rshape = CompatibleInfo::ValueShape(r_op->operand_source(0));
        if (shape == rshape) {
          return r_op;
        }
      }
      return *done_schedule.begin();
    }
  }

  // collect all consumers.
  std::unordered_set<::pir::Operation*> visited, masters;
  std::queue<::pir::Operation*> candidates;
  candidates.push(op);

  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    auto consumers = FindConsumers(candidate, ops_set, virtual_consumers);
    for (auto consumer : consumers) {
      if (visited.count(consumer)) {
        continue;
      }
      if (ops_inline.count(consumer)) {
        candidates.push(consumer);
        visited.insert(consumer);
      } else {
        masters.insert(consumer);
      }
    }
  }

  // nodes-in-order
  for (int idx = 0; idx < ops_in_order.size(); ++idx) {
    if (ops_in_order[idx] == op) {
      for (int idy = idx - 1; idy >= 0; --idy) {
        if (masters.count(ops_in_order[idy])) {
          return ops_in_order[idy];
        }
      }
      break;
    }
  }
  return nullptr;
}

void LoopOrderAssignReduce(ir::IRSchedule& ir_sch,  // NOLINT
                           const std::string& block_name,
                           const std::vector<int>& axes,
                           const cinn::common::Target& target,
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
      PADDLE_ENFORCE_GT(idx,
                        1,
                        phi::errors::InvalidArgument(
                            "Error! Can't find suitable split factor!"));
    }
  }

  // fuse index - 1 times
  for (int idx = 0; idx < index - 1; ++idx) {
    ir_sch.Fuse(block_name, {0, 1});
  }
}

void LoopAssignReduceWithLast(ir::IRSchedule& ir_sch,  // NOLINT
                              const std::string& block_name,
                              const std::vector<int>& inshape,
                              const std::vector<int>& axes,
                              const cinn::common::Target& target) {
  // If the number of current device SM is smaller than the number of SM
  // required by Warp Reduce, the performance of Warp Reduce is better.
  // Otherwise, use Block Reduce.
  auto max_num_threads = cinn::common::DefaultDeviceTarget().max_num_threads();
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
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Error! lane is less equal than max_num_threads, Please check!"));
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
      PADDLE_ENFORCE_GE(idx,
                        max_num_threads / 2,
                        phi::errors::InvalidArgument(
                            "Error! Can't find suitable split factor!"));
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
        PADDLE_ENFORCE_GT(idx,
                          (max_num_threads / 2) / tail,
                          phi::errors::InvalidArgument(
                              "Error! Can't find suitable split factor!"));
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
    // fuse axis before reduce to bind block idx.
    for (int idx = 0; idx < static_cast<int>(inshape.size() - axes.size()) - 1;
         ++idx) {
      ir_sch.Fuse(block_name, {0, 1});
    }
  }
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

void LoopAssignReduceWithoutLast(ir::IRSchedule& ir_sch,  // NOLINT
                                 const std::string& block_name,
                                 const std::vector<int>& inshape,
                                 const std::vector<int>& axes,
                                 const cinn::common::Target& target) {
  int tail = 0;
  bool bound = true;
  auto shape = pe::GetFirstStepReduceShape(inshape, axes, bound, tail);
  CHECK(bound) << std::accumulate(inshape.begin(),
                                  inshape.end(),
                                  std::string(""),
                                  [](const std::string& left, const int right) {
                                    return left + std::to_string(right) + " ";
                                  });

  VLOG(4) << "LoopAssignReduceWithoutLast: The input shape=["
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
          // the axes value need left shift
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

std::vector<int> GetReducerDimAttr(::pir::Operation* reduce_op) {
  int rank = reduce_op->operand_source(0)
                 .type()
                 .dyn_cast<::pir::DenseTensorType>()
                 .dims()
                 .size();

  auto attr = reduce_op->attributes().at("axis");
  auto attr_vec = attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> dim;
  for (auto vec_element : attr_vec) {
    auto axis = vec_element.dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += rank;
    }
    dim.push_back(axis);
  }
  return dim;
}

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
  PADDLE_ENFORCE_GT(src.size(),
                    index,
                    phi::errors::InvalidArgument(
                        "Error! src size is less than index, Please check!"));
  PADDLE_ENFORCE_GT(dst.size(),
                    index,
                    phi::errors::InvalidArgument(
                        "Error! dst size is less than index, Please check!"));

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

void MergeReduceToReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* master,
    PrettyNamer* pretty_name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  VLOG(3) << "start to MergeReduceToReduce...";
  auto op_out_name =
      pretty_name->GetOrNew(op->result(0), CompatibleInfo::kNamePrefix);
  auto master_out_name =
      pretty_name->GetOrNew(master->result(0), CompatibleInfo::kNamePrefix);
  auto shape = CompatibleInfo::ValueShape(op->operand_source(0));

  std::vector<int> axes = GetReducerDimAttr(master);
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }
  if (WithoutLastDimInReduce(shape, axes)) {
    auto mshape = CompatibleInfo::ValueShape(master->operand_source(0));
    if (tmp_tensor_info.count(op_out_name + "_1")) {
      if (shape == mshape) {
        // second step reduce
        {
          auto block = ir_sch.GetBlock(op_out_name);
          auto loops = ir_sch.GetLoops(master_out_name);
          ir_sch.SimpleComputeAt(block, loops.back());
          // reduce init
          {
            auto block = ir_sch.GetBlock(op_out_name + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_out_name + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
        // first step reduce
        {
          auto n_tensor = tmp_tensor_info.at(op_out_name + "_0");
          auto m_tensor = tmp_tensor_info.at(master_out_name + "_0");

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
        auto n_tensor = tmp_tensor_info.at(op_out_name + "_0");
        auto m_tensor = tmp_tensor_info.at(master_out_name + "_0");
        if (n_tensor->shape == m_tensor->shape) {
          // second step reduce
          {
            auto block = ir_sch.GetBlock(op_out_name);
            auto loops = ir_sch.GetLoops(master_out_name);
            ir_sch.SimpleComputeAt(block, loops.back());
            // reduce init
            {
              auto block = ir_sch.GetBlock(op_out_name + "__reduce_init");
              auto loops = ir_sch.GetLoops(master_out_name + "__reduce_init");
              ir_sch.SimpleComputeAt(block, loops.back());
            }
          }
          // first step reduce
          {
            auto n_tensor = tmp_tensor_info.at(op_out_name + "_0");
            auto m_tensor = tmp_tensor_info.at(master_out_name + "_0");

            auto n_loops = ir_sch.GetLoops(n_tensor->name + "__reduce_init");
            auto m_loops = ir_sch.GetLoops(m_tensor->name + "__reduce_init");

            PADDLE_ENFORCE_EQ(n_loops.size(),
                              m_loops.size(),
                              phi::errors::InvalidArgument(
                                  "Error! n_loops size is not equal to m_loops "
                                  "size, Please check!"));
            MergeLoops(ir_sch.GetModule().GetExprs().at(0),
                       n_loops,
                       m_loops,
                       n_loops.size() - 1);
          }
        } else {
          PADDLE_THROW(
              phi::errors::InvalidArgument("not support this type fusion!"));
        }
      }
    } else {
      if (shape == mshape) {
        // reduce loop
        {
          auto block = ir_sch.GetBlock(op_out_name);
          auto loops = ir_sch.GetLoops(master_out_name);
          ir_sch.SimpleComputeAt(block, loops.back());
          // reduce init
          {
            auto block = ir_sch.GetBlock(op_out_name + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_out_name + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
      } else {
        // reduce loop
        {
          auto block = ir_sch.GetBlock(op_out_name);
          auto nloops = ir_sch.GetLoops(op_out_name);
          auto mloops = ir_sch.GetLoops(master_out_name);
          for (int idx = 0; idx < mloops.size(); ++idx) {
            if (GetLoopExtent(nloops[idx]) != GetLoopExtent(mloops[idx])) {
              ir_sch.SimpleComputeAt(block, mloops[idx - 1]);
              break;
            }
          }
          // reduce init
          {
            auto block = ir_sch.GetBlock(op_out_name + "__reduce_init");
            auto loops = ir_sch.GetLoops(master_out_name + "__reduce_init");
            ir_sch.SimpleComputeAt(block, loops.back());
          }
        }
      }
    }
  } else {
    if (tmp_tensor_info.count(op_out_name + "_1")) {
      // identity
      {
        auto block = ir_sch.GetBlock(op_out_name);
        auto loops = ir_sch.GetLoops(master_out_name);
        ir_sch.SimpleComputeAt(block, loops.back());
      }
      // reduce
      {
        auto n_tensor = tmp_tensor_info.at(op_out_name + "_1");
        auto m_tensor = tmp_tensor_info.at(master_out_name + "_1");

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
        auto n_tensor = tmp_tensor_info.at(op_out_name + "_0");
        auto m_tensor = tmp_tensor_info.at(master_out_name + "_0");

        auto n_block = ir_sch.GetBlock(n_tensor->name);
        auto m_block = ir_sch.GetBlock(m_tensor->name);

        auto n_loops = ir_sch.GetLoops(n_tensor->name);
        auto m_loops = ir_sch.GetLoops(m_tensor->name);
        PADDLE_ENFORCE_EQ(
            n_loops.size(),
            m_loops.size(),
            phi::errors::InvalidArgument(
                "Error! n_loops size is not equal to m_loops size, "
                "Please check!"));

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
    } else if (tmp_tensor_info.count(op_out_name + "_0")) {
      // identity
      {
        auto block = ir_sch.GetBlock(op_out_name);
        auto loops = ir_sch.GetLoops(master_out_name);
        ir_sch.SimpleComputeAt(block, loops.back());
      }
      // shuffle reduce
      {
        auto n_tensor = tmp_tensor_info.at(op_out_name + "_0");
        auto m_tensor = tmp_tensor_info.at(master_out_name + "_0");

        auto block = ir_sch.GetBlock(n_tensor->name);
        auto loops = ir_sch.GetLoops(m_tensor->name);
        ir_sch.SimpleComputeAt(block, loops.back());
      }
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Error! Unkown Reduce Type, Please Check!"));
    }
  }
}

void InsertSyncThread(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    PrettyNamer* pretty_name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  auto shape = CompatibleInfo::ValueShape(op->operand_source(0));
  auto axes = GetReducerDimAttr(op);
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }
  if (!WithoutLastDimInReduce(shape, axes)) {
    return;
  }

  auto op_out_name =
      pretty_name->GetOrNew(op->result(0), CompatibleInfo::kNamePrefix);
  std::string post = "";
  for (int idx = 0;; ++idx) {
    if (!tmp_tensor_info.count(op_out_name + post)) {
      break;
    }
    auto tensor = tmp_tensor_info.at(op_out_name + post);
    if (!ir_sch.HasBlock(tensor->name)) {
      break;
    }

    post = "_" + std::to_string(idx);
    if (idx > 0) {
      // insert syncthreads.
      auto loops = ir_sch.GetLoops(op_out_name);
      ir_sch.SyncThreads(loops[loops.size() - 2], false);
      return;
    }
  }
}

void MergeReduceLoop(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* master,
    PrettyNamer* pretty_name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  VLOG(3) << "start to MergeReduceLoop...";
  if (CompatibleInfo::OpKind(*master) == kReduction && op != master) {
    MergeReduceToReduce(
        ir_sch, op, master, pretty_name, tensor_map, tmp_tensor_info);
    return;
  }

  auto op_out_name =
      pretty_name->GetOrNew(op->result(0), CompatibleInfo::kNamePrefix);
  auto master_out_name =
      pretty_name->GetOrNew(master->result(0), CompatibleInfo::kNamePrefix);
  int min_index_loop = INT_MAX;
  std::string post_ = "", post__ = "_0";
  for (int idx = 0;; ++idx) {
    if (!tmp_tensor_info.count(op_out_name + post__)) {
      break;
    }
    auto tensor_ = tmp_tensor_info.at(op_out_name + post_);
    auto tensor__ = tmp_tensor_info.at(op_out_name + post__);
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
  InsertSyncThread(ir_sch, op, pretty_name, tensor_map, tmp_tensor_info);

  if (op == master) return;
  auto node_loops = ir_sch.GetLoops(op_out_name);
  auto master_loops = ir_sch.GetLoops(master_out_name);

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
      auto block = ir_sch.GetBlock(op_out_name);
      auto loops = ir_sch.GetLoops(master_out_name);
      ir_sch.SimpleComputeAt(block, loops.back());

      if (ir_sch.HasBlock(op_out_name + "__reduce_init")) {
        auto block = ir_sch.GetBlock(op_out_name + "__reduce_init");
        auto loops = ir_sch.GetLoops(master_out_name);
        ir_sch.SimpleComputeAt(block, loops.back());
      }
    }

    break;
  } while (--index >= 0);
}

void LoopComputeAt(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* master,
    PrettyNamer* pretty_name,
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  auto op_out_name =
      pretty_name->GetOrNew(op->result(0), CompatibleInfo::kNamePrefix);
  if (!group->output_ops.count(op)) {
    auto block = ir_sch.GetBlock(op_out_name);
    ir_sch.SetBuffer(block, "local");
  }

  if (CompatibleInfo::OpKind(*op) == framework::kReduction) {
    MergeReduceLoop(
        ir_sch, op, master, pretty_name, tensor_map, tmp_tensor_info);
    return;
  }

  if (op == master) return;
  auto master_out = master->result(0);
  auto master_out_name =
      pretty_name->GetOrNew(master_out, CompatibleInfo::kNamePrefix);

  auto node_loops = ir_sch.GetLoops(op_out_name);
  auto master_loops = ir_sch.GetLoops(master_out_name);

  if (CompatibleInfo::OpKind(*master) == framework::kReduction) {
    // find real master loops.
    std::string prefix = "", post = "";
    for (int idx = 0;; ++idx) {
      if (!tmp_tensor_info.count(master_out_name + post)) {
        break;
      }
      auto tensor = tmp_tensor_info.at(master_out_name + post);
      if (!ir_sch.HasBlock(tensor->name)) {
        break;
      }

      prefix = post;
      post = "_" + std::to_string(idx);
    }
    auto tensor = tmp_tensor_info.at(master_out_name + prefix);
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

void LoopAssignReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* reducer,
    PrettyNamer* pretty_name,
    const Target& target,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  // if node is reducer, return.
  if (CompatibleInfo::OpKind(*op) == framework::kReduction) {
    return;
  }
  ::pir::Value op_data = op->result(0);
  ::pir::Value reducer_data = reducer->result(0);
  std::string op_data_name =
      pretty_name->GetOrNew(op_data, CompatibleInfo::kNamePrefix);
  std::string reducer_data_name =
      pretty_name->GetOrNew(reducer_data, CompatibleInfo::kNamePrefix);

  // flatten loops.
  auto loops = ir_sch.GetLoops(op_data_name);
  // do loop flatten.
  if (CompatibleInfo::OpKind(*op) == framework::kElementWise) {
    ir_sch.FlattenLoops(loops, true);
  } else {
    ir_sch.FlattenLoops(loops, false);
  }
  std::vector<int> shape =
      CompatibleInfo::ValueShape(reducer->operand_source(0));
  auto axes = GetReducerDimAttr(reducer);
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
  std::vector<int> op_shape = CompatibleInfo::ValueShape(op_data);
  // The output shape of node is different from that of reduce node
  if (CompatibleInfo::ShapeProduct(shape) !=
      CompatibleInfo::ShapeProduct(op_shape)) {
    // get loop factors of reduce node
    int extend = 1;
    std::vector<int> factors;
    loops = ir_sch.GetLoops(op_data_name);
    auto rloops = ir_sch.GetLoops(reducer_data_name);

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
    const std::string& reduce_block_name = reducer_data_name;
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
      loops = ir_sch.GetLoops(op_data_name);
    }
    if (!second_step_factors.empty()) {
      ir_sch.Split(loops.at(second_start_loop_index), second_step_factors);
      loops = ir_sch.GetLoops(op_data_name);
    }

    // copy loop info form rloops.
    copy_loop_info(loops, rloops);
    return;
  }
  // node output is same shape with reduce input.
  if (WithoutLastDimInReduce(shape, axes)) {
    // if using two strep reduce.
    if (tmp_tensor_info.count(reducer_data_name + "_1")) {
      VLOG(4) << "Try assign loop of " << op_data_name
              << " into two strep reduce loop of " << reducer_data_name;
      LoopAssignReduceWithoutLast(ir_sch, op_data_name, shape, axes, target);
      auto nloops = ir_sch.GetLoops(op_data_name);
      auto rloops =
          ir_sch.GetLoops(tmp_tensor_info.at(reducer_data_name + "_0")->name);

      VLOG(4) << op_data_name << "'s loop level is " << nloops.size()
              << ", and " << reducer_data_name << "'s loop level is "
              << rloops.size();
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(op_data_name);
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else {
      VLOG(4) << "Try assign loop of " << op_data_name
              << " into reduce loop of " << reducer_data_name;

      auto nloops = ir_sch.GetLoops(op_data_name);
      ir_sch.Split(nloops.back(), shape);
      LoopOrderAssignReduce(ir_sch, op_data_name, axes, target);
      nloops = ir_sch.GetLoops(op_data_name);
      auto rloops =
          ir_sch.GetLoops(tensor_map.find(reducer_data)->second->name);
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(op_data_name);
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    }
  } else {
    if (tmp_tensor_info.count(reducer_data_name + "_1")) {
      {
        auto nloops = ir_sch.GetLoops(op_data_name);
        ir_sch.Split(nloops.back(), shape);
      }
      LoopAssignReduceWithLast(ir_sch, op_data_name, shape, axes, target);

      auto nloops = ir_sch.GetLoops(op_data_name);
      auto rloops =
          ir_sch.GetLoops(tmp_tensor_info.at(reducer_data_name + "_1")->name);
      if (nloops.size() < rloops.size()) {
        ir_sch.Split(nloops[0], {1, -1});
      }

      nloops = ir_sch.GetLoops(op_data_name);
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else if (tmp_tensor_info.count(reducer_data_name + "_0")) {
      auto tensor = tmp_tensor_info.at(reducer_data_name + "_0");
      auto rloops = ir_sch.GetLoops(tensor->name);
      std::vector<int> factors;
      for (auto& loop : rloops) {
        // FIXME(Aurelius84): Need add broadcast_to Op
        int factor = loop.As<ir::For>()->extent.as_int32();
        if (factor == 1) {
          factor = -1;
        }
        factors.push_back(factor);
      }
      auto nloops = ir_sch.GetLoops(op_data_name);
      ir_sch.Split(nloops.back(), factors);

      nloops = ir_sch.GetLoops(op_data_name);
      // copy loop info form rloops.
      copy_loop_info(nloops, rloops);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument("Error! Unkown Reduce Type!"));
    }
  }
}

std::unordered_map<std::string, ::pir::Value> GetOutValueSet(
    PrettyNamer* pretty_name,
    const std::unordered_set<::pir::Operation*>& ops_set) {
  std::unordered_map<std::string, ::pir::Value> out_value_set;
  for (auto* op : ops_set) {
    out_value_set[pretty_name->GetOrNew(
        op->result(0), CompatibleInfo::kNamePrefix)] = op->result(0);
  }
  return out_value_set;
}

void SyncThreadWithShared(
    ir::IRSchedule& ir_sch,  // NOLINT
    const GroupPtr& group,
    PrettyNamer* pretty_name,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  auto exprs_inorder = ir_sch.GetAllBlocks();
  auto op_out_set = GetOutValueSet(pretty_name, ops_set);

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

    if (!op_out_set.count(block->name)) {
      continue;
    }
    auto op_data = op_out_set.find(block->name)->second;
    auto* op = op_data.defining_op();
    auto op_shape = CompatibleInfo::ValueShape(op_data);

    auto masters = GetMasters(op, pretty_name, ops_inline, ops_set);
    if (masters.empty()) {
      continue;
    }

    bool do_set_buffer_to_shared = false;
    for (auto master : masters) {
      auto master_data = master->result(0);
      auto master_shape = CompatibleInfo::ValueShape(master_data);
      if (CompatibleInfo::OpKind(*master) == framework::kReduction) {
        master_shape = CompatibleInfo::ValueShape(master->operand_source(0));
      }

      auto op_shape_size = CompatibleInfo::ShapeProduct(op_shape);
      auto master_shape_size = CompatibleInfo::ShapeProduct(master_shape);
      std::string master_data_name =
          pretty_name->GetOrNew(master_data, CompatibleInfo::kNamePrefix);
      if (op_shape_size != master_shape_size) {
        if (check_sync_mark(idx, master_data_name)) {
          auto loops = ir_sch.GetLoops(master_data_name);
          ir_sch.SyncThreads(loops.back(), false);
          sync_mark.insert(master_data_name);
        }
        do_set_buffer_to_shared = true;
      }
    }
    if (do_set_buffer_to_shared &&
        group->output_ops.find(op) == group->output_ops.end()) {
      auto block = ir_sch.GetBlock(
          pretty_name->GetOrNew(op_data, CompatibleInfo::kNamePrefix));
      ir_sch.SetBuffer(block, "shared");
    }
  }
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
