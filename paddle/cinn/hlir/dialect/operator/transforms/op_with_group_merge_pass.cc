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

#include "paddle/cinn/hlir/dialect/operator/transforms/op_with_group_merge_pass.h"

#include <limits.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"

namespace cinn {
namespace dialect {
namespace ir {

std::unordered_map<std::string, OpPatternKind> OpKindMap = {
    {"pd_op.add", OpPatternKind::kElementWise},
    {"pd_op.subtract", OpPatternKind::kElementWise},
    {"pd_op.multiply", OpPatternKind::kElementWise},
    {"pd_op.divide", OpPatternKind::kElementWise},
    {"pd_op.sqrt", OpPatternKind::kElementWise},
    {"pd_op.full", OpPatternKind::kElementWise},
    {"pd_op.relu", OpPatternKind::kElementWise},
    {"pd_op.exp", OpPatternKind::kElementWise},
    {"pd_op.sin", OpPatternKind::kElementWise},
    {"pd_op.cos", OpPatternKind::kElementWise},
    {"pd_op.sum", OpPatternKind::kReduction},
    {"cinn_op.reduce_sum", OpPatternKind::kReduction},
    {"cinn_op.reduce_max", OpPatternKind::kReduction},
    {"cinn_op.broadcast", OpPatternKind::kBroadcast},
};

OpPatternKind GetOpKind(const std::string& op_name) {
  auto found_it = OpKindMap.find(op_name);
  if (found_it == OpKindMap.end()) {
    throw std::runtime_error("not support op yet in op kind map");
  }

  return found_it->second;
}

phi::DDim GetFirstInputShape(const ::pir::Operation* op) {
  auto in = op->operand_source(0);

  return in.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

phi::DDim GetValueShape(const ::pir::Value value) {
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

bool WithoutLastDimInReduce(const std::vector<int64_t>& inshape,
                            const std::vector<int64_t>& axes) {
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int64_t sum_last_axes = 1;
  for (size_t idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    sum_last_axes *= inshape[idx];
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

int GetSharedSize(::pir::Operation* node) {
  auto inshape = phi::vectorize<int64_t>(GetValueShape(node->result(0)));

  auto axes = GetVectorAttr(node, "axis");

  if (WithoutLastDimInReduce(inshape, axes)) {
    int lane = 1;
    for (size_t idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      lane = inshape[idx];
    }
    // int max_num_threads = common::DefaultNVGPUTarget().max_num_threads();
    // todo(phlrain): get gpu max threads
    int max_num_threads = 2048;
    if (lane > max_num_threads / 2) {
      return 0;
    }
    int index = axes.size() - 1;
    for (; index >= 0; --index) {
      if (static_cast<size_t>(index + 1) < axes.size() &&
          axes[index] != axes[index + 1] - 1) {
        break;
      }
      lane *= inshape[axes[index]];
      if (lane > max_num_threads / 2) {
        break;
      }
    }
    // if lane > (max_num_threads / 2),the loop break from lane >
    // max_num_threads / 2.
    int axis = lane > (max_num_threads / 2) ? axes[index] : axes[index + 1];
    if (lane <= max_num_threads) {
      return lane * sizeof(float);
    } else {
      int prefix = inshape[axis];
      int tail = lane / prefix;
      for (int idx = max_num_threads / tail;
           idx > ((max_num_threads / 2) / tail);
           --idx) {
        if (prefix % idx == 0) {
          return idx * tail * sizeof(float);
        }
      }
      int num = max_num_threads / tail;
      return num * tail * sizeof(float);
    }
  }
  return 0;
}

using ConditionFunction =
    std::function<bool(::pir::Operation*, const GroupPtr&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper {
 public:
  explicit OpFusionPassHelper(const std::vector<pir::Operation*>& op_list) {
    // init fusion relation
    InitFusionRelation();
    // filter node data, create group for each node
    // auto nodes_inorder = std::get<0>(graph->topological_order());

    for (auto it = op_list.begin(); it != op_list.end(); ++it) {
      local_ops_.insert(*it);
    }

    int index = 0;
    for (auto it = op_list.begin(); it != op_list.end(); ++it) {
      auto node = *it;
      if (node) {
        nodes_.push_back(node);
        auto group = std::make_shared<Group>();
        // init group
        group->nodes.push_back(node);
        group->nodes_set.insert(node);
        group->output_nodes.insert(node);
        // input node

        for (size_t i = 0; i < node->num_operands(); ++i) {
          auto input =
              node->operand_source(i).dyn_cast<pir::OpResult>().owner();
          if (input && (local_ops_.count(input))) {
            group->input_nodes[input] = 1;
          }
        }

        // group type
        group->op_pattern_kind = GetOpKind(node->name());
        // use current node as master node for schedule
        group->master_nodes.insert(node);

        // get opration unique id
        group->group_id = "id_" + std::to_string(index++);
        fusion_groups_[node] = group;
      }
    }
    // reverse node for output to input
    std::reverse(nodes_.begin(), nodes_.end());
  }

  // return a vector of groups in topological order.
  GroupList operator()(bool do_fusion = true) {
    // do op fusion.
    if (do_fusion) {
      DoOpFusion();
    }

    // find all fusion group.
    GroupList fusion_groups;
    std::unordered_set<Group*> groups_set;
    for (auto node : nodes_) {
      auto& group = fusion_groups_[node];
      if (!groups_set.count(group.get())) {
        groups_set.insert(group.get());
        fusion_groups.push_back(group);
        // reverse nodes order to producer->consumer.
        std::reverse(group->nodes.begin(), group->nodes.end());
      }
    }

    // producer consumer
    for (auto& consumer : fusion_groups) {
      for (auto& input_node : consumer->input_nodes) {
        if (!local_ops_.count(input_node.first)) {
          continue;
        }
        auto& producer = fusion_groups_[input_node.first];
        consumer->mut_producer_groups()->insert(producer);
        producer->mut_consumer_groups()->insert(consumer);
      }
    }

    // init group depth.
    for (auto& group : fusion_groups) {
      for (const auto& consumer : group->consumer_groups()) {
        // update depth.
        group->depth = std::max(group->depth, consumer->depth + 1);
      }
    }

    // reverse to keep fusion group in order.
    std::reverse(fusion_groups.begin(), fusion_groups.end());

    return fusion_groups;
  }

 private:
  void DoOpFusion() {
    for (auto consumer : nodes_) {
      auto consumer_kind = GetOpKind(consumer->name());
      // kNonFusible op can't fuse any other op.
      if (consumer_kind == kNonFusible) {
        continue;
      }

      // fusion op for consumer
      auto consumer_fusion = fusion_groups_[consumer];  //
      // check all linkin node
      for (size_t i = 0; i < consumer->num_operands(); ++i) {
        auto producer_data = consumer->operand_source(i);

        auto producer = producer_data.dyn_cast<pir::OpResult>().owner();
        if (!local_ops_.count(producer)) {
          continue;
        }

        // if producer is fused.
        if (consumer_fusion->nodes_set.count(producer)) {
          // VLOG(3) << "Op " << producer->id() << " is fused.";
          continue;
        }
        // if producer data is placeholder
        if (!producer) {
          continue;
        }
        // kNonFusible op can't fuse any other op.
        auto producer_kind = GetOpKind(producer->name());
        if (producer_kind == kNonFusible) {
          continue;
        }
        // VLOG(3) << "Producer Op: " << producer->id()
        //         << ", Op Pattern: " << producer_kind
        //         << " -> Consumer Op: " << consumer->id()
        //         << ", Op Pattern: " << consumer_kind;
        bool can_fuse = true;
        // checkout producer node outputs are all in fusion op

        // find all the op use by
        size_t producer_data_used_num = 0;
        for (auto it = producer_data.use_begin(); it != producer_data.use_end();
             ++it) {
          auto consumer_node = it->owner();
          producer_data_used_num++;
          // if fusion group can't find node, can't merge
          if (consumer_fusion->nodes_set.find(consumer_node) ==
              consumer_fusion->nodes_set.end()) {
            can_fuse = false;
            break;
          }
        }

        if (!can_fuse || !CanFuse(producer, consumer)) continue;
        // VLOG(3) << "Fuse Op " << producer->id() << " into Op "
        //         << consumer->id();

        // fuse producer to fusion group
        // TODO(phrain) : support id
        // consumer_fusion->group_id =
        //     producer->id() + "_" + consumer_fusion->group_id;

        consumer_fusion->group_id = consumer_fusion->group_id;
        consumer_fusion->nodes.push_back(producer);
        consumer_fusion->nodes_set.insert(producer);
        consumer_fusion->input_nodes.erase(producer);
        consumer_fusion->op_pattern_kind =
            static_cast<int>(consumer_fusion->op_pattern_kind) >
                    static_cast<int>(producer_kind)
                ? consumer_fusion->op_pattern_kind
                : producer_kind;

        if (producer_kind == kReduction) {
          consumer_fusion->master_nodes.insert(producer);
        }

        if (output_nodes_set_.count(producer)) {
          // VLOG(3) << "Insert Global Output Node : " << producer->id();
          consumer_fusion->output_nodes.insert(producer);
        } else if (producer_data_used_num > 1 && producer->num_operands() > 0 &&
                   is_same_size(producer, consumer_fusion)) {
          // producer is not a const value node.
          consumer_fusion->internal_nodes.insert(producer);
        }

        // fuse input node

        auto producer_fusion = fusion_groups_[producer];
        for (auto input_node : producer_fusion->input_nodes) {
          if (consumer_fusion->input_nodes.count(input_node.first)) {
            consumer_fusion->input_nodes[input_node.first] += input_node.second;
          } else {
            consumer_fusion->input_nodes.insert(input_node);
          }
        }
        // update node group
        fusion_groups_[producer] = consumer_fusion;
      }
    }
  }

  void InitFusionRelation() {
    // fusion relation.
    // 1.kElementwise as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {kElementWise, kBroadcast, kReduction, kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Elementwise + *Elementwise*). As
          // has same output shape, can always fuse.
          {kElementWise, always_fuse},
          // must be horizontal, as Elementwise + Broadcast is left to fusion
          // merge pass.
          {kBroadcast,
           [](::pir::Operation* producer, const GroupPtr& consumer) -> bool {
             // NOTE, producer and consumer NEVER be same size
             if (is_same_size(producer, consumer)) {
               return true;
             }

             // NOTE, original code is below, if produer is not output node,
             // result always be true
             // !helper->output_nodes_set_.count(producer);
             return true;
           }},
          // horizontal or vertical relation, check with same output shape with
          // horizontal relation or with last
          // successive dimension less than 1024 for gpu.
          {kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or can compute inline.
          {kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {kOutFusible, is_same_shape}};
      fusion_relation_map_[kElementWise] = std::move(relation);
    }
    // 2.kBroadcast as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {kElementWise, kReduction, kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Broadcast + *Elementwise*), check
          // with same output shape.
          {kElementWise, is_same_size},
          // must be horizontal, as Broadcast + Broadcast is not allowed.
          {kBroadcast, is_same_size},
          // horizontal or vertical relation(Broadcast + Reduce).
          {kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or just one consumer.
          {kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {kOutFusible, is_same_shape}};
      fusion_relation_map_[kBroadcast] = std::move(relation);
    }
    // 3.kReduction as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {kElementWise, kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Reduce + Elementwise*), check
          // without last dimension in reduce.
          {kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape and
          // without last dimension in reduce.
          {kBroadcast, reduce_fuse_broadcast},
          // must be horizontal relation and with same reduce attr.
          {kReduction, reduce_fuse_reduce},
          // no_fuse
          {kInjective, no_fuse},
          // can't fuse.
          {kOutFusible, no_fuse}};
      fusion_relation_map_[kReduction] = std::move(relation);
    }
    // 4.kInjective
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {kElementWise, kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // can be horizontal or vertical(Injective + Elementwise), check with
          // same output shape.
          {kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape.
          {kBroadcast, horizontal_with_same_size},
          // left to fusion merge pass.
          {kReduction, no_fuse},
          // must be horizontal relation, check with same output shape.
          {kInjective, horizontal_or_can_inline},
          // can't fuse.
          {kOutFusible, no_fuse},
      };
      fusion_relation_map_[kInjective] = std::move(relation);
    }
    // 5.kOutFusible
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {kElementWise, kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation, check has same shape.
          {kElementWise, is_same_shape},
          // it must be horizontal relation, check has same shape.
          {kBroadcast, is_same_shape},
          // can't fuse.
          {kReduction, no_fuse},
          // must be horizontal relation, check has same shape.
          {kInjective, is_same_shape},
          // can't fuse.
          {kOutFusible, no_fuse},
      };
      fusion_relation_map_[kOutFusible] = std::move(relation);
    }
  }

  bool CanFuse(::pir::Operation* producer, const ::pir::Operation* consumer) {
    auto& relation = fusion_relation_map_[GetOpKind(producer->name())];
    // first step: check producer can be fused into consumer
    if (relation.op_kind.count(GetOpKind(consumer->name()))) {
      auto& consumer_group = fusion_groups_[consumer];
      // second step: check producer can be fused into consumer group
      VLOG(3) << "Call ConditionFunction, Producer Op Pattern : "
              << GetOpKind(producer->name()) << " , Consumer Group Pattern : "
              << consumer_group->op_pattern_kind;
      return relation.fusion_op_kind[consumer_group->op_pattern_kind](
          producer, fusion_groups_[consumer]);
    }

    return false;
  }
  std::vector<::pir::Operation*> nodes_;
  std::unordered_map<const ::pir::Operation*, GroupPtr> fusion_groups_;
  std::unordered_set<const ::pir::Operation*> output_nodes_set_;

  std::vector<std::shared_ptr<Group>> groups_;

  std::unordered_set<const ::pir::Operation*> local_ops_;

  struct FusionRelation {
    // producer -> consumer
    std::unordered_set<OpPatternKind> op_kind = {};
    // producer -> fusion sonsumer
    std::unordered_map<OpPatternKind, ConditionFunction> fusion_op_kind = {};
  };
  std::unordered_map<OpPatternKind, FusionRelation> fusion_relation_map_;
};

GroupList OpFusionPassInternal(const std::vector<pir::Operation*>& op_list) {
  VLOG(3) << "OpFusionPass...!";
  auto op_fusion_helper = OpFusionPassHelper(op_list);
  auto res = op_fusion_helper();

  for (size_t i = 0; i < res.size(); ++i) {
    auto group = res[i];

    for (size_t j = 0; j < group->nodes.size(); ++j) {
    }
  }
  VLOG(3) << "OpFusionPass Finish...!";

  return res;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
