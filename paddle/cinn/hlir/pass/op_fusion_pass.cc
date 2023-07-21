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

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/pass/op_fusion_pass_util.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;

using GroupPtr = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using ConditionFunction =
    std::function<bool(const FusionHelperBase*, const Node*, const GroupPtr&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper : public FusionHelperBase {
 public:
  explicit OpFusionPassHelper(const Graph* graph) : FusionHelperBase(graph) {
    // init fusion relation
    InitFusionRelation();
    // filter node data, create group for each node
    auto nodes_inorder = std::get<0>(graph->topological_order());
    for (auto graph_node : nodes_inorder) {
      auto node = graph_node->safe_as<Node>();
      if (node) {
        nodes_.push_back(node);
        auto group = std::make_shared<Graph::Group>(graph);
        // init group
        group->nodes.push_back(node);
        group->nodes_set.insert(node);
        group->output_nodes.insert(node);
        // input node
        for (auto& edge : node->inlinks()) {
          auto input_graph_node = edge->source();
          auto input_node_data = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          // input data has no source node
          if (input_node_data->source_node.get()) {
            group->input_nodes[input_node_data->source_node.get()] = 1;
          }
        }

        // group type
        group->op_pattern_kind = GetOpKind(node);
        // use current node as master node for schedule
        group->master_nodes.insert(node);
        group->group_id = node->id();
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
    std::unordered_set<Graph::Group*> groups_set;
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
      auto consumer_kind = GetOpKind(consumer);
      // kNonFusible op can't fuse any other op.
      if (consumer_kind == framework::kNonFusible) {
        continue;
      }

      // fusion op for consumer
      auto consumer_fusion = fusion_groups_[consumer];  //
      // check all linkin node
      for (auto& edge : consumer->inlinks()) {
        auto graph_node = edge->source();
        auto producer_data = graph_node->safe_as<NodeData>();
        CHECK(producer_data);

        auto producer = producer_data->source_node.get();
        // if producer is fused.
        if (consumer_fusion->nodes_set.count(producer)) {
          VLOG(3) << "Op " << producer->id() << " is fused.";
          continue;
        }
        // if producer data is placeholder
        if (!producer) {
          continue;
        }

        // kNonFusible op can't fuse any other op.
        auto producer_kind = GetOpKind(producer);
        if (producer_kind == framework::kNonFusible) {
          continue;
        }
        VLOG(3) << "Producer Op: " << producer->id()
                << ", Op Pattern: " << producer_kind
                << " -> Consumer Op: " << consumer->id()
                << ", Op Pattern: " << consumer_kind;
        bool can_fuse = true;
        // checkout producer node outputs are all in fusion op
        for (auto& link : producer_data->outlinks()) {
          auto consumer_node = link->sink()->safe_as<Node>();
          CHECK(consumer_node);
          // if fusion group can't find node, can't merge
          if (consumer_fusion->nodes_set.find(consumer_node) ==
              consumer_fusion->nodes_set.end()) {
            can_fuse = false;
            break;
          }
        }

        if (!can_fuse || !CanFuse(producer, consumer)) continue;
        VLOG(3) << "Fuse Op " << producer->id() << " into Op "
                << consumer->id();

        // fuse producer to fusion group
        consumer_fusion->group_id =
            producer->id() + "_" + consumer_fusion->group_id;
        consumer_fusion->nodes.push_back(producer);
        consumer_fusion->nodes_set.insert(producer);
        consumer_fusion->input_nodes.erase(producer);
        consumer_fusion->op_pattern_kind =
            static_cast<int>(consumer_fusion->op_pattern_kind) >
                    static_cast<int>(producer_kind)
                ? consumer_fusion->op_pattern_kind
                : producer_kind;

        if (producer_kind == framework::kReduction) {
          consumer_fusion->master_nodes.insert(producer);
        }

        if (this->output_nodes_set_.count(producer)) {
          VLOG(3) << "Insert Global Output Node : " << producer->id();
          consumer_fusion->output_nodes.insert(producer);
        } else if (producer_data->outlinks().size() > 1 &&
                   producer->inlinks().size() > 0 &&
                   is_same_size(this, producer, consumer_fusion)) {
          // producer is not a const value node.
          consumer_fusion->internal_nodes.insert(producer);
        }

        // fuse input node
        auto& producer_fusion = fusion_groups_[producer];
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
      relation.op_kind = {framework::kElementWise,
                          framework::kBroadcast,
                          framework::kReduction,
                          framework::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Elementwise + *Elementwise*). As
          // has same output shape, can always fuse.
          {framework::kElementWise, always_fuse},
          // must be horizontal, as Elementwise + Broadcast is left to fusion
          // merge pass.
          {framework::kBroadcast,
           [](const FusionHelperBase* helper,
              const Node* producer,
              const GroupPtr& consumer) -> bool {
             if (is_same_size(helper, producer, consumer)) {
               return true;
             }
             return !helper->output_nodes_set_.count(producer);
           }},
          // horizontal or vertical relation, check with same output shape with
          // horizontal relation or with last
          // successive dimension less than 1024 for gpu.
          {framework::kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or can compute inline.
          {framework::kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {framework::kOutFusible, is_same_shape}};
      fusion_relation_map_[framework::kElementWise] = std::move(relation);
    }
    // 2.kBroadcast as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElementWise,
                          framework::kReduction,
                          framework::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Broadcast + *Elementwise*), check
          // with same output shape.
          {framework::kElementWise, is_same_size},
          // must be horizontal, as Broadcast + Broadcast is not allowed.
          {framework::kBroadcast, is_same_size},
          // horizontal or vertical relation(Broadcast + Reduce).
          {framework::kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or just one consumer.
          {framework::kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {framework::kOutFusible, is_same_shape}};
      fusion_relation_map_[framework::kBroadcast] = std::move(relation);
    }
    // 3.kReduction as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElementWise, framework::kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Reduce + Elementwise*), check
          // without last dimension in reduce.
          {framework::kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape and
          // without last dimension in reduce.
          {framework::kBroadcast, reduce_fuse_broadcast},
          // must be horizontal relation and with same reduce attr.
          {framework::kReduction, reduce_fuse_reduce},
          // no_fuse
          {framework::kInjective, no_fuse},
          // can't fuse.
          {framework::kOutFusible, no_fuse}};
      fusion_relation_map_[framework::kReduction] = std::move(relation);
    }
    // 4.kInjective
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElementWise, framework::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // can be horizontal or vertical(Injective + Elementwise), check with
          // same output shape.
          {framework::kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape.
          {framework::kBroadcast, horizontal_with_same_size},
          // left to fusion merge pass.
          {framework::kReduction, no_fuse},
          // must be horizontal relation, check with same output shape.
          {framework::kInjective, horizontal_or_can_inline},
          // can't fuse.
          {framework::kOutFusible, no_fuse},
      };
      fusion_relation_map_[framework::kInjective] = std::move(relation);
    }
    // 5.kOutFusible
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElementWise, framework::kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation, check has same shape.
          {framework::kElementWise, is_same_shape},
          // it must be horizontal relation, check has same shape.
          {framework::kBroadcast, is_same_shape},
          // can't fuse.
          {framework::kReduction, no_fuse},
          // must be horizontal relation, check has same shape.
          {framework::kInjective, is_same_shape},
          // can't fuse.
          {framework::kOutFusible, no_fuse},
      };
      fusion_relation_map_[framework::kOutFusible] = std::move(relation);
    }
  }

  bool CanFuse(const Node* producer, const Node* consumer) {
    auto& relation = fusion_relation_map_[GetOpKind(producer)];
    // first step: check producer can be fused into consumer
    if (relation.op_kind.count(GetOpKind(consumer))) {
      auto& consumer_group = fusion_groups_[consumer];
      // second step: check producer can be fused into consumer group
      VLOG(3) << "Call ConditionFunction, Producer Op Pattern : "
              << GetOpKind(producer) << " , Consumer Group Pattern : "
              << consumer_group->op_pattern_kind;
      return relation.fusion_op_kind[consumer_group->op_pattern_kind](
          this, producer, fusion_groups_[consumer]);
    }

    return false;
  }
  std::vector<Node*> nodes_;
  std::unordered_map<const Node*, GroupPtr> fusion_groups_;

  struct FusionRelation {
    // producer -> consumer
    std::unordered_set<framework::OpPatternKind> op_kind = {};
    // producer -> fusion sonsumer
    std::unordered_map<framework::OpPatternKind, ConditionFunction>
        fusion_op_kind = {};
  };
  std::unordered_map<framework::OpPatternKind, FusionRelation>
      fusion_relation_map_;
};

void OpFusionPassInternal(Graph* graph) {
  VLOG(3) << "OpFusionPass...!";
  auto op_fusion_helper = OpFusionPassHelper(graph);
  graph->fusion_groups = op_fusion_helper();

  for (auto& group : graph->fusion_groups) {
    VLOG(3) << "Group Id : " << group->group_id;
    for (const auto& producer : group->producer_groups()) {
      VLOG(3) << "  producer group -> " << producer->group_id;
    }
    for (const auto& consumer : group->consumer_groups()) {
      VLOG(3) << "  consumer group -> " << consumer->group_id;
    }
  }
  VLOG(3) << "OpFusionPass Finish...!";
}

void BuildNonFusedGroupsPassInternal(framework::Graph* graph) {
  auto op_fusion_helper = OpFusionPassHelper(graph);
  VLOG(3) << "Apply OpFusionPass to generate initial non-fusion groups";
  graph->fusion_groups = op_fusion_helper(false);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(OpFusionPass) {
  CINN_REGISTER_PASS(OpFusionPass)
      .describe(
          "Op Fusion Pass which performs Ops fusion, Producer Ops are fused "
          "into Consumer Ops with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPassInternal);

  CINN_REGISTER_PASS(BuildNonFusedGroupsPass)
      .describe("Build No Fused Groups.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::BuildNonFusedGroupsPassInternal);

  return true;
}
