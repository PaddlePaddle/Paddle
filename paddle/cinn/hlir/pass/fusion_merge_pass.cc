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

#include "paddle/cinn/hlir/pass/fusion_merge_pass_util.h"

PD_DECLARE_bool(enhance_vertical_fusion_with_recompute);

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

using Comparator = Graph::Group::SharedGroupComparator;
using Hasher = Graph::Group::SharedGroupHasher;

using GroupPtr = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using ConditionFunction = std::function<bool(
    const FusionHelperBase*, const GroupPtr&, const GroupPtr&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class FusionMergePassHelper : public FusionHelperBase {
 public:
  explicit FusionMergePassHelper(const Graph* graph) : FusionHelperBase(graph) {
    fusion_groups_ = graph->fusion_groups;
    // init fusion relation.
    InitFusionRelation();
    // init input to consumers.
    InitInputToConsumers();
    // init fusion group index.
    InitFusionGroupsAndIndex();
  }

  GroupList operator()() {
    // run fusion merge untill no update.
    DoFusionMerge();
    for (auto& group : fusion_groups_) {
      VLOG(3) << "Fusion Group -> " << group->group_id;
      for (auto& sub_group : group->fused_sub_groups) {
        VLOG(3) << "  Fused Sub-Group -> " << sub_group->group_id;
      }
      for (const auto& producer : group->producer_groups()) {
        VLOG(3) << "  Producer -> " << producer->group_id;
      }
      for (const auto& consumer : group->consumer_groups()) {
        VLOG(3) << "  Consumer -> " << consumer->group_id;
      }
    }
    return fusion_groups_;
  }

 private:
  void DoFusionMerge() {
    VLOG(3) << "DoFusionMerge...!";
    while (DoHorizontalFusion()) {
    }
    while (DoVerticalFusion(/* recompute=*/false)) {
    }
    while (DoVerticalFusion(/* recompute=*/true)) {
    }
  }

  bool DoHorizontalFusion() {
    VLOG(3) << "DoHorizontalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= HorizontalFusion(producer, producer->consumer_groups());
    }

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoVerticalFusion(bool recompute) {
    VLOG(3) << "DoVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      if (!recompute) {
        updated |= HorizontalFusion(producer, producer->consumer_groups());
      }
      updated |=
          VerticalFusion(producer, producer->consumer_groups(), recompute);
    }
    // fuse input consumers
    updated |= FuseInputToConsumers();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  void UpdateFusionGroup() {
    VLOG(3) << "UpdateFusionGroup...";
    GroupList fusion_groups;
    std::unordered_set<GroupPtr, Hasher, Comparator> fusion_groups_set;
    // update fusion_groups_
    for (auto& group : fusion_groups_) {
      if (!group->belong_groups.size()) {
        fusion_groups.push_back(group);
        fusion_groups_set.insert(group);
      }
    }
    // keep group in order
    fusion_groups_.clear();
    fusion_groups_index_.clear();
    while (!fusion_groups_set.empty()) {
      bool is_ring = true;
      for (int idx = 0; idx < fusion_groups.size(); ++idx) {
        auto& group = fusion_groups[idx];
        if (!group.get()) {
          continue;
        }

        bool exist = false;
        for (const auto& producer : group->producer_groups()) {
          if (fusion_groups_set.count(producer)) {
            VLOG(4) << group->group_id << " " << producer->group_id;
            exist = true;
            break;
          }
        }

        if (!exist) {
          fusion_groups_index_[group] = fusion_groups_.size();
          fusion_groups_.push_back(group);
          fusion_groups_set.erase(group);
          group.reset();
          is_ring = false;
          continue;
        }
      }
      if (is_ring) {
        LOG(FATAL) << "Exists Ring, Please Check!";
      }
    }
  }

  bool HorizontalFusion(
      GroupPtr producer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    VLOG(3) << "HorizontalFusion...!";
    if (consumers.size() <= 1) {
      return false;
    }

    std::unordered_set<GroupPtr, Hasher, Comparator> candidates;
    for (const auto& consumer : consumers) {
      // relation
      auto& relation = fusion_relation_map_[consumer->op_pattern_kind];
      // check horizontal relation exist
      if (!relation.horizontal_relation.size()) {
        continue;
      }
      candidates.insert(consumer);
    }

    std::vector<GroupList> fusionable_consumers;
    for (auto& candidate : candidates) {
      // check dependency
      if (IsDependencySimplify(producer, candidate, candidates)) {
        VLOG(4) << "IsDependencySimplify, Can't fuse " << candidate->group_id
                << ", As it depency others!";
        continue;
      }

      if (IsDependency(producer, candidate, candidates)) {
        VLOG(4) << "IsDependency, Can't fuse " << candidate->group_id
                << ", As it depency others!";
        continue;
      }

      if (!fusionable_consumers.size()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }

      // check each fusionable groups
      bool fusionable = false;
      auto& relation = fusion_relation_map_[candidate->op_pattern_kind];
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!relation.horizontal_relation.count(last->op_pattern_kind)) {
          continue;
        }

        if (!relation.horizontal_relation[last->op_pattern_kind](
                this, candidate, last)) {
          continue;
        }

        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    bool updated = false;
    for (auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        updated = true;
        HorizontalFuse(groups);
      }
    }

    return updated;
  }

  void HorizontalFuse(const GroupList& consumers) {
    VLOG(3) << "HorizontalFuse Groups...";
    // create fusion group
    auto fused_group = std::make_shared<Graph::Group>();
    // As recompute exist which may case sub-group used by more than one time.
    std::vector<GroupPtr> repeat_sub_groups;
    std::unordered_set<GroupPtr, Hasher, Comparator> sub_group_set;
    // find the first consumer.
    GroupPtr first_consumer(nullptr);
    // fuse all group into fusion group.
    for (auto& consumer : consumers) {
      VLOG(3) << "fuse consumer " << consumer->group_id << " into fused_group!";
      // update depth
      fused_group->max_depth =
          std::max(fused_group->max_depth, consumer->max_depth);
      fused_group->min_depth =
          std::min(fused_group->min_depth, consumer->min_depth);
      // update group id
      if (fused_group->group_id.size()) {
        fused_group->group_id += "_" + consumer->group_id;
      } else {
        fused_group->group_id = consumer->group_id;
      }
      // set op pattern kind
      fused_group->op_pattern_kind =
          static_cast<int>(fused_group->op_pattern_kind) >=
                  static_cast<int>(consumer->op_pattern_kind)
              ? fused_group->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      for (auto& node : consumer->input_nodes) {
        if (fused_group->input_nodes.count(node.first)) {
          fused_group->input_nodes[node.first] += node.second;
        } else {
          fused_group->input_nodes.insert(node);
        }
      }
      // output node
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }
      // internal node
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // master node
      for (auto& node : consumer->master_nodes) {
        if (GetOpKind(node) == framework::kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }
      // insert sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          // check sub group is repeat.
          if (sub_group_set.count(sub_group)) {
            VLOG(3) << sub_group->group_id << " is repeated!";
            repeat_sub_groups.push_back(sub_group);
            continue;
          }
          // record sub group
          sub_group_set.insert(sub_group);

          // insert to fused sub group.
          fused_group->fused_sub_groups.push_back(sub_group);
          // update belongs group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      // producer group
      for (auto& producer : *consumer->mut_producer_groups()) {
        fused_group->mut_producer_groups()->insert(producer);
        // update producer's consumer
        producer->mut_consumer_groups()->erase(consumer);
        producer->mut_consumer_groups()->insert(fused_group);
      }
      // consumer group
      for (auto& gconsumer : *consumer->mut_consumer_groups()) {
        fused_group->mut_consumer_groups()->insert(gconsumer);
        // update consumer's producer
        gconsumer->mut_producer_groups()->erase(consumer);
        gconsumer->mut_producer_groups()->insert(fused_group);
      }
      // belongs group
      consumer->belong_groups.insert(fused_group);

      // find the first consumer.
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id
          << " index in fusion_groups_index_!";
      if (first_consumer.get()) {
        if (fusion_groups_index_[consumer] <
            fusion_groups_index_[first_consumer]) {
          first_consumer = consumer;
        }
      } else {
        first_consumer = consumer;
      }
    }

    // if node is output nodes of sub_group, check it can't be internal node.
    for (auto& sub_group : repeat_sub_groups) {
      // check each output node in sub_group.
      for (auto& node : sub_group->output_nodes) {
        // if node is not output node of fused_group.
        if (!fused_group->output_nodes.count(node)) {
          fused_group->internal_nodes.insert(node);
        }
      }
    }

    if (static_cast<int>(framework::kReduction) >
        static_cast<int>((consumers.back())->op_pattern_kind)) {
      auto consumer = consumers.back();

      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }
    } else {
      for (auto consumer = consumers.rbegin(); consumer != consumers.rend();
           ++consumer) {
        Node* master_node = nullptr;
        for (auto& node : (*consumer)->master_nodes) {
          if (GetOpKind(node) != framework::kReduction) {
            master_node = node;
            break;
          }
        }
        if (master_node) {
          VLOG(3) << "Insert Master node : " << master_node->id()
                  << " into group : " << fused_group->group_id;
          fused_group->master_nodes.insert(master_node);
          break;
        }
      }
    }

    auto postion = fusion_groups_index_[first_consumer];
    fusion_groups_[postion] = fused_group;
    fusion_groups_index_[fused_group] = postion;

    CHECK(fused_group->output_nodes.size())
        << "No output node is found, " << fused_group->group_id;
  }

  bool VerticalFusion(
      const GroupPtr& producer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers,
      bool recompute) {
    VLOG(3) << "VerticalFusion, Number of Consumers : " << consumers.size();
    auto& relation = fusion_relation_map_[producer->op_pattern_kind];
    // if producer can't fuse others
    if (!relation.vertical_relation.size()) {
      return false;
    }

    std::unordered_set<GroupPtr, Hasher, Comparator> fuse_consumers_unsafe;
    std::unordered_set<GroupPtr, Hasher, Comparator> fuse_consumers;
    for (const auto& consumer : consumers) {
      VLOG(4) << "Check consuemr " << consumer->group_id
              << " can fuse to producer " << producer->group_id;
      // if can't fuse
      if (!relation.vertical_relation.count(consumer->op_pattern_kind)) {
        VLOG(4) << "Can't fuse producer " << producer->group_id << " consumer "
                << consumer->group_id;
        continue;
      }

      // if condition function is false
      if (!relation.vertical_relation[consumer->op_pattern_kind](
              this, producer, consumer)) {
        VLOG(4) << "Can't fuse producer " << producer->group_id << " consumer "
                << consumer->group_id;
        continue;
      }

      fuse_consumers_unsafe.insert(consumer);

      if (IsDependencySimplify(producer, consumer, consumers)) {
        VLOG(4) << "IsDependencySimplify, Consumer " << consumer->group_id
                << " can't be master fused group!";
        continue;
      }

      if (IsDependency(producer, consumer, consumers)) {
        VLOG(4) << "IsDependency, Consumer " << consumer->group_id
                << " can't be master fused group!";
        continue;
      }

      fuse_consumers.insert(consumer);
    }

    VLOG(3) << "VerticalFusion, Number of fuse Consumers : "
            << fuse_consumers.size();
    VLOG(3) << "VerticalFusion, Number of unsafe fuse Consumers : "
            << fuse_consumers.size();

    if (fuse_consumers.size() == 0) {
      return false;
    }
    // if can_fuse_consumers == consumers
    // if producer op kind == kElementwise
    // if use recompute
    if (fuse_consumers_unsafe.size() == producer->consumer_groups().size() &&
        producer->op_pattern_kind == framework::kElementWise) {
      if (!recompute) {
        return false;
      } else {
        RecomputeEleGraph(producer, &fuse_consumers_unsafe);
        VerticalFuse(producer, fuse_consumers_unsafe);
        return true;
      }
    }

    if (fuse_consumers.size()) {
      SelectConsumerToFuse(producer, &fuse_consumers);
    }

    // if fusionable consumers exist
    if (fuse_consumers.size()) {
      VerticalFuse(producer, fuse_consumers);
      return true;
    }

    return false;
  }

  void VerticalFuse(const GroupPtr& producer,
                    const std::unordered_set<GroupPtr, Hasher, Comparator>&
                        fusionable_consumers) {
    VLOG(3) << "VerticalFuse...!";
    GroupList fused_groups;
    GroupPtr master_fuesd_group(nullptr);
    for (auto& consumer : fusionable_consumers) {
      auto fused_group = std::make_shared<Graph::Group>();
      // update depth using consumer depth.
      fused_group->max_depth =
          std::max(producer->max_depth, consumer->max_depth);
      fused_group->min_depth =
          std::min(producer->min_depth, consumer->min_depth);
      // update group id
      fused_group->group_id = producer->group_id + "_" + consumer->group_id;
      VLOG(3) << "fuse producer " << producer->group_id << " into consumer "
              << consumer->group_id;
      // fuse producer into fusion group
      fused_group->op_pattern_kind =
          static_cast<int>(producer->op_pattern_kind) >=
                  static_cast<int>(consumer->op_pattern_kind)
              ? producer->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      fused_group->input_nodes = producer->input_nodes;

      // internal nodes
      if (producer->fused_sub_groups.size()) {
        for (auto& node : producer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // convert producer's output node to internal.
      for (auto node : producer->output_nodes) {
        // if node is used more than 1 time.
        if (consumer->input_nodes.count(node)) {
          if (consumer->input_nodes[node] > 1 && node->inlinks().size() > 0) {
            fused_group->internal_nodes.insert(node);
          }
        }
      }
      // master nodes
      for (auto& node : producer->master_nodes) {
        if (GetOpKind(node) == framework::kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }

      // producer groups
      for (auto& group : *producer->mut_producer_groups()) {
        fused_group->mut_producer_groups()->insert(group);
        // update producer's producer's consumer
        group->mut_consumer_groups()->erase(producer);
        group->mut_consumer_groups()->insert(fused_group);
      }

      // sub groups
      if (producer->fused_sub_groups.size()) {
        for (auto& group : producer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(group);
          // update belong group
          group->belong_groups.erase(producer);
          group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(producer);
      }
      producer->belong_groups.insert(fused_group);

      // input nodes
      for (auto& input_node : consumer->input_nodes) {
        // if input node not in producer output.
        if (!producer->output_nodes.count(input_node.first)) {
          if (fused_group->input_nodes.count(input_node.first)) {
            fused_group->input_nodes[input_node.first] += input_node.second;
          } else {
            fused_group->input_nodes.insert(input_node);
          }
        }
      }

      // output nodes
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }

      // internal nodes
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }

      // master nodes
      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }

      // producer nodes
      for (auto& group : *consumer->mut_producer_groups()) {
        if (group.get() != producer.get()) {
          fused_group->mut_producer_groups()->insert(group);
          // update consumer's producer's consumer
          group->mut_consumer_groups()->erase(consumer);
          group->mut_consumer_groups()->insert(fused_group);
        }
      }
      // consumer nodes
      for (auto& group : *consumer->mut_consumer_groups()) {
        fused_group->mut_consumer_groups()->insert(group);
        // update consumer's consumer's producer
        group->mut_producer_groups()->erase(consumer);
        group->mut_producer_groups()->insert(fused_group);
      }

      // sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          if (std::find(fused_group->fused_sub_groups.begin(),
                        fused_group->fused_sub_groups.end(),
                        sub_group) == fused_group->fused_sub_groups.end()) {
            fused_group->fused_sub_groups.push_back(sub_group);
          }
          // update belong group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      consumer->belong_groups.insert(fused_group);

      fused_groups.push_back(fused_group);
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id
          << " index in fusion_groups_index_!";
      auto postion = fusion_groups_index_[consumer];
      fusion_groups_[postion] = fused_group;
      fusion_groups_index_[fused_group] = postion;

      if (!master_fuesd_group.get()) {
        master_fuesd_group = fused_group;
      }
      CHECK(fused_group->output_nodes.size())
          << "No output node is found, " << fused_group->group_id;
    }

    for (auto& node : producer->output_nodes) {
      bool be_output = true;
      for (const auto& consumer : producer->consumer_groups()) {
        // if consumer is in fusionable.
        if (fusionable_consumers.count(consumer)) {
          if (consumer->input_nodes.count(node)) {
            be_output = false;
          }
          continue;
        }
        // if consumer is not in fusionable.
        if (consumer->input_nodes.count(node)) {
          be_output = true;
          break;
        }
        // others node is as graph output.
      }

      if (output_nodes_set_.count(node)) {
        be_output = true;
      }

      if (be_output) {
        VLOG(4) << "Insert Id " << node->id() << " Into Group "
                << master_fuesd_group->group_id;
        master_fuesd_group->output_nodes.insert(node);
      }
    }
    // insert unfusionable consumer groups
    for (auto& consumer : *producer->mut_consumer_groups()) {
      if (fusionable_consumers.count(consumer)) {
        continue;
      }
      master_fuesd_group->mut_consumer_groups()->insert(consumer);
      // update consumer's producer
      consumer->mut_producer_groups()->erase(producer);
      consumer->mut_producer_groups()->insert(master_fuesd_group);
    }
  }

  void RecomputeEleGraph(
      const GroupPtr& producer,
      std::unordered_set<GroupPtr, Hasher, Comparator>* fusionable_consumers) {
    if (producer->op_pattern_kind != framework::kElementWise) {
      SelectConsumerToFuse(producer, fusionable_consumers);
    }
  }

  void SelectConsumerToFuse(
      const GroupPtr& producer,
      std::unordered_set<GroupPtr, Hasher, Comparator>* fusionable_consumers) {
    // if is const op
    if (is_const_group(this, producer)) {
      std::unordered_set<GroupPtr, Hasher, Comparator> candidates;
      for (auto& consumer : *fusionable_consumers) {
        // if can be output node.
        if (is_same_shape(this, producer, consumer)) {
          candidates.insert(consumer);
        } else {
          VLOG(4) << "Fuse Producer : " << producer->group_id
                  << " into Consumer : " << consumer->group_id;
          consumer->group_id = producer->group_id + "_" + consumer->group_id;
          // just merge the node into group.
          auto& sub_group = consumer->fused_sub_groups.front();
          sub_group->group_id = producer->group_id + "_" + sub_group->group_id;
          sub_group->nodes.insert(sub_group->nodes.begin(),
                                  producer->CollectNodes()[0]);
          sub_group->nodes_set.insert(producer->CollectNodes()[0]);
          // remove depency.
          consumer->input_nodes.erase(producer->CollectNodes()[0]);
          consumer->mut_producer_groups()->erase(producer);
          producer->mut_consumer_groups()->erase(consumer);
        }
      }

      CHECK_GE(producer->consumer_groups().size(), candidates.size());
      if (producer->consumer_groups().size() == 0 && candidates.size() == 0 &&
          output_nodes_set_.count(producer->CollectNodes()[0]) == 0) {
        producer->belong_groups.insert(*fusionable_consumers->begin());
      }

      *fusionable_consumers = candidates;
      return;
    }
    // 1 to 1 fusion.
    if (producer->consumer_groups().size() == 1) {
      return;
    }

    if (FLAGS_enhance_vertical_fusion_with_recompute) {
      std::vector<GroupPtr> candidates;
      for (auto& consumer : *fusionable_consumers) {
        if (consumer->op_pattern_kind == framework::kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto producer_output_shape =
            this->GetNodeDataShape(*producer->output_nodes.begin());
        auto consumer_output_shape =
            this->GetNodeDataShape(*consumer->output_nodes.begin());
        auto consumer_master_input_shape =
            this->GetNodeInputShape(*(consumer->master_nodes.begin()));
        int producer_output_numel =
            std::accumulate(producer_output_shape.begin(),
                            producer_output_shape.end(),
                            1,
                            std::multiplies<int>());
        int consumer_output_numel =
            std::accumulate(consumer_output_shape.begin(),
                            consumer_output_shape.end(),
                            1,
                            std::multiplies<int>());
        int consumer_master_input_numel =
            std::accumulate(consumer_master_input_shape.begin(),
                            consumer_master_input_shape.end(),
                            1,
                            std::multiplies<int>());
        if (producer_output_numel == consumer_output_numel) {
          candidates.push_back(consumer);
          continue;
        }

        if (producer->op_pattern_kind != framework::kInjective &&
            consumer->op_pattern_kind == framework::kReduction &&
            producer_output_numel == consumer_master_input_numel) {
          candidates.push_back(consumer);
        }
      }
      sort(candidates.begin(),
           candidates.end(),
           [](const auto& lhs, const auto& rhs) {
             return lhs->op_pattern_kind < rhs->op_pattern_kind;
           });

      fusionable_consumers->clear();
      if (candidates.size()) {
        fusionable_consumers->insert(*candidates.begin());
      }
    } else {
      std::unordered_set<GroupPtr, Hasher, Comparator> candidates;
      for (auto& consumer : *fusionable_consumers) {
        if (consumer->op_pattern_kind == framework::kElementWise) {
          candidates.insert(consumer);
          continue;
        }

        auto shape0 = this->GetNodeDataShape(*producer->output_nodes.begin());
        auto shape1 = this->GetNodeDataShape(*consumer->output_nodes.begin());

        if (std::accumulate(
                shape0.begin(), shape0.end(), 1, std::multiplies<int>()) ==
            std::accumulate(
                shape1.begin(), shape1.end(), 1, std::multiplies<int>())) {
          candidates.insert(consumer);
        }
      }

      fusionable_consumers->clear();
      if (candidates.size()) {
        fusionable_consumers->insert(*candidates.begin());
      }
    }
  }

  bool IsDependency(
      const GroupPtr& producer_g,
      const GroupPtr& consumer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);

    std::unordered_set<GroupPtr, Hasher, Comparator> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (const auto& producer : candidate->producer_groups()) {
        if (producer.get() == producer_g.get()) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool IsDependencySimplify(
      const GroupPtr& producer_g,
      const GroupPtr& consumer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);
    // check upper.
    int check_upper_depth = producer_g.get() ? producer_g->max_depth : INT_MAX;
    std::unordered_set<GroupPtr, Hasher, Comparator> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (auto& producer : candidate->producer_groups()) {
        if (producer.get() == producer_g.get()) {
          continue;
        }
        if (producer->min_depth > check_upper_depth) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool FuseInputToConsumers() {
    VLOG(3) << "FuseInputToConsumers...!";
    auto updated = false;
    UpdateInputToConsumers();
    GroupPtr producer(nullptr);
    for (auto& input_consumers : input_to_consumers_) {
      // if group set size == 1.
      if (input_consumers.second.size() == 1) {
        continue;
      }
      // do horizontal fusion.
      auto st = HorizontalFusion(producer, input_consumers.second);
      if (st) {
        // fused consumers, update
        UpdateInputToConsumers();
      }
      updated |= st;
    }

    return updated;
  }

  void UpdateInputToConsumers() {
    for (auto& input_consumers : input_to_consumers_) {
      auto& consumers = input_consumers.second;
      std::unordered_set<GroupPtr, Hasher, Comparator> updated_consumers;
      for (auto& consumer : consumers) {
        std::queue<GroupPtr> fused_groups;
        fused_groups.push(consumer);
        while (!fused_groups.empty()) {
          auto& cur = fused_groups.front();
          fused_groups.pop();
          // if group is sub group
          if (cur->belong_groups.empty()) {
            updated_consumers.insert(cur);
          } else {
            for (auto& belong_group : cur->belong_groups) {
              if (belong_group->group_id == cur->group_id) {
                updated_consumers.insert(belong_group);
              } else {
                fused_groups.push(belong_group);
              }
            }
          }
        }
      }
      consumers = updated_consumers;
    }
  }

  void InitInputToConsumers() {
    VLOG(3) << "InitInputToConsumers...!";
    // init input data node -> fusion group map.
    for (auto& group : fusion_groups_) {
      for (auto& node : group->nodes_set) {
        // collect producer node data.
        auto producer_node_datas = GetProducerNodeData(node);
        for (auto& node_data : producer_node_datas) {
          // node data's source node is null.
          if (!node_data->source_node.get()) {
            // insert group to set.
            input_to_consumers_[node_data].insert(group);
          }
        }
      }
    }
  }

  void InitFusionGroupsAndIndex() {
    VLOG(3) << "InitFusionGroupsAndIndex...!";
    // init the postion of groups in fusion groups.
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto group = fusion_groups_[idx];
      auto belong_group = std::make_shared<Graph::Group>();
      // copy from group.
      belong_group->max_depth = group->depth;
      belong_group->min_depth = group->depth;
      belong_group->group_id = group->group_id;
      belong_group->input_nodes = group->input_nodes;
      belong_group->output_nodes = group->output_nodes;
      belong_group->op_pattern_kind = group->op_pattern_kind;
      belong_group->master_nodes = group->master_nodes;
      (*belong_group->mut_producer_groups()) = group->producer_groups();
      (*belong_group->mut_consumer_groups()) = group->consumer_groups();
      belong_group->fused_sub_groups.push_back(group);
      group->belong_groups.insert(belong_group);
      // replace group to fused_group
      fusion_groups_[idx] = belong_group;
      // record idx
      fusion_groups_index_[belong_group] = idx;
    }

    // update producer and consumer.
    for (auto& group : fusion_groups_) {
      std::unordered_set<GroupPtr, Hasher, Comparator> producers;
      std::unordered_set<GroupPtr, Hasher, Comparator> consumers;

      for (const auto& producer : group->producer_groups()) {
        CHECK(producer->belong_groups.size());
        producers.insert(*producer->belong_groups.begin());
      }

      for (auto& consumer : *group->mut_consumer_groups()) {
        CHECK(consumer->belong_groups.size());
        consumers.insert(*consumer->belong_groups.begin());
      }
      CHECK_EQ(group->producer_groups().size(), producers.size());
      CHECK_EQ(group->consumer_groups().size(), consumers.size());
      (*group->mut_producer_groups()) = producers;
      (*group->mut_consumer_groups()) = consumers;
    }
  }

  void InitFusionRelation() {
    VLOG(3) << "InitFusionRelation...!";
    // kElementWise
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kElementWise];
      // horizontal
      relation.horizontal_relation = {
          {framework::kElementWise, is_same_size},
          // element-wise and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, is_same_size},
          // element-wise and injective op must be horizontal relation.
          {OpPatternKind::kInjective, is_same_size},
          // element-wise and reduce op must be horizontal relation.
          {OpPatternKind::kReduction, honrizontal_elementwise_fuse_reduce}};
      // vertical
      relation.vertical_relation = {
          {OpPatternKind::kElementWise, is_same_size},
          // element-wise and broadcast can be vertical/horizontal relation.
          {OpPatternKind::kBroadcast, elementwise_fuse_broadcast},
          // element-wise and injective op must be horizontal relation.
          {OpPatternKind::kInjective, horizontal_with_injective},
          // element-wise and reduce can be vertical/horizontal relation.
          {OpPatternKind::kReduction, elementwise_fuse_reduce}};
    }
    // kBroadcast
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kBroadcast];
      // horizontal
      relation.horizontal_relation = {
          // broadcast and element-wise op must be horizontal relation.
          {framework::kElementWise, is_same_size},
          // broadcast and broadcast op must be horizontal relation.
          {framework::kBroadcast, is_same_size},
          // broadcast and injective op must be horizontal relation.
          {OpPatternKind::kInjective, is_same_size},
          // broadcast and reduce op must be horizontal relation.
          {OpPatternKind::kReduction, is_same_size}};
      // vertical
      relation.vertical_relation = {
          // broadcast and element-wise op must be vertical relation.
          {OpPatternKind::kElementWise, is_same_size},
          // broadcast and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, is_same_size},
          // broadcast and injective op must be horizontal relation.
          {OpPatternKind::kInjective, horizontal_with_injective},
          // broadcast and reduce must be vertical relation.
          {OpPatternKind::kReduction, broadcast_fuse_reduce}};
    }
    // kInjective
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kInjective];
      // horizontal
      relation.horizontal_relation = {
          // injective and element-wise op must be horizontal relation.
          {OpPatternKind::kElementWise, is_same_size},
          // injective and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, is_same_size},
          // injective and injective op must be horizontal relation.
          {OpPatternKind::kInjective, is_same_size},
          // injective and reduce must be horizontal relation.
          {OpPatternKind::kReduction, is_same_size}};
      // vertical
      relation.vertical_relation = {
          // injective and element-wise op must be horizontal relation.
          {OpPatternKind::kElementWise, is_same_size},
          // injective and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, is_same_size},
          // injective and injective op must be horizontal relation.
          {OpPatternKind::kInjective, horizontal_with_injective},
          // injective and reduce can be horizontal/vertical relation.
          {OpPatternKind::kReduction, injective_horizontal_with_reduce}};
    }
    // kReduction
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kReduction];
      // horizontal
      relation.horizontal_relation = {
          // reduce and element-wise op must be horizontal relation.
          {OpPatternKind::kElementWise, honrizontal_elementwise_fuse_reduce},
          // reduce and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, is_same_size},
          // reduce and injective op must be horizontal relation.
          {OpPatternKind::kInjective, is_same_size},
          // reduce and reduce must be horizontal relation.
          {OpPatternKind::kReduction, reduce_fuse_reduce}};
      // vertical
      relation.vertical_relation = {
          // reduce and elementwise can be horizontal/vertical relation.
          {OpPatternKind::kElementWise, reduce_fuse_elementwise},
          // reduce and broadcast op must be horizontal relation.
          {OpPatternKind::kBroadcast, reduce_fuse_broadcast},
          // reduce and injective op must be horizontal relation.
          {OpPatternKind::kInjective, horizontal_with_injective},
          // reduce and reduce must be horizontal relation.
          {OpPatternKind::kReduction, reduce_fuse_reduce}};
    }
  }

  GroupList fusion_groups_;
  std::unordered_map<GroupPtr, int, Hasher, Comparator> fusion_groups_index_;
  std::unordered_map<NodeData*,
                     std::unordered_set<GroupPtr, Hasher, Comparator>>
      input_to_consumers_;

  struct Relation {
    std::unordered_map<framework::OpPatternKind, ConditionFunction>
        vertical_relation;
    std::unordered_map<framework::OpPatternKind, ConditionFunction>
        horizontal_relation;
  };
  std::unordered_map<framework::OpPatternKind, Relation> fusion_relation_map_;
};

void FusionMergePassInternal(Graph* graph) {
  if (graph->fusion_groups.size() <= 1) {
    VLOG(3) << "Don't do Fusoin Merge Pass...!";
    return;
  }

  FusionMergePassHelper fusion_merge_pass_helper(graph);
  graph->fusion_groups = fusion_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(FusionMergePass) {
  CINN_REGISTER_PASS(FusionMergePass)
      .describe(
          "Fusion Merge Pass which performs Fusion-Ops fusion, Producer "
          "Fusion-Ops are fused into Consumer Fusion-Ops "
          "with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::FusionMergePassInternal);

  return true;
}
