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

#include <map>
#include <unordered_map>

#include "glog/logging.h"

#include "paddle/cinn/api/op_group.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/fusion_pass_map.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/graph_group_input_fuse_pass_ctx.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/graph_group_lightware_fuse_pass_ctx.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/input_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass_ctx.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass_utils.h"

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

using GroupPtr = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using Comparator = Graph::Group::SharedGroupComparator;
using Hasher = Graph::Group::SharedGroupHasher;

using OpGroupPtr = api::OpGroup;
using OpGroupList = std::vector<OpGroupPtr>;

using ConditionFunction = std::function<bool(
    const FusionHelperBase*, const GroupPtr&, const GroupPtr&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class GeneralFusionMergePassHelper : public FusionHelperBase {
 public:
  explicit GeneralFusionMergePassHelper(const Graph* graph)
      : FusionHelperBase(graph), graph_(graph) {
    fusion_groups_ = graph->fusion_groups;
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
    while (DoGeneralHorizontalFusion()) {
    }
    while (DoGeneralVerticalFusion()) {
    }
    while (DoGeneralRecomputeAndVerticalFusion()) {
    }
  }

  bool DoGeneralHorizontalFusion() {
    VLOG(3) << "DoGeneralHorizontalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
    }

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralVerticalFusion() {
    VLOG(3) << "DoGeneralVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
      updated |= GeneralVerticalFuse(producer);
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralRecomputeAndVerticalFusion() {
    VLOG(3) << "DoGeneralRecomputeAndVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      bool recompute_success = GeneralRecomputeFuse(producer);
      updated |= recompute_success;
      if (!recompute_success) {
        updated |= GeneralVerticalFuse(producer);
      }
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

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

  std::vector<std::shared_ptr<LightwareFusePass>> RawHorizontalFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "HorizontalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>&
  GetHorizontalFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawHorizontalFusePasses();
    return fuse_passes;
  }

  void EnableFusedHorizontalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer.consumers().size() <= 1) {
      return;
    }
    const auto& fuse_passes = GetHorizontalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralHorizontalFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralHorizontalFuse handling producer : "
            << producer->group_id;
    const auto& GetFusableConsumerGroupLists =
        [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& MarkFusible = [&](const OpGroupList& candidates) {
        tagged_lists.push_back(candidates);
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(
          this, api::OpGroup(producer), MarkFusible);
      EnableFusedHorizontalGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(group.GetGroup());
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  std::vector<std::shared_ptr<InputFusePass>> RawInputFusePasses() const {
    return FusionPassMap::Instance().GetInputFusePasses();
  }

  const std::vector<std::shared_ptr<InputFusePass>>& GetInputFusePasses()
      const {
    thread_local static std::vector<std::shared_ptr<InputFusePass>>
        fuse_passes = RawInputFusePasses();
    return fuse_passes;
  }

  void EnableFusedInputGroups(InputFusePassCtx* ctx) const {
    const auto& fuse_passes = GetInputFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool CallGeneralInputFusePass(
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    VLOG(3) << "CallGeneralInputFusePass...!";
    const auto& GetFusableConsumerGroupLists =
        [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& MarkFusible = [&](const OpGroupList& candidates) {
        tagged_lists.push_back(candidates);
      };
      OpGroupList consumer_groups;
      consumer_groups.reserve(consumers.size());
      for (auto& consumer : consumers) {
        consumer_groups.push_back(api::OpGroup(consumer));
      }
      GraphGroupInputFusePassCtx fuse_ctx(this, consumer_groups, MarkFusible);
      EnableFusedInputGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(group.GetGroup());
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  void HorizontalFuse(const GroupList& consumers) {
    VLOG(3) << "HorizontalFuse Groups...";
    // create fusion group
    auto fused_group = std::make_shared<Graph::Group>(graph_);
    // As recompute exist which may case sub-group used by more than one time.
    std::vector<GroupPtr> repeat_sub_groups;
    std::unordered_set<GroupPtr, Hasher, Comparator> sub_group_set;
    // find the first consumer.
    GroupPtr first_consumer(nullptr);
    // fuse all group into fusion group.
    for (const auto& consumer : consumers) {
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

  std::vector<std::shared_ptr<LightwareFusePass>> RawVerticalFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "VerticalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>& GetVerticalFusePasses()
      const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawVerticalFusePasses();
    return fuse_passes;
  }

  void TagVerticalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer.consumers().size() == 0) {
      return;
    }
    const auto& fuse_passes = GetVerticalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralVerticalFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralVerticalFuse...!";
    using GroupSets = std::vector<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& MarkFusible = [&](const OpGroupPtr& first,
                                    const OpGroupPtr& second) {
        tagged_sets.push_back(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(
          this, api::OpGroup(producer), MarkFusible);
      TagVerticalGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet =
        [&]() -> std::unordered_set<GroupPtr, Hasher, Comparator> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr, Hasher, Comparator> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(group_pair.second.GetGroup());
      }
      return ret;
    };

    bool update = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size()) {
      SelectConsumerToFuse(producer, &consumer_groups);
    }
    if (consumer_groups.size() > 0) {
      VerticalFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void VerticalFuse(const GroupPtr& producer,
                    const std::unordered_set<GroupPtr, Hasher, Comparator>&
                        fusionable_consumers) {
    VLOG(3) << "VerticalFuse...!";
    GroupList fused_groups;
    GroupPtr master_fuesd_group(nullptr);
    for (auto& consumer : fusionable_consumers) {
      auto fused_group = std::make_shared<Graph::Group>(graph_);
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

  std::vector<std::shared_ptr<LightwareFusePass>> RawRecomputeFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "RecomputeFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>&
  GetRecomputeFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawRecomputeFusePasses();
    return fuse_passes;
  }

  void TagRecomputeGroups(LightwareFusePassCtx* ctx) const {
    const auto& fuse_passes = GetRecomputeFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralRecomputeFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralRecomputeFuse handling producer : "
            << producer->group_id;
    using GroupSets = std::set<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& MarkFusible = [&](const OpGroupPtr& first,
                                    const OpGroupPtr& second) {
        tagged_sets.insert(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(
          this, api::OpGroup(producer), MarkFusible);
      TagRecomputeGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet =
        [&]() -> std::unordered_set<GroupPtr, Hasher, Comparator> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr, Hasher, Comparator> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(group_pair.second.GetGroup());
      }
      return ret;
    };

    bool update = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size() > 0) {
      CHECK(consumer_groups.size() == producer->mut_consumer_groups()->size())
          << "Recompute requires fuse all consumers!";
      RecomputeFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void RecomputeFuse(const GroupPtr& producer,
                     const std::unordered_set<GroupPtr, Hasher, Comparator>&
                         fusionable_consumers) {
    VerticalFuse(producer, fusionable_consumers);
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
      std::vector<GroupPtr> candidates;
      for (auto& consumer : *fusionable_consumers) {
        if (consumer->op_pattern_kind == framework::kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto shape0 = this->GetNodeDataShape(*producer->output_nodes.begin());
        auto shape1 = this->GetNodeDataShape(*consumer->output_nodes.begin());

        if (std::accumulate(
                shape0.begin(), shape0.end(), 1, std::multiplies<int>()) ==
            std::accumulate(
                shape1.begin(), shape1.end(), 1, std::multiplies<int>())) {
          candidates.push_back(consumer);
        }
      }

      fusionable_consumers->clear();
      if (candidates.size()) {
        fusionable_consumers->insert(candidates.front());
      }
    }
  }

  bool GeneralInputFuse() {
    VLOG(3) << "GeneralInputFuse...!";
    auto updated = false;
    UpdateInputToConsumers();
    for (auto& input_consumers : input_to_consumers_) {
      // if group set size == 1.
      if (input_consumers.second.size() == 1) {
        continue;
      }
      // do input fusion.
      auto st = CallGeneralInputFusePass(input_consumers.second);
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
      auto belong_group = std::make_shared<Graph::Group>(graph_);
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

  const Graph* graph_;
  GroupList fusion_groups_;
  std::unordered_map<GroupPtr, int> fusion_groups_index_;
  std::unordered_map<NodeData*,
                     std::unordered_set<GroupPtr, Hasher, Comparator>>
      input_to_consumers_;
};

void GeneralFusionMergePassInternal(Graph* graph) {
  if (graph->fusion_groups.size() <= 1) {
    VLOG(3) << "Don't do Fusoin Merge Pass...!";
    return;
  }

  GeneralFusionMergePassHelper fusion_merge_pass_helper(graph);
  graph->fusion_groups = fusion_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(GeneralFusionMergePass) {
  CINN_REGISTER_PASS(GeneralFusionMergePass)
      .describe(
          "Fusion Merge Pass which performs Fusion-Ops fusion, Producer "
          "Fusion-Ops are fused into Consumer Fusion-Ops "
          "with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::GeneralFusionMergePassInternal);

  return true;
}
