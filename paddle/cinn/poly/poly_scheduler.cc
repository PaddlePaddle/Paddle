// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/poly/poly_scheduler.h"

#include <glog/logging.h>

#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace poly {

namespace detail {

//! Visit the nodes in topological order, if one node is valid to visit, visit
//! it and check whether its out link children are ready to visit, merge them to
//! the same group. NOTE this is discarded.
std::vector<Group> PartitionGraphByIterationDomain(cinn::common::Graph* graph) {
  VLOG(3) << "graph:\n" << graph->Visualize();
  // collect indegrees for naive topological traversal.
  std::map<DataFlowGraphNode*, uint16_t> indegree;
  for (cinn::common::GraphNode* n : graph->nodes()) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    indegree[node] = node->inlinks().size();
  }

  std::map<std::string, DataFlowGraphNode*> name2node;
  for (auto* n : graph->nodes()) {
    name2node[n->id()] = n->safe_as<DataFlowGraphNode>();
  }

  // topological sort.
  std::deque<DataFlowGraphNode*> queue;
  for (auto* n : graph->start_points()) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    queue.push_back(node);
  }
  while (!queue.empty()) {
    auto* node = queue.front();
    queue.pop_front();
    VLOG(4) << "to visit " << node->id();

    for (auto& c : node->outlinks()) {
      auto* child = c->sink()->safe_as<DataFlowGraphNode>();
      --indegree[child];

      VLOG(3) << node->stage->transformed_domain() << " -> "
              << child->stage->transformed_domain();
      if (indegree[child] == 0) {
        // Merge the two groups if their iteration domain is the same.
        if (DataFlowGraphNode::TransformedDomainIsSame(node, child)) {
          VLOG(4) << child->id() << " ready to merge " << node->id() << " with "
                  << child->id();
          DataFlowGraphNode::MergeGroup(node, child);
        }
        queue.push_back(child);
      }
    }
  }

  // process the ComputeAt relation.
  for (auto* n : graph->nodes()) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    for (auto& compute_at : node->stage->compute_ats()) {
      PADDLE_ENFORCE_EQ(compute_at.IsCompatible(node->stage.get()),
                        true,
                        ::common::errors::InvalidArgument(
                            "The registered ComputeAt is not compatible."));
      // check the endpoints of compute_at has data dependency.
      auto* node0 = node;
      auto* node1 = name2node[compute_at.stage->id()];
      VLOG(3) << "a -> b: " << node0->id() << " -> " << node1->id();

      DataFlowGraphNode::MergeGroup(node0, node1);
      // TODO(Superjomn) Consider the case node1 is a parent.
    }
  }

  // gather groups
  std::set<DataFlowGraphNode*> groups_gathered;
  std::vector<DataFlowGraphNode*> groups_in_topo_order;

  std::map<DataFlowGraphNode*, std::vector<DataFlowGraphNode*>> node_groups;

  auto topo_order = graph->topological_order();
  auto& nodes_in_order = std::get<0>(topo_order);
  auto& edges_in_order = std::get<1>(topo_order);

  for (auto* n : nodes_in_order) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    auto* ancestor = node->group_ancestor();
    if (!groups_gathered.count(ancestor)) {
      groups_gathered.insert(ancestor);
      groups_in_topo_order.push_back(ancestor);
    }

    node_groups[ancestor].push_back(node);
  }

  std::vector<Group> groups;
  // preparing result
  for (auto* ancestor : groups_in_topo_order) {
    Group group;
    for (auto* c : node_groups[ancestor]) {
      group.nodes.push_back(c);
    }
    groups.emplace_back(group);
  }

  // NOTE DEBUG
  // check there are same count of nodes both in the original graph and the
  // groups.
  // @{
  int num_node_in_groups = 0;
  for (auto& group : groups) num_node_in_groups += group.nodes.size();
  PADDLE_ENFORCE_EQ(num_node_in_groups,
                    graph->num_nodes(),
                    ::common::errors::InvalidArgument(
                        "The number of nodes in groups should be the same as "
                        "the number of nodes in the graph"));
  // @}

  return groups;
}

//! Check whether a group partition is valid. The ComputeAt and some other
//! transform may broke data dependency, use this to check validity.
// TODO(Superjomn) Implement this and integrate it into ComputeAt transform for
// checking transform validity.
bool CheckGroupValid(const std::vector<Group>& groups) {
  CINN_NOT_IMPLEMENTED
  return false;
}

//! Tell if \param a links to \param b.
bool IsLinkTo(const cinn::common::GraphNode* a,
              const cinn::common::GraphNode* b) {
  // dfs
  std::stack<const cinn::common::GraphNode*> stack({a});
  std::unordered_set<const cinn::common::GraphNode*> visited;
  while (!stack.empty()) {
    auto* top = stack.top();
    stack.pop();
    if (visited.count(top)) continue;

    if (top == b) return true;

    for (auto& out : top->outlinks()) {
      auto* x = out->sink();
      if (!visited.count(x)) {
        if (x == b) return true;
        stack.push(x);
      }
    }
    visited.insert(top);
  }

  return false;
}

bool IsBetween(const cinn::common::GraphNode* x,
               const cinn::common::GraphNode* a,
               const cinn::common::GraphNode* b) {
  if (IsLinkTo(a, x) && IsLinkTo(x, b)) return true;
  if (IsLinkTo(x, a) && IsLinkTo(b, x)) return true;
  return false;
}

std::vector<Group> TopoSortGroups(std::vector<Group>& groups) {  // NOLINT
  // collect indegree.
  absl::flat_hash_map<Group*, int> group_indegree;
  std::vector<Group*> start_groups;
  std::deque<Group*> queue;
  std::vector<Group> group_order;
  absl::flat_hash_map<std::string, Group*> node2group;
  for (int i = 0; i < groups.size(); i++) {
    Group* group = &groups[i];
    int in_degree = 0;
    for (auto& node : group->nodes) {
      node2group[node->id()] = group;
      in_degree += node->inlinks().size();
      for (auto& node2 : group->nodes) {
        if (node2->as<cinn::common::GraphNode>()->IsLinkedTo(
                node->as<cinn::common::GraphNode>())) {
          in_degree--;
        }
      }
    }
    group_indegree[group] = in_degree;
    if (in_degree == 0) {
      start_groups.push_back(group);
    }
  }

  // insert start points first.
  for (auto* n : start_groups) {
    queue.push_back(n);
  }

  // start to visit
  while (!queue.empty()) {
    auto* top_group = queue.front();
    group_order.push_back(*top_group);

    queue.pop_front();
    std::set<std::string> all_nodes;

    for (auto& node : top_group->nodes) {
      all_nodes.insert(node->id());
    }
    for (auto& node : top_group->nodes) {
      for (auto& edge : node->outlinks()) {
        PADDLE_ENFORCE_EQ(
            edge->source()->id(),
            node->id(),
            ::common::errors::InvalidArgument(
                "The edge source should be the same as the node"));
        auto* sink = edge->sink();
        if (all_nodes.count(sink->id()) == 0 &&
            (--group_indegree[node2group[sink->id()]]) == 0) {
          queue.push_back(node2group[sink->id()]);
        }
      }
    }
  }
  return group_order;
}

/**
 * Naive idea to split a graph.
 *
 * 1. treat each stage as a separate group.
 * 2. If ComputeAt is set between two stages and their iteration domain matches,
 * the stages will be put in a group with relative order.
 */
std::vector<Group> NaivePartitionGraph(cinn::common::Graph* graph) {
  std::map<DataFlowGraphNode*, std::vector<DataFlowGraphNode*>> node_groups;
  auto topo_order = graph->topological_order();
  auto& nodes_in_order = std::get<0>(topo_order);
  auto& edges_in_order = std::get<1>(topo_order);

  std::map<std::string, DataFlowGraphNode*> name2node;
  for (auto* n : graph->nodes()) {
    name2node[n->id()] = n->safe_as<DataFlowGraphNode>();
  }

  // process compute_at
  absl::flat_hash_map<const cinn::common::GraphNode*, uint32_t>
      node2score;  // record each node's score for sorting.
  int score = 0;
  for (auto* n : nodes_in_order) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    node2score[node] = score++;
    for (ComputeAtRelation& compute_at : node->stage->compute_ats()) {
      PADDLE_ENFORCE_EQ(compute_at.IsCompatible(node->stage.get()),
                        true,
                        ::common::errors::InvalidArgument(
                            "The registered ComputeAt is not compatible."));
      // check the endpoints of compute_at has data dependency.
      auto* node0 = node;
      if (name2node.count(compute_at.stage->id()) == 0) {
        continue;
        std::stringstream ss;
        ss << "Didn't find node with name " << compute_at.stage->id() << " !";
        PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      }
      auto* node1 = name2node[compute_at.stage->id()];
      VLOG(3) << "a -> b: " << node0->id() << " -> " << node1->id();

      DataFlowGraphNode::MergeGroup(node0, node1);
      // process single level of outlinks
      for (auto& outlink : node0->outlinks()) {
        if (IsBetween(outlink->sink(), node0, node1)) {
          DataFlowGraphNode::MergeGroup(
              node0, outlink->sink()->safe_as<DataFlowGraphNode>());
        }
      }

      // TODO(Superjomn) Consider the case node1 is a parent.
    }
  }
  // generate final groups.
  absl::flat_hash_map<DataFlowGraphNode* /*ancestor*/,
                      std::vector<DataFlowGraphNode*>>
      clusters;
  for (auto* n : nodes_in_order) {
    auto* node = n->safe_as<DataFlowGraphNode>();
    clusters[node->group_ancestor()].push_back(node);
  }
  std::vector<Group> groups;
  for (auto& item : clusters) {
    Group group;
    for (auto* c : item.second) {
      group.nodes.emplace_back(c);
    }
    groups.push_back(std::move(group));
  }
  auto group_order = TopoSortGroups(groups);
#ifdef CINN_DEBUG
  VLOG(2) << "Group Partition result:";
  int graph_node_count = 0;
  for (auto& group : group_order) {
    std::stringstream ss;
    for (auto& node : group.nodes) {
      ss << node->id() << " ";
    }
    VLOG(2) << "group: { " << ss.str() << " }";
    graph_node_count += group.nodes.size();
  }
  // check the groups contains all the nodes in graph.
  PADDLE_ENFORCE_EQ(
      graph_node_count,
      graph->nodes().size(),
      ::common::errors::InvalidArgument(
          "the groups should contain all the nodes in the graph"));
#endif

  return group_order;
}

}  // namespace detail

std::unique_ptr<Schedule> PolyScheduler::BuildSchedule() {
  std::unique_ptr<Schedule> res(new Schedule);

  // partition the DataFlowGraph to groups.
  auto dfg_groups = PartitionGroups(dfg_.get());
  PADDLE_ENFORCE_NE(
      dfg_groups.empty(),
      true,
      ::common::errors::InvalidArgument("DFG graph is empty! Please check."));

  // transform the DFG groups to schedule groups.
  PADDLE_ENFORCE_NE(schedule_graph_.nodes().empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Schedule graph is empty! Please check."));
  PADDLE_ENFORCE_EQ(schedule_graph_.nodes().size(),
                    dfg_->nodes().size(),
                    ::common::errors::InvalidArgument(
                        "DFG graph is not match schedule graph"));
  schedule_groups_.clear();
  for (auto& dfg_group : dfg_groups) {
    ScheduleGroup group;
    for (auto& node : dfg_group.nodes) {
      auto* schedule_node = schedule_graph_.RetrieveNode(node->id());
      PADDLE_ENFORCE_NOT_NULL(
          schedule_node,
          ::common::errors::InvalidArgument(
              "Missing node %s in schedule graph.", node->id()));
      group.nodes.push_back(schedule_node->safe_as<ScheduleGraphNode>());
    }
    schedule_groups_.emplace_back(std::move(group));
  }
  PADDLE_ENFORCE_EQ(schedule_groups_.size(),
                    dfg_groups.size(),
                    ::common::errors::InvalidArgument(
                        "The number of groups should be the same as the DFG "
                        "groups"));

  // Schedule each group
  ScheduleGroups();

  // Collect result.
  res->groups = schedule_groups_;

  for (auto& group : schedule_groups_) {
    for (auto& node : group.nodes) {
      res->schedule[node->id()] =
          node->time_schedule.to_isl(Context::isl_ctx());
    }
  }

  return res;
}

PolyScheduler::PolyScheduler(
    const std::vector<Stage*>& stages,
    const std::vector<std::pair<std::string, std::string>>& extra_links) {
  PADDLE_ENFORCE_NE(
      stages.empty(),
      true,
      ::common::errors::InvalidArgument("No stage is provided! Please check."));
  // collect extra links
  auto _extra_links = extra_links;
  if (extra_links.empty()) {
    _extra_links = ExtractExtraDepLinksFromStages(stages);
  }

  dfg_ = CreateGraph(stages, _extra_links);

  for (auto* stage : stages) {
    AddStage(*stage);
  }
  FinishStageAdd();
}

std::vector<detail::Group> PolyScheduler::PartitionGroups(
    DataFlowGraph* graph) {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      ::common::errors::InvalidArgument(
          "The DataFlowGraph pointer is null in PolyScheduler."));
  PADDLE_ENFORCE_NE(
      graph->nodes().empty(),
      true,
      ::common::errors::InvalidArgument("Graph is empty! Please check."));
  return detail::NaivePartitionGraph(graph);
}

void PolyScheduler::ScheduleAGroup(ScheduleGroup* group) {
  PADDLE_ENFORCE_NOT_NULL(
      group,
      ::common::errors::InvalidArgument(
          "The ScheduleGroup pointer is null in ScheduleAGroup."));
  PADDLE_ENFORCE_NE(
      group->nodes.empty(),
      true,
      ::common::errors::InvalidArgument("Group is empty! Please check."));

  // create scheduler for this group.
  std::vector<Stage*> stages;
  for (auto& node : group->nodes) {
    stages.push_back(const_cast<Stage*>(node->stage));
  }

  PolyGroupScheduler scheduler(stages);
  group->nodes = scheduler.Build();
  group->dimension_names = scheduler.detailed_dimension_names();
}

void PolyScheduler::ScheduleGroups() {
  PADDLE_ENFORCE_NE(schedule_groups_.empty(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "You should call PartitionGroups first."));
  for (auto& group : schedule_groups_) {
    ScheduleAGroup(&group);
  }
}

std::vector<Shared<ScheduleGraphNode>> PolyGroupScheduler::Build() {
  // consider compute_at
  std::map<std::string, Stage*> stage_map;
  std::map<std::string, ComputeAtRelation> compute_at_links;
  for (int i = 0; i < stages_.size(); i++) {
    auto& stage = stages_[i];
    stage_map[stage->tensor_->name] = stage;
    for (auto& item : stage->compute_ats()) {
      compute_at_links[stage->tensor_->name] = item;
    }
  }
  std::map<std::string, int> stage_level;
  for (auto& link : compute_at_links) {
    PADDLE_ENFORCE_NE(stage_map.count(link.first),
                      0,
                      ::common::errors::InvalidArgument(
                          "The stage  is not found in the stage map"));
    PADDLE_ENFORCE_NE(stage_map.count(link.second.stage->tensor_->name),
                      0,
                      ::common::errors::InvalidArgument(
                          "The stage  is not found in the stage map"));
    auto* a = stage_map.at(link.first);
    auto* b = stage_map.at(link.second.stage->tensor_->name);
    After(*a, *b, link.second.level);
    stage_level[a->id()] = link.second.level;
  }

  for (int i = 0; i < stages_.size() - 1; i++) {
    Stage* a = stages_[i];
    Stage* b = stages_[i + 1];

    auto a_set = a->transformed_domain();
    auto b_set = b->transformed_domain();

    // a -> b not in the compute_at_links
    if (!compute_at_links.count(a->tensor_->name) ||
        compute_at_links[a->tensor_->name].stage->tensor_->name !=
            b->tensor_->name) {
      int min_level = INT_MAX;
      if (stage_level.count(a->id()))
        min_level = std::min(min_level, stage_level[a->id()]);
      if (stage_level.count(b->id()))
        min_level = std::min(min_level, stage_level[b->id()]);
      if (min_level < INT_MAX) {
        After(*a, *b, min_level);
      }
    }
  }

  auto topo_order = schedule_graph_.topological_order();
  auto& nodes_in_order = std::get<0>(topo_order);
  auto& edges_in_order = std::get<1>(topo_order);
  std::vector<Shared<ScheduleGraphNode>> res;

  // update the time schedule info.
  for (auto& edge : edges_in_order) {
    auto* node0 = edge->source()->safe_as<ScheduleGraphNode>();
    auto* node1 = edge->sink()->safe_as<ScheduleGraphNode>();
    int level = edge->as<ScheduleGraphEdge>()->level;
    if (level < 0) continue;
    VLOG(2) << "schedule " << node0->id() << " -> " << node1->id() << " level "
            << level;
    node1->time_schedule.OrderAfter(node0->time_schedule, level);
  }

  for (auto& node : nodes_in_order) {
    res.emplace_back(node->safe_as<ScheduleGraphNode>());
  }
  return res;
}

PolyGroupScheduler::PolyGroupScheduler(const std::vector<Stage*>& stages)
    : stages_(stages) {
  PADDLE_ENFORCE_GT(stages.size(),
                    0,
                    ::common::errors::InvalidArgument("No stage is provided"));
  for (auto* stage : stages) {
    AddStage(*stage);
  }
  FinishStageAdd();
}

}  // namespace poly
}  // namespace cinn
