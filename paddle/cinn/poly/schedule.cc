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

#include "paddle/cinn/poly/schedule.h"

#include <deque>
#include <set>
#include <sstream>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/poly/naive_scheduler.h"
#include "paddle/cinn/poly/poly_scheduler.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace poly {

std::string TimeSchedule::__str__() const {
  CHECK_LE(time_dims_.size(), kMaxDims);

  // generate range: [dup, t0, t1...]
  std::vector<std::string> range_dims, cond_dims;
  range_dims.push_back("r");  // root level
  for (int i = 0; i < time_dims_.size(); i++) {
    range_dims.push_back("t" + std::to_string(i));
    range_dims.push_back("d" + std::to_string(i));
  }

  for (int i = 0; i < time_dims_.size(); i++) {
    cond_dims.push_back("d" + std::to_string(i));
    cond_dims.push_back("t" + std::to_string(i));
  }

  // generate conditions
  std::vector<std::string> conds;
  conds.push_back(utils::StringFormat("r=%d", root_time_));
  for (int i = 0; i < time_dims_.size(); i++) {
    conds.push_back(
        utils::StringFormat("%s=%s",
                            cond_dims[2 * i].c_str(),
                            std::to_string(time_dims_[i].time).c_str()));
    conds.push_back(utils::StringFormat(
        "%s=%s", cond_dims[2 * i + 1].c_str(), time_dims_[i].dim.c_str()));
  }

  return utils::StringFormat("{ %s[%s] -> [%s]: %s }",
                             id_.c_str(),
                             utils::Join(domain_dims, ", ").c_str(),
                             utils::Join(range_dims, ", ").c_str(),
                             utils::Join(conds, " and ").c_str());
}

std::vector<std::string> TimeSchedule::final_axis_names() const {
  std::vector<std::string> dims;
  for (int i = 0; i < time_dims_.size(); i++) {
    dims.push_back(std::to_string(time_dims_[i].time).c_str());
    dims.push_back(time_dims_[i].dim.c_str());
  }
  return dims;
}

TimeSchedule::TimeSchedule(const std::string &id,
                           const std::vector<std::string> &dims) {
  CHECK_LE(dims.size(), kMaxDims);
  id_ = id;
  domain_dims = dims;
  for (auto &dim : domain_dims) {
    CHECK(!dim.empty());
    time_dims_.emplace_back(dim, 0);
  }
}

void TimeSchedule::OrderAfter(const TimeSchedule &other, int level) {
  CHECK_EQ(space_size(), other.space_size()) << "space not match";
  CHECK_LT(level, other.space_size()) << other.__str__();
  CHECK_GE(level, 0);
  CHECK(!time_dims_.empty());

  root_time_ = std::max(root_time_, other.root_time_);

  if (level == -1) {
    root_time_ = std::max(root_time_, other.root_time_ + 1);
  }

  for (int i = 0; i < level; i++) {
    this->time_dims_[i].time =
        std::max(other.time_dims_[i].time, this->time_dims_[i].time);
  }

  this->time_dims_[level].time =
      std::max(this->time_dims_[level].time, other.time_dims_[level].time + 1);
}

isl::map TimeSchedule::to_isl(isl::ctx ctx) const {
  VLOG(4) << "isl: " << __str__();
  return isl::map(ctx, __str__());
}

const std::string &TimeSchedule::id() const {
  CHECK(!id_.empty());
  return id_;
}

void TimeSchedule::ResizeTimeSpace(int size) {
  CHECK_LE(size, kMaxDims);
  for (int i = time_dims_.size(); i < size; i++) {
    time_dims_.emplace_back("0", 0);
  }
}

/*
std::unique_ptr<Schedule> CreateSchedule(const ir::Tensor &tensor, ScheduleKind
schedule_kind) { auto stages = GatherStagesInTensors({tensor}); VLOG(3) <<
"collected " << stages.size() << " stages"; return CreateSchedule(stages,
schedule_kind);
}
 */

std::unique_ptr<Schedule> CreateSchedule(
    const std::vector<Stage *> &stages,
    ScheduleKind schedule_kind,
    const std::vector<std::pair<std::string, std::string>> &extra_links) {
  CHECK(!stages.empty());
  for (auto &stage : stages) {
    VLOG(4) << "stage: " << stage->domain();
  }
  switch (schedule_kind) {
    case ScheduleKind::Naive: {
      NaiveScheduler scheduler(stages);
      return scheduler.BuildSchedule();
    } break;
    case ScheduleKind::Poly: {
      PolyScheduler scheduler(stages, extra_links);
      return scheduler.BuildSchedule();
    } break;
    default:
      CINN_NOT_IMPLEMENTED
  }
  return nullptr;
}

std::map<std::string, isl::map> CollectScheduleMapFromGroup(
    const ScheduleGroup &group) {
  std::map<std::string, isl::map> map;

  std::vector<Stage *> stages;
  for (auto &node : group.nodes) {
    CHECK(node->stage);
    stages.push_back(node->stage);
  }

  PolyGroupScheduler group_scheduler(stages);
  group_scheduler.Build();

  return group_scheduler.schedule_map();
}

void SchedulerBase::AddStage(const Stage &x) {
  CHECK(!registration_finalized_) << "element registration has been finalized.";
  space_size_ =
      std::max(space_size_, isl_map_dim(x.transform().get(), isl_dim_out));
  VLOG(3) << "space_size: " << space_size_;
  VLOG(3) << "schedule: " << x.transform();

  // Use the dimensions from element's schedule's range as the new domain
  // dimensions because in Element, the schedule is like '{ S0[i,j] ->
  // S0[i_outer, i_inner, j] }', the scheduler should schedule base on the
  // range.
  auto dims = isl_get_dim_names(x.transform(), isl_dim_out);
  std::string id = isl_map_get_tuple_name(x.transform().get(), isl_dim_in);
  schedule_graph_.RegisterNode(
      x.id(),
      common::make_shared<ScheduleGraphNode>(
          id, isl_get_dim_names(x.transform(), isl_dim_out), &x));

  // record the longest dimensions.
  if (dims.size() > detailed_dimension_names_.size())
    detailed_dimension_names_ = dims;

  if (!ctx_.get()) {
    ctx_ = x.domain().ctx();
  } else {
    CHECK_EQ(ctx_.get(), x.domain().ctx().get()) << "isl ctx not match";
  }
}

void SchedulerBase::FinishStageAdd() {
  for (auto *node : schedule_graph_.nodes()) {
    auto *schedule_node = node->safe_as<ScheduleGraphNode>();
    for (auto &depend : schedule_node->stage->ctrl_depends()) {
      auto *depend_node = schedule_graph_.RetrieveNode(depend->name);
      if (depend_node) {  // some dependencies might be in another graph.
        auto *a_node = depend_node->safe_as<ScheduleGraphNode>();
        auto *b_node = node->safe_as<ScheduleGraphNode>();
        auto _a_edge_b_edge_ = a_node->LinkTo<ScheduleGraphEdge>(
            b_node);  // Add link from extra depend statment to current node.
        auto &a_edge = std::get<0>(_a_edge_b_edge_);
        auto &b_edge = std::get<1>(_a_edge_b_edge_);
        a_edge->as<ScheduleGraphEdge>()->level = -1;
        b_edge->as<ScheduleGraphEdge>()->level = -1;
      }
    }
  }

  CHECK(!schedule_graph_.nodes().empty())
      << "No node is registered to the graph, use RegisterElement to collect "
         "some elements";
  registration_finalized_ = true;

  for (auto &item : schedule_graph_.nodes()) {
    VLOG(6)
        << "original dims in time_schedule: "
        << utils::Join(
               item->safe_as<ScheduleGraphNode>()->time_schedule.domain_dims,
               ", ");
    item->safe_as<ScheduleGraphNode>()->time_schedule.ResizeTimeSpace(
        space_size_);
  }
}

std::vector<std::string> SchedulerBase::WrapIteratorNames(
    const std::vector<std::string> &names) {
  std::vector<std::string> res;
  for (int i = 0; i < names.size(); i++) {
    res.push_back("");        // fake name for time space.
    res.push_back(names[i]);  // name for the corresponding iterator.
  }
  return res;
}

SchedulerBase &SchedulerBase::After(const Stage &a, const Stage &b, int level) {
  CHECK_LT(level, space_size_);
  auto *a_node =
      schedule_graph_.RetrieveNode(a.id())->safe_as<ScheduleGraphNode>();
  auto *b_node =
      schedule_graph_.RetrieveNode(b.id())->safe_as<ScheduleGraphNode>();
  CHECK(a_node) << "no node called " << a.id() << " registered in the graph";
  CHECK(b_node) << "no node called " << b.id() << " registered in the graph";

  auto _a_edge_b_edge_ = a_node->LinkTo<ScheduleGraphEdge>(b_node);  // NOLINT
  auto &a_edge = std::get<0>(_a_edge_b_edge_);
  auto &b_edge = std::get<1>(_a_edge_b_edge_);
  a_edge->as<ScheduleGraphEdge>()->level = level;
  b_edge->as<ScheduleGraphEdge>()->level = level;
  VLOG(2) << "In After, Set [" << a.id() << "] -> [b: ]" << b.id()
          << "] with level = " << level;
  return *this;
}

SchedulerBase &SchedulerBase::Before(const Stage &a,
                                     const Stage &b,
                                     int level) {
  return After(b, a, level);
}

std::map<std::string, isl::map> SchedulerBase::schedule_map() const {
  std::map<std::string, isl::map> res;
  for (auto &node : schedule_graph_.nodes()) {
    auto *schedule_node = node->safe_as<ScheduleGraphNode>();
    res[schedule_node->id()] =
        schedule_node->time_schedule.to_isl(Context::isl_ctx());
  }
  return res;
}

const char *ScheduleGraphNode::__type_info__ = "ScheduleGraphNode";

}  // namespace poly
}  // namespace cinn
