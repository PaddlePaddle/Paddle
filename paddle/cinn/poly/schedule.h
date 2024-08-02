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

#pragma once
/**
 * This file defines Schedule related concepts.
 */
#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/poly/graph.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/map.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace poly {

/**
 * The dimension with time space.
 */
struct TimeDim {
  //! time of this dimension.
  int time;
  //! name of this dimension.
  std::string dim;

  TimeDim() = default;
  TimeDim(const std::string &dim, int time) : dim(dim), time(time) {
    PADDLE_ENFORCE_EQ(
        !dim.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The dimension is empty. Please provide a valid dimension."));
  }
};

class ScheduleGraphNode;
struct ScheduleGraph : public cinn::common::Graph {};

/**
 * ISL schedule map with time space, used to generate the final schedule.
 * The map it generates is like { [x,y] -> [t0,x,t1,y] }, the t0 and t1 are time
 * space.
 */
struct TimeSchedule {
  TimeSchedule(const std::string &id, const std::vector<std::string> &dims);

  void ResizeTimeSpace(int size);

  //! Schedule this after \p other in \p level.
  void OrderAfter(const TimeSchedule &other, int level);

  //! How many dimensions of this time schedule space.
  size_t space_size() const { return time_dims_.size(); }

  //! The unique ID of the time schedule.
  const std::string &id() const;

  //! Get the isl map.
  isl::map to_isl(isl::ctx ctx) const;

  //! ISL range format, such as '[dup, t0, t1]: dup=0 and t0=0 and t1=i]'
  std::string __str__() const;

  //! Get the axis names with the original dimension names and faked time
  //! dimensions.
  std::vector<std::string> final_axis_names() const;

  std::vector<std::string> domain_dims;
  int duplicate_id{};

  constexpr static int kMaxDims = 50;

 private:
  int root_time_{0};
  std::vector<TimeDim> time_dims_;
  std::string id_;
};

struct ScheduleGroup;
/**
 * A container type to contain the schedule information of a graph(several
 * groups).
 */
struct Schedule {
  //! The schedule groups partitioned from the graph.
  std::vector<ScheduleGroup> groups;
  //! id to the isl schedule for each node.
  std::map<std::string, isl::map> schedule;
};

/**
 * The base class for all the Scheduler, it helps to schedule the nodes in a
 * group(isl space). All the schedule in the same group should have the same
 * number of dimensions, and each have some dependency with others.
 */
class SchedulerBase {
 public:
  /**
   * Wrap the iterator names with time space fake names, it is used for isl AST
   * to set iterator names.
   * @param names the original iterator names.
   * @return the iterator names with time space included.
   */
  static std::vector<std::string> WrapIteratorNames(
      const std::vector<std::string> &names);

  /**
   * Mark this should schedule after another.
   *
   * @param b
   * @param level
   */
  SchedulerBase &After(const Stage &a, const Stage &b, int level);
  /**
   * Mark this should schedule before another.
   * @param b
   * @param level
   */
  SchedulerBase &Before(const Stage &a, const Stage &b, int level);

  std::map<std::string, isl::map> schedule_map() const;

  const std::vector<std::string> &detailed_dimension_names() const {
    return detailed_dimension_names_;
  }

 protected:
  /**
   * Register an Element to the scheduler.
   */
  void AddStage(const Stage &x);

  /**
   * Finalize the registration.
   */
  void FinishStageAdd();

  /**
   * Tell whether the registration is finalized.
   */
  bool finalized() const { return registration_finalized_; }
  int space_size() const { return space_size_; }

 protected:
  /**
   * The polyhedral schedule, any schedule is performed on it.
   * We use the time-space map to record the schedule information, the format is
   * borrowed from Tiramisu project: [time,dim,time,dim,time,dim ...]
   */
  int space_size_{0};
  mutable isl::ctx ctx_{Context::isl_ctx()};
  mutable ScheduleGraph schedule_graph_;
  // Record the longest dimensions(of some stage) to be the final detailed
  // dimension names. It might be used for ISL AST to set iterator names and
  // generate readable code.
  mutable std::vector<std::string> detailed_dimension_names_;

 private:
  bool registration_finalized_{false};
};

/**
 * Schedule Kind.
 */
enum class ScheduleKind {
  //! Basic strategy, each status is scheduled separately.
  Naive = 0,
  //! The strategy with iteration domain considered.
  Poly = 1,
};

//! Create a schedule from a tensor.
// std::unique_ptr<Schedule> CreateSchedule(const ir::Tensor &tensor,
// ScheduleKind schedule_kind = ScheduleKind::Poly);
//! Create a schedule from a list of stages, it will schedule the stages using
//! the information from data dependency, iteration domains.
std::unique_ptr<Schedule> CreateSchedule(
    const std::vector<Stage *> &stages,
    ScheduleKind schedule_kind = ScheduleKind::Poly,
    const std::vector<std::pair<std::string, std::string>> &extra_links = {});

/**
 * Gather the stages in the input tensors and their dependencies
 * @param xs The input tensors.
 * @param with_placeholder Whether to include placeholders(default false).
 * @returns The stages in topological order follow the connection to `xs`.
 */
// std::vector<Stage *> GatherStagesInTensors(const std::vector<ir::Tensor> &xs,
// bool with_placeholder = false);

struct ScheduleGraphEdge : public cinn::common::GraphEdge {
  ScheduleGraphEdge(cinn::common::GraphNode *a, cinn::common::GraphNode *b)
      : cinn::common::GraphEdge(a, b) {}

  //! Dependency level.
  int level{-1};
};

/**
 * Node in the schedule graph.
 */
struct ScheduleGraphNode : public cinn::common::GraphNode {
  TimeSchedule time_schedule;
  Stage *stage{};

  //! NOTE this id is not human-readable.
  // std::string id() const override { return
  // std::to_string(reinterpret_cast<size_t>(this)); }
  std::string id() const override { return time_schedule.id(); }

  explicit ScheduleGraphNode(const std::string &id,
                             const std::vector<std::string> &dims,
                             const Stage *stage)
      : time_schedule(id, dims), stage(const_cast<Stage *>(stage)) {}

  const char *type_info() const override { return __type_info__; }

  static const char *__type_info__;
};

struct ScheduleGroup {
  std::vector<Shared<ScheduleGraphNode>> nodes;
  std::vector<std::string> dimension_names;
};

std::map<std::string, isl::map> CollectScheduleMapFromGroup(
    const ScheduleGroup &group);

}  // namespace poly
}  // namespace cinn
