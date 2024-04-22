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
#include <absl/container/flat_hash_map.h>

#include <map>
#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/schedule_desc.pb.h"
#include "paddle/cinn/utils/registry.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace ir {

// A ScheduleDesc describe the scheduling process of an ir::ModuleExpr, it
// records all transform/getting operations executed by a corresponding
// ir::IRSchedule. A ScheduleDesc can be serialized to JSON format and saved to
// file. For deserializing, it can be re-applied to a new IRSchedule that is
// initialized by a semantics-equal original ir::ModuleExpr, and then achieves
// the same result.

class IRSchedule;  // forward declaration to avoid cross-reference
class ScheduleDesc {
 public:
  // each operation executed through IRSchedule is recorded as a step
  struct Step {
    std::string type;  // step name
    absl::flat_hash_map<std::string, std::vector<Expr>> inputs;
    utils::AttributeMap attrs;
    std::vector<Expr> outputs;
    Step() = default;
    Step(std::string type_i,
         absl::flat_hash_map<std::string, std::vector<Expr>> inputs_i,
         utils::AttributeMap attrs_i,
         std::vector<Expr> outputs_i)
        : type(type_i), inputs(inputs_i), attrs(attrs_i), outputs(outputs_i) {}
  };

  /**
   * \brief Re-applied a scheduling process represented as a proto::ScheduleDesc
   * to a new IRSchedule object.
   * @param desc_proto The proto of the ScheduleDesc to be re-applied.
   * @param sch The original IRSchedule to be replayed the description on.
   * @param without_post_schedule Determine whether to delete the post
   * schedules.
   */
  static std::vector<Expr> ReplayWithProto(
      const proto::ScheduleDesc& desc_proto,
      IRSchedule* sch,
      bool without_post_schedule = false);

  ScheduleDesc() = default;

  explicit ScheduleDesc(const std::vector<Step>& steps) : steps_(steps) {}

  explicit ScheduleDesc(std::vector<Step>&& steps) : steps_(steps) {}

  // Append a new step
  void Append(Step&& step);

  // Pop the last step
  void Pop();

  /**
   * \brief Replay this description to a new IRSchedule that is initialized by a
   * semantics-equal original ModuleExpr.
   * @param schedule The original IRSchedule to be replayed the description on.
   * @param without_post_schedule Determine whether to delete the post
   * schedules.
   */
  void Replay(IRSchedule* schedule, bool without_post_schedule = false) const;

  // convert to a proto::ScheduleDesc object
  proto::ScheduleDesc ToProto() const;

  // return detail string of a ScheduleDesc for debug;
  std::string DebugString() const { return ToProto().DebugString(); }

  std::vector<Step> Steps() const { return steps_; }

  bool Empty() const { return steps_.empty(); }

  /**
   * \brief Fork this ScheduleDesc and update a step of the new ScheduleDesc
   * with a new decision.
   * @param step_idx The index of the step to be update.
   * @param decision The new decision.
   * @param without_post_schedule Determine whether to delete the post
   * schedules.
   * @return The new ScheduleDesc.
   */
  ScheduleDesc ForkAndUpdate(int step_idx,
                             utils::Attribute decision,
                             bool without_post_schedule) const;

 private:
  std::vector<Step> steps_;  // all operations are recorded in order.
};

}  // namespace ir
}  // namespace cinn
