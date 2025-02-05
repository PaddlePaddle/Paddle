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

#pragma once

#include <string>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule_block_graph.h"

namespace cinn {
namespace ir {

struct IterativeSpaceInfo {
  enum class AxisType : int {
    kSerial = 0,
    kCudaThreadX = 1,
    kCudaThreadY = 2,
    kCudaThreadZ = 3,
    kCudaBlockX = 4,
    kCudaBlockY = 5,
    kCudaBlockZ = 6,
  };
  // pure spatial iterative space
  std::vector<std::tuple<ir::Expr, AxisType>> sp_space;
  // reduce or broadcast iterative space
  std::vector<std::tuple<ir::Expr, AxisType>> rb_space;
  // total sp extent
  ir::Expr total_sp_extent;
  // total rb extent
  ir::Expr total_rb_extent;
  // original loop order with same iteration order as the memory order
  std::vector<std::pair<std::string, ir::Expr>> memory_consistent_order_space;
  // memory consistent order space info with merging continuous same type
  // [S: 16, S: a, R: 32] -> [S: 16 * a, R: 32]
  std::vector<std::pair<std::string, ir::Expr>>
      memory_consistent_order_homogeneous_merged_space;
  // index that transform from memory consistent order to rb last order
  // for example:
  // the memory consistent order axis is [A, B, C], and the B axis is reduce，
  // the rb last order axis is [A, C, B], and rb_last_order is [0, 2, 1].
  std::vector<int> rb_last_order;

  std::string PrintIterSpace() const {
    std::stringstream ss;
    ss << "[sp space]: ";
    for (const auto& axis : sp_space) {
      ss << "<" << std::get<0>(axis) << ", AxisType = ["
         << static_cast<int>(std::get<1>(axis)) << "]>  ";
    }
    ss << "\n[rb space]: ";
    for (const auto& axis : rb_space) {
      ss << "<" << std::get<0>(axis) << ", AxisType = ["
         << static_cast<int>(std::get<1>(axis)) << "]>  ";
    }
    ss << "\n[memory_consistent_order_space]: [";
    for (const auto& item : memory_consistent_order_space) {
      ss << item.first << "(" << item.second << "), ";
    }
    ss << "] ";
    ss << "\n[memory_consistent_order_homogeneous_merged_space]: [";
    for (const auto& item : memory_consistent_order_homogeneous_merged_space) {
      ss << item.first << "(" << item.second << "), ";
    }
    ss << "] ";
    return ss.str();
  }
};

struct ScheduleContext {
  std::unordered_set<std::string> output_names;
  Target target;
  // TODO(liangshuhao): this struct is deprecated and will be removed later.
  IterativeSpaceInfo iter_space_info;
  BucketInfo bucket_info;
  ScheduleConfig config;
};

class ScheduleTactic {
 public:
  // Attribute key to record which tile tactic has been applied on a graph.
  // Exactly one tile tactic is applied on a graph during scheduling.
  static constexpr char* kTileMethod = "tile_method";

  virtual void Init(ScheduleContext* context) {
    PADDLE_THROW(::common::errors::Unimplemented(
        "ScheduleTactic subclass must implement one of the Init method."));
  }
  virtual void Init(ScheduleContext* context, ir::IRSchedule* sch) {
    Init(context);
  }

  virtual void Apply(ir::IRSchedule* sch, const std::string& block_id) = 0;

  virtual std::string TacticName() const = 0;
};

}  // namespace ir
}  // namespace cinn
