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

#include "paddle/cinn/ir/group_schedule/tactic/tile_tactic.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

void TileTactic::Init(ScheduleContext* context) {
  context_ = context;
  // TODO(BiynXu): Create schedule config and bucket info based on hardware
  // information, and get the config here. naive strategy
  auto GetFirstFactor = [](int64_t num) -> int64_t {
    if (num == 1) return 1;
    int factor = 1;
    for (int64_t i = num - 1; i >= 1; --i) {
      if (num % i == 0) {
        return i;
      }
    }
  };
  auto GetTreeReduceSize = [&](const ir::Expr& total_rb_extent) -> ir::Expr {
    if (total_rb_extent.is_constant()) {
      int64_t extent = static_cast<int64_t>(total_rb_extent.get_constant());
      return ir::Expr(GetFirstFactor(extent));
    }
    return ir::Expr(context_->bucket_info.rb_lower_bound);
  };

  bool has_rb_iter = !context_->iter_space_info.rb_space.empty();
  bool has_sp_iter = !context_->iter_space_info.sp_space.empty();
  VLOG(6) << "has_sp_iter = " << has_sp_iter
          << ", has_rb_iter = " << has_rb_iter;
  context_->iter_space_info.rb_space.clear();
  context_->iter_space_info.sp_space.clear();

  if (has_rb_iter) {
    context_->iter_space_info.sp_space.emplace_back(
        context_->iter_space_info.total_sp_extent,
        IterativeSpaceInfo::AxisType::kCudaBlockX);
    context_->iter_space_info.rb_space.emplace_back(
        GetTreeReduceSize(context_->iter_space_info.total_rb_extent),
        IterativeSpaceInfo::AxisType::kCudaThreadX);
    context_->iter_space_info.rb_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kSerial);
  } else {
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kCudaBlockX);
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(512), IterativeSpaceInfo::AxisType::kCudaThreadX);
  }
  VLOG(6) << context_->iter_space_info.PrintIterSpace();
}

void TileTactic::Apply(ir::IRSchedule* sch, const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  CHECK(loops.size() == 1 || loops.size() == 2)
      << "All loops must be unified as sp_loop or rb_loop.";
  if (loops.size() == 2) {
    std::vector<ir::Expr> rb_factors;
    for (const auto& axis : context_->iter_space_info.rb_space) {
      rb_factors.push_back(std::get<0>(axis));
    }
    sch->Split(loops[1], rb_factors);
    loops = sch->GetLoops(block_id);
    VLOG(6) << "after split rb loop of " << block_id << ": "
            << sch->GetModule().GetExprs()[0];
  }
  std::vector<ir::Expr> sp_factors;
  for (const auto& axis : context_->iter_space_info.sp_space) {
    sp_factors.push_back(std::get<0>(axis));
  }
  sch->Split(loops[0], sp_factors);
  VLOG(6) << "after split sp loop of " << block_id << ": "
          << sch->GetModule().GetExprs()[0];
}

}  // namespace ir
}  // namespace cinn
