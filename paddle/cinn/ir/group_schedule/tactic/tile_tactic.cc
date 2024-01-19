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
  // fake strategy
  auto GetFirstFactor = [](int num) {
    if (num == 1) return 1;
    int factor = 1;
    for (int i = num - 1; i >= 1; --i) {
      if (num % i == 0) {
        return i;
      }
    }
  };

  bool has_rb_iter = !context_->iter_space_info.rb_space.empty();
  bool has_sp_iter = !context_->iter_space_info.sp_space.empty();
  VLOG(6) << "has_sp_iter = " << has_sp_iter
          << ", has_rb_iter = " << has_rb_iter;
  context_->iter_space_info.rb_space.clear();
  context_->iter_space_info.sp_space.clear();

  if (has_sp_iter) {
    int sp_factor = GetFirstFactor(context_->bucket_info.sp_lower_bound);
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(context_->bucket_info.sp_lower_bound / sp_factor),
        has_rb_iter ? IterativeSpaceInfo::AxisType::kCudaBlockY
                    : IterativeSpaceInfo::AxisType::kCudaBlockX);
    VLOG(6) << "sp_space: <"
            << std::get<0>(context_->iter_space_info.sp_space.back())
            << ", AxisType["
            << static_cast<int>(
                   std::get<1>(context_->iter_space_info.sp_space.back()))
            << "]>";
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(sp_factor),
        has_rb_iter ? IterativeSpaceInfo::AxisType::kCudaBlockX
                    : IterativeSpaceInfo::AxisType::kCudaThreadX);
    VLOG(6) << "sp_space: <"
            << std::get<0>(context_->iter_space_info.sp_space.back())
            << ", AxisType["
            << static_cast<int>(
                   std::get<1>(context_->iter_space_info.sp_space.back()))
            << "]>";
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kSerial);
    VLOG(6) << "sp_space: <"
            << std::get<0>(context_->iter_space_info.sp_space.back())
            << ", AxisType["
            << static_cast<int>(
                   std::get<1>(context_->iter_space_info.sp_space.back()))
            << "]>";
  }

  if (has_rb_iter) {
    context_->iter_space_info.rb_space.emplace_back(
        ir::Expr(context_->bucket_info.rb_lower_bound),
        IterativeSpaceInfo::AxisType::kCudaThreadX);
    VLOG(6) << "rb_space: <"
            << std::get<0>(context_->iter_space_info.rb_space.back())
            << ", AxisType["
            << static_cast<int>(
                   std::get<1>(context_->iter_space_info.rb_space.back()))
            << "]>";
    context_->iter_space_info.rb_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kSerial);
    VLOG(6) << "rb_space: <"
            << std::get<0>(context_->iter_space_info.rb_space.back())
            << ", AxisType["
            << static_cast<int>(
                   std::get<1>(context_->iter_space_info.rb_space.back()))
            << "]>";
  }
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
