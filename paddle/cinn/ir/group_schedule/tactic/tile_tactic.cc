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
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace ir {

class TileTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileTactic"; }

 private:
  ScheduleContext* context_;
};

void TileTactic::Init(ScheduleContext* context) {
  context_ = context;
  // TODO(BiynXu): Create schedule config and bucket info based on hardware
  // information, and get the config here. now it is a naive strategy.
  auto GetFirstFactor = [](int64_t num) -> int64_t {
    if (num == 1) return 1;
    int factor = 1;
    for (int64_t i = num - 1; i >= 1; --i) {
      if (num % i == 0) {
        return i;
      }
    }
  };
  auto GetTreeReduceSize = [&](const ir::Expr& total_rb_extent) -> int64_t {
    const int64_t max_num_threads =
        cinn::common::DefaultDeviceTarget().max_num_threads();
    int64_t nums_thread_per_block = max_num_threads;
    if (total_rb_extent.is_constant()) {
      int64_t extent = static_cast<int64_t>(total_rb_extent.get_constant());
      nums_thread_per_block = GetFirstFactor(extent);
    } else {
      if (context->bucket_info.space.size() == 2 &&
          context->bucket_info.space[1].iter_type == "R") {
        nums_thread_per_block = context_->bucket_info.space[1].lower_bound;
      } else {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Now, the function GetTreeReduceSize doesn't support the cases "
            "except SR"));
      }
    }
    return nums_thread_per_block > max_num_threads ? max_num_threads
                                                   : nums_thread_per_block;
  };
  auto GetNumThreadPerBlock = [&](int64_t lower_bound) -> int64_t {
    // When designing the tile config, we can further subdivided.
    if (lower_bound >= 1024) {
      return 256;
    } else if (lower_bound >= 256) {
      return 32;
    } else {
      return 4;
    }
  };

  bool has_rb_iter = !context_->iter_space_info.rb_space.empty();
  bool has_sp_iter = !context_->iter_space_info.sp_space.empty();
  VLOG(6) << "has_sp_iter = " << has_sp_iter
          << ", has_rb_iter = " << has_rb_iter;
  context_->iter_space_info.rb_space.clear();
  context_->iter_space_info.sp_space.clear();

  // naive strategy
  if (has_rb_iter) {
    // If there is a reduce dimension.
    // Bind all spatial axis on cuda block.
    context_->iter_space_info.sp_space.emplace_back(
        context_->iter_space_info.total_sp_extent,
        IterativeSpaceInfo::AxisType::kCudaBlockX);
    // Bind first part of reduce axis on cuda thread to do tree form reduction.
    context_->iter_space_info.rb_space.emplace_back(
        ir::Expr(GetTreeReduceSize(context_->iter_space_info.total_rb_extent)),
        IterativeSpaceInfo::AxisType::kCudaThreadX);
    // The rest part of reduce axis will be executed serially.
    context_->iter_space_info.rb_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kSerial);
  } else {
    // If there is no reduce dimension.
    // Divide the spatial space into two parts, one bound to cuda block and the
    // other bound to cuda thread.
    context_->iter_space_info.sp_space.emplace_back(
        ir::Expr(-1), IterativeSpaceInfo::AxisType::kCudaBlockX);
    if (context->bucket_info.space.size() == 2 &&
        context->bucket_info.space[1].iter_type == "R") {
      context_->iter_space_info.sp_space.emplace_back(
          ir::Expr(
              GetNumThreadPerBlock(context_->bucket_info.space[1].upper_bound)),
          IterativeSpaceInfo::AxisType::kCudaThreadX);
    } else {
      PADDLE_THROW(::common::errors::Unimplemented(
          "Now, the function GetTreeReduceSize doesn't support the cases "
          "except SR"));
    }
  }
  VLOG(6) << context_->iter_space_info.PrintIterSpace();
}

void TileTactic::Apply(ir::IRSchedule* sch, const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  PADDLE_ENFORCE_EQ((loops.size() == 1 || loops.size() == 2),
                    true,
                    ::common::errors::InvalidArgument(
                        "All loops must be unified as sp_loop or "
                        "rb_loop. Current loop size: %d.",
                        loops.size()));
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

std::unique_ptr<ScheduleTactic> CreateTileTactic() {
  return std::make_unique<TileTactic>();
}

}  // namespace ir
}  // namespace cinn
