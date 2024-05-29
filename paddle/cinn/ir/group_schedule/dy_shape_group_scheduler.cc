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

#include "paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/common/enforce.h"

PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace ir {

void DynamicShapeGroupScheduler::Init() {
  VLOG(4) << "=============================Start group "
             "schedule==============================";
  VLOG(4) << "original group func body: \n"
          << ir_sch_->GetModule().GetExprs()[0];
  InitBuckets();
  tactics_.emplace_back(CreateTileFirstGeneralTactic());
  VLOG(4) << "CreateTileFirstGeneralTactic End";
  tactics_.emplace_back(CreateComputeInlineTactic());
  VLOG(4) << "CreateTileCreateComputeInlineTactic End";
}

void DynamicShapeGroupScheduler::InitBuckets() {
  std::unordered_set<std::string> output_names = OutputTensorNames();

  auto OutOfRange =
      [](ir::Expr extent, int lower_bound, int upper_bound) -> bool {
    if (!extent.is_constant()) return false;
    int extent_value = static_cast<int>(extent.get_constant());
    VLOG(5) << "extent_value: " << extent_value
            << ",lower_bound: " << lower_bound
            << ",upper_bound: " << upper_bound;
    if (extent_value < lower_bound || extent_value > upper_bound) {
      return true;
    }
    return false;
  };

  auto InitBucket = [&](BucketInfo&& bucket_info, ScheduleConfig&& config) {
    std::unique_ptr<ir::IRSchedule> ir_sch =
        std::make_unique<ir::IRSchedule>(*ir_sch_);
    std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph =
        std::make_unique<ir::ScheduleBlockGraph>(*ir_sch);
    ir::ScheduleBlockNode* global_master =
        FindGlobalMasterNode(schedule_block_graph);
    IterativeSpaceInfo iter_space_info = ConstructIterSpaceInfo(global_master);
    VLOG(4) << "iter_space_info.total_sp_extent: "
            << iter_space_info.total_sp_extent;
    VLOG(4) << "iter_space_info.total_rb_extent: "
            << iter_space_info.total_rb_extent;
    VLOG(4) << "bucket_info.sp_lower_bound: " << bucket_info.sp_lower_bound;
    VLOG(4) << "bucket_info.sp_upper_bound: " << bucket_info.sp_upper_bound;
    VLOG(4) << "bucket_info.rb_lower_bound: " << bucket_info.rb_lower_bound;
    VLOG(4) << "bucket_info.rb_upper_bound: " << bucket_info.rb_upper_bound;
    if (OutOfRange(iter_space_info.total_sp_extent,
                   bucket_info.sp_lower_bound,
                   bucket_info.sp_upper_bound) ||
        OutOfRange(iter_space_info.total_rb_extent,
                   bucket_info.rb_lower_bound,
                   bucket_info.rb_upper_bound)) {
      VLOG(4) << "Out of range";
      return;
    }
    SymbolicPredicate sp_lower_bound_predicate = ir::GE::Make(
        iter_space_info.total_sp_extent, ir::Expr(bucket_info.sp_lower_bound));
    SymbolicPredicate sp_upper_bound_predicate = ir::LE::Make(
        iter_space_info.total_sp_extent, ir::Expr(bucket_info.sp_upper_bound));
    SymbolicPredicate rb_lower_bound_predicate = ir::GE::Make(
        iter_space_info.total_rb_extent, ir::Expr(bucket_info.rb_lower_bound));
    SymbolicPredicate rb_upper_bound_predicate = ir::LE::Make(
        iter_space_info.total_rb_extent, ir::Expr(bucket_info.rb_upper_bound));
    SymbolicPredicate sp_predicate =
        ir::And::Make(sp_lower_bound_predicate, sp_upper_bound_predicate);
    SymbolicPredicate rb_predicate =
        ir::And::Make(rb_lower_bound_predicate, rb_upper_bound_predicate);
    SymbolicPredicate predicate = ir::And::Make(sp_predicate, rb_predicate);
    ScheduleContext schedule_context{output_names,
                                     target_,
                                     std::move(iter_space_info),
                                     std::move(bucket_info),
                                     std::move(config)};
    BucketContext bucket_context{std::move(predicate),
                                 std::move(ir_sch),
                                 std::move(schedule_block_graph),
                                 std::move(schedule_context)};
    bucket_contexts_.emplace_back(std::move(bucket_context));
  };

  ScheduleConfigManager& schedule_config_manager =
      ScheduleConfigManager::Instance();
  std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash> configs =
      schedule_config_manager.ExtractConfigs(target_, group_info_);
  for (std::pair<BucketInfo, ScheduleConfig>&& config : configs) {
    InitBucket(std::move(config.first), std::move(config.second));
  }
}

void DynamicShapeGroupScheduler::Schedule() {
  VLOG(4) << "bucket_context_.size() = " << bucket_contexts_.size();
  for (BucketContext& bucket_context : bucket_contexts_) {
    VLOG(4) << "===========================Apply tactics on Bucket ["
            << bucket_context.predicate << "]==========================";
    ApplyTactics(&bucket_context);
  }
}

void DynamicShapeGroupScheduler::ApplyTactics(BucketContext* bucket_context) {
  bucket_context->schedule_block_graph->Update(*(bucket_context->ir_sch));
  for (const auto& tactic : tactics_) {
    VLOG(5) << "[Start " << tactic->TacticName() << "] func body:\n"
            << bucket_context->ir_sch->GetModule().GetExprs().front();
    auto ApplyTacticFunc = [&](ir::ScheduleBlockNode* node) {
      VLOG(6) << "before applying [" << tactic->TacticName()
              << "] on ScheduleBlockNode [" << node->id() << "] func body:\n"
              << bucket_context->ir_sch->GetModule().GetExprs().front();
      tactic->Apply(bucket_context->ir_sch.get(), node->id());
      VLOG(6) << "after applying [" << tactic->TacticName()
              << "] on ScheduleBlockNode [" << node->id() << "] func body:\n"
              << bucket_context->ir_sch->GetModule().GetExprs().front();
    };
    tactic->Init(&(bucket_context->schedule_context));
    bucket_context->schedule_block_graph->DFSTopoWalk(ApplyTacticFunc);
    bucket_context->schedule_block_graph->Update(*(bucket_context->ir_sch));
    VLOG(5) << "[End " << tactic->TacticName() << "] func body: "
            << bucket_context->ir_sch->GetModule().GetExprs().front();
  }
}

std::vector<std::pair<SymbolicPredicate, ir::Expr>>
DynamicShapeGroupScheduler::GetIRs() {
  std::vector<std::pair<SymbolicPredicate, ir::Expr>> irs;
  for (BucketContext& context : bucket_contexts_) {
    irs.emplace_back(context.predicate,
                     context.ir_sch->GetModule().GetExprs()[0]);
  }
  return irs;
}

IterativeSpaceInfo DynamicShapeGroupScheduler::ConstructIterSpaceInfo(
    ScheduleBlockNode* node) {
  VLOG(5) << "global master: " << node->id();
  IterativeSpaceInfo info;
  std::vector<int> sp_iter_indices;
  std::vector<int> rb_iter_indices;

  ir::Expr block = node->Block();
  std::vector<ir::Expr> iter_values =
      block.As<ir::ScheduleBlockRealize>()->iter_values;
  std::vector<ir::Var> iter_vars = block.As<ir::ScheduleBlockRealize>()
                                       ->schedule_block.As<ir::ScheduleBlock>()
                                       ->iter_vars;
  std::vector<ir::Expr> loops = node->GetLoops();
  std::unordered_set<ir::Var> reduce_iter_vars =
      analyzer::GetReduceIterVars(block);
  std::unordered_map<ir::Var, ir::Expr> iter_var2value =
      analyzer::GetIterVarToValueOfSBlock(block);

  // init iter info
  if (!reduce_iter_vars.empty()) {
    std::set<ir::Expr> reduce_loads = ir::ir_utils::CollectIRNodesWithoutTensor(
        block,
        [&](const ir::Expr* x) {
          bool find_reduce_var = false;
          if (x->As<ir::Load>()) {
            for (ir::Expr index : x->As<ir::Load>()->indices) {
              if (index.as_var() &&
                  reduce_iter_vars.count(index.as_var_ref()) > 0) {
                find_reduce_var = true;
                break;
              }
            }
          }
          return find_reduce_var;
        },
        /* uniq_target = */ true);
    PADDLE_ENFORCE_EQ(reduce_loads.size(),
                      1,
                      ::common::errors::PreconditionNotMet(
                          "Required reduce_loads shall be 1."));

    std::vector<ir::Expr> reduce_load_indices =
        reduce_loads.begin()->As<ir::Load>()->indices;
    int loop_idx = 0;
    for (int i = 0; i < reduce_load_indices.size(); ++i) {
      ir::Expr& index = reduce_load_indices[i];
      if (index.is_constant()) continue;
      PADDLE_ENFORCE_NOT_NULL(index.as_var(),
                              ::common::errors::PreconditionNotMet(
                                  "Required index shall be var type."));
      ir::Var iter_var = index.as_var_ref();
      ir::Expr iter_value = iter_var2value.at(iter_var);
      PADDLE_ENFORCE_EQ(
          iter_value.as_var() || iter_value.is_constant(),
          true,
          ::common::errors::PreconditionNotMet(
              "Required iter_value shall be var or constant type."));
      ir::For* for_node;
      if (iter_value.as_var()) {
        for (ir::Expr& loop : loops) {
          if (loop.As<ir::For>()->loop_var == iter_value.as_var_ref()) {
            for_node = loop.As<ir::For>();
          }
        }
      } else if (iter_value.is_constant()) {
        for_node = loops.at(loop_idx).As<ir::For>();
      }
      PADDLE_ENFORCE_NOT_NULL(for_node,
                              ::common::errors::PreconditionNotMet(
                                  "Required for_node shall not be nullptr."));
      bool is_reduce_iter_var = reduce_iter_vars.count(iter_var) > 0;
      if (is_reduce_iter_var) {
        info.rb_space.emplace_back(for_node->extent,
                                   IterativeSpaceInfo::AxisType::kSerial);
        info.memory_consistent_order_space.emplace_back(for_node->extent);
        rb_iter_indices.push_back(loop_idx);
      } else {
        info.sp_space.emplace_back(for_node->extent,
                                   IterativeSpaceInfo::AxisType::kSerial);
        info.memory_consistent_order_space.emplace_back(for_node->extent);
        sp_iter_indices.push_back(loop_idx);
      }
      ++loop_idx;
    }
    info.rb_last_order.insert(info.rb_last_order.end(),
                              sp_iter_indices.begin(),
                              sp_iter_indices.end());
    info.rb_last_order.insert(info.rb_last_order.end(),
                              rb_iter_indices.begin(),
                              rb_iter_indices.end());
  } else {
    for (int i = 0; i < loops.size(); ++i) {
      ir::For* for_node = loops[i].As<ir::For>();
      info.memory_consistent_order_space.emplace_back(for_node->extent);
      info.sp_space.emplace_back(for_node->extent,
                                 IterativeSpaceInfo::AxisType::kSerial);
      info.rb_last_order.push_back(i);
    }
  }
  // init total extents
  ir::Expr sp_extent = ir::Expr(1);
  ir::Expr rb_extent = ir::Expr(1);
  for (const auto& axis : info.sp_space) {
    const ir::Expr& extent = std::get<0>(axis);
    sp_extent = sp_extent * extent;
  }
  for (const auto& axis : info.rb_space) {
    const ir::Expr& extent = std::get<0>(axis);
    rb_extent = rb_extent * extent;
  }
  info.total_sp_extent = common::AutoSimplify(sp_extent);
  info.total_rb_extent = common::AutoSimplify(rb_extent);

  return info;
}

ir::ScheduleBlockNode* DynamicShapeGroupScheduler::FindGlobalMasterNode(
    const std::unique_ptr<ir::ScheduleBlockGraph>& schedule_block_graph) {
  ir::ScheduleBlockNode* master = nullptr;
  // 1. reduce
  auto FindReduce = [&](ir::ScheduleBlockNode* node) {
    if (analyzer::IsReductionSBlock(node->Block())) {
      master = node;
    }
  };
  schedule_block_graph->NodesWalk(FindReduce);
  if (master != nullptr) {
    VLOG(6) << "Find the global master node: " << master->id();
    return master;
  }
  // 2. broadcast
  auto FindBroadcast = [&](ir::ScheduleBlockNode* node) {
    if (analyzer::IsBroadcastSBlock(node->Block())) {
      master = node;
    }
  };
  schedule_block_graph->NodesWalk(FindBroadcast);
  if (master != nullptr) {
    VLOG(6) << "Find the global master node: " << master->id();
    return master;
  }
  // 3. end point
  master = schedule_block_graph->EndPoints().back();
  VLOG(6) << "Find the global master node: " << master->id();
  return master;
}

}  // namespace ir
}  // namespace cinn
