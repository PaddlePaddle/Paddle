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
#include "paddle/cinn/ir/group_schedule/tactic/align_iter_space_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/arrange_storage_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/bind_cuda_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/optimize_reduction_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/tile_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace ir {

void DynamicShapeGroupScheduler::Init() {
  InitBuckets();
  tactics_.emplace_back(new AlignIterSpaceTactic());
  tactics_.emplace_back(new ComputeInlineTactic());
  tactics_.emplace_back(new TileTactic());
  tactics_.emplace_back(new OptimizeReductionTactic());
  tactics_.emplace_back(new BindCudaTactic());
  tactics_.emplace_back(new ArrangeStorageTactic());
}

void DynamicShapeGroupScheduler::InitBuckets() {
  std::unordered_set<std::string> output_names = OutputTensorNames();

  auto OutOfRange =
      [](ir::Expr extent, int lower_bound, int upper_bound) -> bool {
    if (!extent.is_constant()) return false;
    int extent_value = static_cast<int>(extent.get_constant());
    if (extent_value < lower_bound || extent_value >= upper_bound) {
      return true;
    }
    return false;
  };

  auto InitBucket = [&](BucketInfo&& bucket_info) {
    std::unique_ptr<ir::IRSchedule> ir_sch =
        std::make_unique<ir::IRSchedule>(*ir_sch_);
    std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph =
        std::make_unique<ir::ScheduleBlockGraph>(*ir_sch);
    ir::ScheduleBlockNode* global_master =
        FindGlobalMasterNode(schedule_block_graph);
    IterativeSpaceInfo iter_space_info = ConstructIterSpaceInfo(global_master);
    if (OutOfRange(iter_space_info.total_sp_extent,
                   bucket_info.sp_lower_bound,
                   bucket_info.sp_upper_bound) ||
        OutOfRange(iter_space_info.total_rb_extent,
                   bucket_info.rb_lower_bound,
                   bucket_info.rb_upper_bound)) {
      return;
    }
    SymbolicPredicate sp_lower_bound_predicate = ir::GE::Make(
        iter_space_info.total_sp_extent, ir::Expr(bucket_info.sp_lower_bound));
    SymbolicPredicate sp_upper_bound_predicate = ir::LT::Make(
        iter_space_info.total_sp_extent, ir::Expr(bucket_info.sp_upper_bound));
    SymbolicPredicate rb_lower_bound_predicate = ir::GE::Make(
        iter_space_info.total_rb_extent, ir::Expr(bucket_info.rb_lower_bound));
    SymbolicPredicate rb_upper_bound_predicate = ir::LT::Make(
        iter_space_info.total_rb_extent, ir::Expr(bucket_info.rb_upper_bound));
    SymbolicPredicate sp_predicate =
        ir::And::Make(sp_lower_bound_predicate, sp_upper_bound_predicate);
    SymbolicPredicate rb_predicate =
        ir::And::Make(rb_lower_bound_predicate, rb_upper_bound_predicate);
    SymbolicPredicate predicate = ir::And::Make(sp_predicate, rb_predicate);
    ScheduleContext schedule_context{output_names,
                                     target_,
                                     std::move(iter_space_info),
                                     std::move(bucket_info)};
    BucketContext bucket_context{std::move(predicate),
                                 std::move(ir_sch),
                                 std::move(schedule_block_graph),
                                 std::move(schedule_context)};
    bucket_contexts_.emplace_back(std::move(bucket_context));
  };

  // naive buckets
  // 1. {sp_extent[1 - 1024], rb_extent[1 - 256]}
  InitBucket({/* sp_lower_bound = */ 1,
              /* sp_upper_bound = */ 1024,
              /* rb_lower_bound = */ 1,
              /* rb_upper_bound = */ 256});
  // 2. {sp_extent[1024 - +oo], rb_extent[1 - 256]}
  InitBucket({/* sp_lower_bound = */ 1024,
              /* sp_upper_bound = */ INT_MAX,
              /* rb_lower_bound = */ 1,
              /* rb_upper_bound = */ 256});
  // 3. {sp_extent[1 - 1024], rb_extent[256 - +oo]}
  InitBucket({/* sp_lower_bound = */ 1,
              /* sp_upper_bound = */ 1024,
              /* rb_lower_bound = */ 256,
              /* rb_upper_bound = */ INT_MAX});
  // 4. {sp_extent[1024 - +oo], rb_extent[256 - +oo]}
  InitBucket({/* sp_lower_bound = */ 1024,
              /* sp_upper_bound = */ INT_MAX,
              /* rb_lower_bound = */ 256,
              /* rb_upper_bound = */ INT_MAX});
}

void DynamicShapeGroupScheduler::Schedule() {
  // for (BucketContext& bucket_context : bucket_contexts_) {
  //   VLOG(4) << "===========================Apply tactics on Bucket ["
  //           << bucket_context.predicate << "]==========================";
  //   ApplyTactics(&bucket_context);
  // }
  std::cerr << "dy shape schedule\n";
  LoopReorderAligment();
  Tiling();
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
    CHECK_EQ(reduce_loads.size(), 1);

    std::vector<ir::Expr> reduce_load_indices =
        reduce_loads.begin()->As<ir::Load>()->indices;
    int loop_idx = 0;
    for (int i = 0; i < reduce_load_indices.size(); ++i) {
      ir::Expr& index = reduce_load_indices[i];
      if (index.is_constant()) continue;
      CHECK_NOTNULL(index.as_var());
      ir::Var iter_var = index.as_var_ref();
      ir::Expr iter_value = iter_var2value.at(iter_var);
      CHECK_NOTNULL(iter_value.as_var());
      ir::For* for_node;
      for (ir::Expr& loop : loops) {
        if (loop.As<ir::For>()->loop_var == iter_value.as_var_ref()) {
          for_node = loop.As<ir::For>();
        }
      }
      CHECK_NOTNULL(for_node);
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

void DynamicShapeGroupScheduler::LoopReorderAligment() {
  std::cerr << "loop reorder func body: "
            << ir_sch_->GetModule().GetExprs().front() << std::endl;
  std::vector<std::string> node_list;

  auto loop_name_get = [&](ir::ScheduleBlockNode* node) {
    node_list.push_back(node->id());
  };

  schedule_block_graph_->DFSTopoWalk(loop_name_get, false);

  // broadcast
  for (auto& name : node_list) {
    // skip reduce init block
    if (group_tile_info_->broadcast_info.count(name)) {
      // broadcast loops
      // std::cerr << "broadcast axes \n";
      // for (auto& axis :
      // group_tile_info_->broadcast_info[name].broadcast_axes) {
      //   std::cerr << "axis" << axis << std::endl;
      // }
      // std::cerr << "out shape \n";
      // for (auto& s : group_tile_info_->broadcast_info[name].output_shape) {
      //   std::cerr << "dim " << s << std::endl;
      // }

      ir_sch_->Broadcast(name,
                         group_tile_info_->broadcast_info[name].broadcast_axes,
                         group_tile_info_->broadcast_info[name].output_shape,
                         group_tile_info_->broadcast_info[name].with_constrain);
    }

    if (group_tile_info_->broadcast_to_elementwise.count(name)) {
      ir_sch_->BroadcastToElementwise(
          name,
          group_tile_info_->broadcast_to_elementwise[name].broadcast_axes);
    }
  }

  size_t base_rank = 0;
  for (auto& name : node_list) {
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }

    auto loops = ir_sch_->GetLoops(name);

    if (base_rank == 0) {
      base_rank = loops.size();
    } else {
      if (base_rank != loops.size()) {
        std::cerr << "name " << name << "\t" << base_rank << "\t"
                  << loops.size() << std::endl;
        throw std::runtime_error("loops  rank not same ");
      }
    }
  }

  if (!NeedOrderLoops()) {
    // std::cerr << "no need reorder \n";
    return;
  }

  // std::cerr << "need re-order here !!!!!!\n";

  // std::cerr << "node size " << node_list.size() << std::endl;

  // Get re-order loops
  std::set<int64_t> reduce_set(group_tile_info_->reduce_axis_.begin(),
                               group_tile_info_->reduce_axis_.end());

  std::vector<int32_t> new_order;
  for (int32_t i = 0; i < group_tile_info_->data_rank; ++i) {
    if (!reduce_set.count(i)) {
      new_order.push_back(i);
    }
  }

  for (auto axis : group_tile_info_->reduce_axis_) {
    new_order.push_back(axis);
  }

  for (auto& name : node_list) {
    // skip reduce init block
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }

    if (group_tile_info_->reduce_var_names.count(name)) {
      continue;
    }

    ir_sch_->Reorder(name, new_order);
  }

  // std::cerr << "after loop reorder func body: "
  //           << ir_sch_->GetModule().GetExprs().front() << std::endl;
}

bool DynamicShapeGroupScheduler::NeedOrderLoops() {
  if (group_tile_info_) {
    if (group_tile_info_->reduce_axis_.size() == 0) {
      return false;
    }
    // std::cerr << "reduce rank " << group_tile_info_->data_rank << std::endl;

    std::vector<int64_t> vec_axis = group_tile_info_->reduce_axis_;

    std::sort(vec_axis.begin(), vec_axis.end());

    if (vec_axis.front() == group_tile_info_->data_rank - vec_axis.size()) {
      return false;
    } else {
      return true;
    }
  }

  return false;
}

// void StaticShapeGroupScheduler::BindCudaBlockThread()
// {

// }

void DynamicShapeGroupScheduler::Tiling() {
  // apply tiling
  std::vector<std::string> node_list;

  auto loop_name_get = [&](ir::ScheduleBlockNode* node) {
    node_list.push_back(node->id());
  };

  schedule_block_graph_->DFSTopoWalk(loop_name_get, false);

  // std::cerr << "node size " << node_list.size() << std::endl;

  auto vec_axis = group_tile_info_->reduce_axis_;

  // merge flatten axis and reduce axis
  std::set<int64_t> reduce_set(vec_axis.begin(), vec_axis.end());

  std::vector<int32_t> vec_flatten_axis;
  std::vector<int32_t> vec_reduce_axis;

  // reduce axis have be re-order to last
  int32_t reduce_start_idx = group_tile_info_->data_rank - vec_axis.size();
  for (int32_t i = 0; i < group_tile_info_->data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis.push_back(i);
    } else {
      vec_flatten_axis.push_back(i);
    }
  }

  if (vec_flatten_axis.size() >= 2) {
    for (auto& name : node_list) {
      // skip reduce init block
      if (ir::IsReduceInitTensorName(name)) {
        continue;
      }
      ir_sch_->Fuse(name, vec_flatten_axis);
    }
  }

  int reduce_current_axis = 1;
  if (vec_reduce_axis.size() >= 2) {
    for (auto& name : node_list) {
      // skip reduce init block
      if (ir::IsReduceInitTensorName(name)) {
        continue;
      }
      // std::cerr << "reduce axis\n";

      // for (auto axis : vec_reduce_axis) {
      //   //std::cerr << axis << std::endl;
      // }
      ir_sch_->Fuse(name, vec_reduce_axis);
    }
  }

  // std::cerr << "after flatten fuse: " <<
  // ir_sch_->GetModule().GetExprs().front()
  //           << std::endl;

  if (group_tile_info_->flatten_inner_num > 1) {
    // split flatten inner here
    for (auto& name : node_list) {
      // skip reduce init block
      if (ir::IsReduceInitTensorName(name)) {
        continue;
      }
      auto loops = ir_sch_->GetLoops(name);

      auto split_loops = ir_sch_->Split(
          loops[0],
          std::vector<int>({-1, group_tile_info_->flatten_inner_num}));
    }

    reduce_current_axis += 1;
  }
  // std::cerr << "split flatten inner: "
  //           << ir_sch_->GetModule().GetExprs().front() << std::endl;

  // std::cerr << "current reduce " << reduce_current_axis << std::endl;
  // split reduce inner here

  if (group_tile_info_->reduce_inner_num > 1) {
    for (auto& name : node_list) {
      // skip reduce init block
      if (ir::IsReduceInitTensorName(name)) {
        continue;
      }
      auto loops = ir_sch_->GetLoops(name);

      auto split_expr = loops[reduce_current_axis].As<ir::For>();
      // std::cerr << "split expr" << split_expr << std::endl;
      // std::cerr << "extent " << split_expr->extent.as_int32() << std::endl;

      if (split_expr->extent.as_int32() == 1) {
        continue;
      }

      std::vector<int> split_factors{
          group_tile_info_->reduce_block / group_tile_info_->reduce_inner_num,
          group_tile_info_->reduce_inner_num};

      auto split_loops =
          ir_sch_->Split(loops[reduce_current_axis], split_factors);

      if (group_tile_info_->reduce_var_names.count(name)) {
        ir_sch_->FactorizeReduction(split_loops[0], 0);
      }
    }
  }

  std::cerr << "after split reduce: " << ir_sch_->GetModule().GetExprs().front()
            << std::endl;

  // re-order flatten inner num with last dim
  if (group_tile_info_->flatten_inner_num > 1 &&
      (group_tile_info_->reduce_axis_.size() > 0)) {
    for (auto& name : node_list) {
      // skip reduce init block
      if (ir::IsReduceInitTensorName(name)) {
        continue;
      }
      auto loops = ir_sch_->GetLoops(name);

      ir_sch_->Reorder({loops[2], loops[1]});

      if (group_tile_info_->reduce_var_names.count(name)) {
        auto loops = ir_sch_->GetLoops(name + "_rf");
        ir_sch_->Reorder({loops[2], loops[1]});
      }
    }

    // std::cerr << "after reorder flatten inner num: "
    //           << ir_sch_->GetModule().GetExprs().front() << std::endl;
  }

  // std::cerr << "split warp \n";
  // get warp num
  if (group_tile_info_->warp_num > 1) {
    if (group_tile_info_->reduce_axis_.size() == 0) {
      // std::cerr << "split loop 0\n";
      // only elementwise and broadcast
      // get num warp from flatten num
      for (auto& name : node_list) {
        // skip reduce init block
        if (ir::IsReduceInitTensorName(name)) {
          continue;
        }
        auto loops = ir_sch_->GetLoops(name);

        ir_sch_->Split(loops[0],
                       std::vector<int>({group_tile_info_->block_num,
                                         group_tile_info_->warp_num * 32}));
      }
    } else if (group_tile_info_->flatten_inner_num > 1) {
      // get num warp from flatten num
      for (auto& name : node_list) {
        // skip reduce init block
        if (ir::IsReduceInitTensorName(name)) {
          continue;
        }
        auto loops = ir_sch_->GetLoops(name);

        ir_sch_->Split(loops[0],
                       std::vector<int>({-1, group_tile_info_->warp_num}));

        loops = ir_sch_->GetLoops(name);

        ir_sch_->Fuse({loops[1], loops[2]});

        if (group_tile_info_->reduce_var_names.count(name)) {
          auto loops = ir_sch_->GetLoops(name + "_rf");

          ir_sch_->Split(loops[0],
                         std::vector<int>({-1, group_tile_info_->warp_num}));

          loops = ir_sch_->GetLoops(name + "_rf");

          ir_sch_->Fuse({loops[1], loops[2]});
        }
      }
    }
  } else {
    // get num warp from reduce num
    // do nothing for now
  }

  // std::cerr << "after merge warp num info: "
  //           << ir_sch_->GetModule().GetExprs().front() << std::endl;

  // bind cuda block and thread info
  for (auto& name : node_list) {
    // skip reduce init block
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }
    auto loops = ir_sch_->GetLoops(name);
    if (loops.size() == 1) {
      ir_sch_->Split(loops[0], std::vector<int>({1, -1}));
    }

    loops = ir_sch_->GetLoops(name);

    ir_sch_->Bind(loops[0], "blockIdx.x");

    ir_sch_->Bind(loops[1], "threadIdx.x");

    if (group_tile_info_->reduce_var_names.count(name)) {
      auto loops = ir_sch_->GetLoops(name + "_rf");

      ir_sch_->Bind(loops[0], "blockIdx.x");

      ir_sch_->Bind(loops[1], "threadIdx.x");
    }
  }

  // std::cerr << "after bind block and thread info: "
  //           << ir_sch_->GetModule().GetExprs().front() << std::endl;

  for (auto& name : node_list) {
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }
    if (name.find("_out") != std::string::npos) {
      continue;
    }
    // if (!output_tensor_names_.count(name)) {
    //  std::cerr << "temp name " << name << std::endl;
    auto block = ir_sch_->GetBlock(name);
    if (group_tile_info_->shared_var_names.count(name)) {
      ir_sch_->SetBuffer(block, "shared", false);
    } else {
      ir_sch_->SetBuffer(block, "local", false);
    }
    //}

    if (group_tile_info_->reduce_var_names.count(name)) {
      auto block = ir_sch_->GetBlock(name + "_rf");
      ir_sch_->SetBuffer(block, "local", false);
    }
  }

  for (auto& name : group_tile_info_->copyed_var_names) {
    auto block = ir_sch_->GetBlock(name);
    ir_sch_->SetBuffer(block, "local", false);
  }

  for (auto& name : group_tile_info_->thread_sync_before_names) {
    auto loops = ir_sch_->GetLoops(name);
    ir_sch_->SyncThreads(loops.front(), false);
  }

  // set unroll
  for (auto& name : node_list) {
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }
    auto loops = ir_sch_->GetLoops(name);

    if (loops.size() > 2) {
      ir_sch_->Unroll(loops[2]);
    }
    if (loops.size() > 3) {
      ir_sch_->Unroll(loops[3]);
    }

    if (group_tile_info_->reduce_var_names.count(name)) {
      auto loops = ir_sch_->GetLoops(name + "_rf");

      if (loops.size() > 2) {
        ir_sch_->Unroll(loops[2]);
      }
      if (loops.size() > 3) {
        ir_sch_->Unroll(loops[3]);
      }
    }
  }

  for (auto& name : node_list) {
    if (ir::IsReduceInitTensorName(name)) {
      continue;
    }

    if (group_tile_info_->reduce_var_names.count(name)) {
      auto block = ir_sch_->GetBlock(name)
                       .As<ir::ScheduleBlockRealize>()
                       ->schedule_block.As<ir::ScheduleBlock>();
      if (group_tile_info_->reduce_type == 0) {
        block->reduce_type = 0;
      }
      // std::cerr << "block type " << block->reduce_type << std::endl;
    }
  }

  schedule_block_graph_->Update(*ir_sch_);
}

}  // namespace ir
}  // namespace cinn
