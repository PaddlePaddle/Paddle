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

#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"

namespace cinn {
namespace ir {

std::unique_ptr<GroupScheduler> GroupScheduler::Make(
    ir::IRSchedule* ir_sch,
    const std::unordered_set<std::string>& output_tensor_names,
    const cinn::common::Target& target,
    bool is_dy_shape,
    std::shared_ptr<GroupTileInfo> group_tile_info) {
  if (is_dy_shape) {
    std::cerr << "dy shape " << (group_tile_info != nullptr) << std::endl;
    return std::make_unique<DynamicShapeGroupScheduler>(
        ir_sch, output_tensor_names, target, group_tile_info);
  } else {
    return std::make_unique<StaticShapeGroupScheduler>(
        ir_sch, output_tensor_names, target, group_tile_info);
  }
}

std::unordered_set<std::string> GroupScheduler::OutputTensorNames() const {
  std::unordered_set<std::string> output_tensor_names{output_tensor_names_};
  for (ir::ScheduleBlockNode* node : schedule_block_graph_->EndPoints()) {
    output_tensor_names.insert(node->id());
  }
  return output_tensor_names;
}

void GroupScheduler::LoopReorderAligment() {
  // std::cerr << "loop reorder func body: "
  //           << ir_sch_->GetModule().GetExprs().front() << std::endl;
  std::vector<std::string> node_list;

  auto loop_name_get = [&](ir::ScheduleBlockNode* node) {
    node_list.push_back(node->id());
  };

  schedule_block_graph_->DFSTopoWalk(loop_name_get, false);

  std::cerr << "group_tile_info_ " << (group_tile_info_ != nullptr)
            << std::endl;
  // broadcast
  for (auto& name : node_list) {
    // skip reduce init block

    if (group_tile_info_->broadcast_info.count(name)) {
      // broadcast loops
      std::cerr << "broadcast axes \n";
      for (auto& axis : group_tile_info_->broadcast_info[name].broadcast_axes) {
        std::cerr << "axis " << axis << std::endl;
      }
      std::cerr << "out shape \n";
      for (auto& s : group_tile_info_->broadcast_info[name].output_shape) {
        std::cerr << "dim " << s << std::endl;
      }
      if (group_tile_info_->broadcast_info[name].full_broadcast) {
        // split first
        std::vector<int32_t> vec_out_split(
            group_tile_info_->broadcast_info[name].output_shape.size(), 1);
        std::cerr << "split size " << vec_out_split.size() << std::endl;

        auto loops = ir_sch_->GetLoops(name);
        std::cerr << "before split\n " << loops[0] << std::endl;
        ir_sch_->Split(loops[0], vec_out_split);

        loops = ir_sch_->GetLoops(name);
        std::cerr << "after split\n " << loops[0] << std::endl;
      } else if (group_tile_info_->broadcast_info[name].split_first) {
        std::cerr << "split first !!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        for (auto& info : group_tile_info_->broadcast_info[name].split_info) {
          auto axis = info.first;
          auto split_res = info.second;

          auto loops = ir_sch_->GetLoops(name);
          std::cerr << "before split\n " << loops[0] << std::endl;
          ir_sch_->Split(loops[axis], split_res);

          loops = ir_sch_->GetLoops(name);
          std::cerr << "after split\n " << loops[0] << std::endl;
        }
      }

      ir_sch_->Broadcast(name, group_tile_info_->broadcast_info[name]);
    }

    std::cerr << "fin broadcast " << name << std::endl;
    if (group_tile_info_->broadcast_to_elementwise.count(name)) {
      std::cerr << "begin to broadcat to elementwise\n";
      ir_sch_->BroadcastToElementwise(
          name,
          group_tile_info_->broadcast_to_elementwise[name].broadcast_axes);
      std::cerr << "fin to broadcat to elementwise\n";
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

bool GroupScheduler::NeedOrderLoops() {
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

void GroupScheduler::Tiling() {
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

  std::cerr << "before flatten fuse: "
            << ir_sch_->GetModule().GetExprs().front() << std::endl;

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

  std::cerr << "after flatten fuse: " << ir_sch_->GetModule().GetExprs().front()
            << std::endl;

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

      if (split_expr->extent.as_int64() == 1) {
        continue;
      }

      std::vector<int> split_factors;
      if (group_tile_info_->reduce_block >= 2048) {
        split_factors.emplace_back(
            std::ceil(group_tile_info_->reduce_numel * 1.0 /
                      group_tile_info_->reduce_inner_num));
        split_factors.emplace_back(group_tile_info_->reduce_inner_num);
      } else {
        split_factors.emplace_back(
            std::ceil(group_tile_info_->reduce_block * 1.0 /
                      group_tile_info_->reduce_inner_num));
        split_factors.emplace_back(group_tile_info_->reduce_inner_num);
      }

      auto split_loops =
          ir_sch_->Split(loops[reduce_current_axis], split_factors);

      if (group_tile_info_->reduce_var_names.count(name)) {
        ir_sch_->FactorizeReduction(split_loops[0], 0);
      }
    }
  }

  // std::cerr << "after split reduce: " <<
  // ir_sch_->GetModule().GetExprs().front()
  //           << std::endl;

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

  std::cerr << "after bind block and thread info: "
            << ir_sch_->GetModule().GetExprs().front() << std::endl;

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

    {
      if (!group_tile_info_->direct_output_var_names.count(name)) {
        std::cerr << "set local " << name << std::endl;
        ir_sch_->SetBuffer(block, "local", false);
      }
    }
    //}

    if (group_tile_info_->reduce_var_names.count(name)) {
      auto block = ir_sch_->GetBlock(name + "_rf");
      ir_sch_->SetBuffer(block, "local", false);
    }
  }

  // for (auto& name : group_tile_info_->copyed_var_names) {
  //   auto block = ir_sch_->GetBlock(name);
  //   ir_sch_->SetBuffer(block, "local", false);
  // }

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
