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
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule_block_graph.h"

namespace cinn {
namespace ir {

using SymbolicPredicate = Expr;

struct GroupTileInfo {
  GroupTileInfo() {}

  std::vector<int64_t> reduce_axis_;
  int64_t data_rank;

  int64_t block_num{-1};
  int64_t warp_num;
  int64_t flatten_inner_num;
  int64_t reduce_numel;
  int64_t reduce_inner_num;
  int64_t reduce_block;

  std::set<std::string> reduce_var_names;
  std::set<std::string> temp_var_names;

  std::set<std::string> shared_var_names;
  std::set<std::string> direct_output_var_names;
  std::vector<std::string> thread_sync_before_names;

  int reduce_type{-1};

  std::unordered_map<std::string, BroadcastInfo> broadcast_info;
  std::unordered_map<std::string, BroadcastInfo> broadcast_to_elementwise;
};

/**
 * The base class used for scheduling fusion groups.
 */
class GroupScheduler {
 public:
  GroupScheduler(ir::IRSchedule* ir_sch,
                 const std::unordered_set<std::string>& output_tensor_names,
                 const cinn::common::Target& target,
                 std::shared_ptr<GroupTileInfo> group_tile_info)
      : ir_sch_(ir_sch),
        output_tensor_names_(output_tensor_names),
        target_(target),
        group_tile_info_(group_tile_info) {
    schedule_block_graph_ = std::make_unique<ir::ScheduleBlockGraph>(*ir_sch_);

    auto loop_name_get = [&](ir::ScheduleBlockNode* node) {
      node_list.push_back(node->id());
    };

    schedule_block_graph_->DFSTopoWalk(loop_name_get, false);

    if (group_tile_info_) {
      auto vec_axis = group_tile_info_->reduce_axis_;

      // reduce axis have be re-order to last
      int32_t reduce_start_idx = group_tile_info_->data_rank - vec_axis.size();
      for (int32_t i = 0; i < group_tile_info_->data_rank; ++i) {
        if (i >= reduce_start_idx) {
          vec_reduce_axis.push_back(i);
        } else {
          vec_flatten_axis.push_back(i);
        }
      }
    }
  }

  static std::unique_ptr<GroupScheduler> Make(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const cinn::common::Target& target,
      bool is_dy_shape = false,
      std::shared_ptr<GroupTileInfo> group_tile_info = nullptr);

  virtual ~GroupScheduler() = default;

  virtual void Schedule() = 0;

  virtual std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetIRs() = 0;

  std::unordered_set<std::string> OutputTensorNames() const;

  void LoopReorderAligment();

  void Tiling();

  bool NeedOrderLoops();

  void Unroll();
  void VariableTypeAssignment();
  void SetReduceType();
  void BindCudaInfo();

  void MergeFlattenAxis();
  void MergeReduceAxis();
  void SplitFlattenInner();
  void SplitReduceInner();
  void ReorderFlattenInnerWithReduceAxis();
  void SplitWarpNumber();

 protected:
  ir::IRSchedule* ir_sch_;
  const std::unordered_set<std::string>& output_tensor_names_;
  const cinn::common::Target& target_;
  // Graph in units of ScheduleBlockNode, each node corresponds to a
  // ScheduleBlock in IR.
  std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph_;

  std::shared_ptr<GroupTileInfo> group_tile_info_;

  std::vector<std::string> node_list;

  std::vector<int32_t> vec_flatten_axis;
  std::vector<int32_t> vec_reduce_axis;

  int reduce_current_axis{0};
};

}  // namespace ir
}  // namespace cinn
