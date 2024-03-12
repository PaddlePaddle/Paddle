// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include <memory>
#include "paddle/cinn/ir/schedule/schedule_base.h"

namespace cinn {

namespace hlir::framework::pir {
struct GroupInfo;
}  // namespace hlir::framework::pir

namespace ir {

struct GroupTileInfo {
  GroupTileInfo() {}

  std::vector<int64_t> reduce_axis_;
  int64_t data_rank;

  int64_t block_num{-1};
  int64_t warp_num;
  int64_t spatial_inner_num;
  int64_t reduce_numel;
  int64_t reduce_inner_num;
  int64_t reduce_block;

  bool is_reduce_all{false};

  std::set<std::string> reduce_tensor_names;
  std::set<std::string> temp_var_names;

  std::set<std::string> shared_var_names;
  std::set<std::string> direct_output_var_names;
  std::vector<std::string> thread_sync_before_names;

  ReduceMethod reduce_method{NoneReduceMethod()};

  std::unordered_map<std::string, BroadcastInfo> broadcast_info;
  std::unordered_map<std::string, BroadcastInfo> broadcast_to_elementwise;
};

std::shared_ptr<GroupTileInfo> GetGroupTileInfo(
    const std::shared_ptr<cinn::hlir::framework::pir::GroupInfo>& group_info);

}  // namespace ir
}  // namespace cinn
