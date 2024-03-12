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

#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

namespace cinn {
namespace ir {

int64_t Next2Power(int64_t n) {
  if (n == 1) {
    return 1;
  }
  return int64_t(std::pow(2.0, std::ceil(std::log2(n))));
}

std::shared_ptr<GroupTileInfo> GetGroupTileInfo(
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info) {
  std::shared_ptr<GroupTileInfo> group_tile_info =
      std::make_shared<GroupTileInfo>();

  group_tile_info->data_rank = group_info->data_space.size();
  std::set<int64_t> reduce_set;
  for (auto dim : group_info->reduce_axis) {
    if (dim < 0) {
      dim += group_tile_info->data_rank;
    }

    group_tile_info->reduce_axis_.push_back(dim);
    reduce_set.insert(dim);
  }

  int64_t spatial_numel = 1;
  int64_t reduce_numel = 1;

  bool spatial_is_dynamic = false;
  bool reduce_is_dynamic = false;
  for (int64_t i = 0; i < group_tile_info->data_rank; ++i) {
    if (reduce_set.count(i)) {
      reduce_numel *= group_info->data_space[i];
      if (group_info->data_space[i] < 0) {
        reduce_is_dynamic = true;
      }
    } else {
      spatial_numel *= group_info->data_space[i];

      if (group_info->data_space[i] < 0) {
        spatial_is_dynamic = true;
      }
    }
  }

  bool is_reduce_all =
      (group_tile_info->reduce_axis_.size() == group_tile_info->data_rank);

  if (is_reduce_all) {
    reduce_is_dynamic = false;
  }

  PADDLE_ENFORCE_EQ(
      reduce_is_dynamic,
      false,
      phi::errors::Unimplemented("not support dynamic reduce yet"));

  int64_t reduce_block = 1;
  int64_t spatial_block = 1;

  int64_t reduce_inner_num = 1;
  int64_t spatial_inner_num = 1;
  int warp_num = 1;
  group_tile_info->is_reduce_all = is_reduce_all;

  if (is_reduce_all) {
    // warp reduce
    reduce_block = 1024;
    spatial_block = 1;
    spatial_inner_num = 1;
    reduce_inner_num = 4;
    warp_num = 8;

  } else if (reduce_numel == 1) {
    reduce_block = 1;
    if (spatial_is_dynamic) {
      spatial_block = 1024;

      reduce_inner_num = 1;
      warp_num = 8;

      spatial_inner_num = 4;

      group_tile_info->block_num = -1;
    } else {
      spatial_block = Next2Power(spatial_numel);
      if (spatial_block > 1024) {
        spatial_block = 1024;
      }
      reduce_inner_num = 1;
      warp_num = spatial_block / 128;
      if (warp_num == 0) {
        warp_num = 1;
      }
      spatial_inner_num = spatial_block / (warp_num * 32);
      if (spatial_inner_num == 0) {
        spatial_inner_num = 1;
      }

      int64_t block_num =
          int64_t(std::ceil(spatial_numel * 1.0 / spatial_block));
      group_tile_info->block_num = block_num;
    }
  } else if (reduce_numel <= 256) {
    // warp reduce
    reduce_block = Next2Power(reduce_numel);
    spatial_block = 256 / reduce_block;
    spatial_inner_num = spatial_block;
    reduce_inner_num = reduce_block / 32;
    if (reduce_inner_num == 0) {
      reduce_inner_num = 2;
    }
    warp_num = 8;
  } else if (reduce_numel > 256 && reduce_numel <= 2048) {
    spatial_block = 1;
    reduce_block = int64_t(std::ceil(reduce_numel * 1.0 / 256.0)) * 256;
    warp_num = reduce_block / 256;
    spatial_inner_num = 1;
    reduce_inner_num = 8;
  } else if (reduce_numel > 2048) {
    spatial_block = 1;
    reduce_block = 2048;
    warp_num = 8;
    reduce_inner_num = int64_t(std::ceil(reduce_numel * 1.0 / 256.0));
    spatial_inner_num = 1;
  }

  group_tile_info->reduce_numel = reduce_numel;
  group_tile_info->reduce_block = reduce_block;

  VLOG(6) << "block num " << group_tile_info->block_num << std::endl;
  VLOG(6) << "num warp " << warp_num << std::endl;
  VLOG(6) << "flatten block " << spatial_block << std::endl;
  VLOG(6) << "reduce block  " << reduce_block << std::endl;
  VLOG(6) << "flatten inner num " << spatial_inner_num << std::endl;
  VLOG(6) << "reduce inner num " << reduce_inner_num << std::endl;

  group_tile_info->warp_num = warp_num;
  group_tile_info->spatial_inner_num = spatial_inner_num;
  group_tile_info->reduce_inner_num = reduce_inner_num;

  if (reduce_block > 1 && reduce_block <= 256) {
    group_tile_info->reduce_method = ir::WarpReduceMethod();
  }

  group_tile_info->reduce_tensor_names = group_info->reduce_var_names;
  group_tile_info->shared_var_names = group_info->shared_var_names;
  group_tile_info->direct_output_var_names =
      group_info->direct_output_var_names;
  group_tile_info->thread_sync_before_names =
      group_info->thread_sync_before_names;
  group_tile_info->broadcast_info = group_info->broadcast_info;
  group_tile_info->broadcast_to_elementwise =
      group_info->broadcast_to_elementwise;

  return group_tile_info;
}

}  // namespace ir
}  // namespace cinn
