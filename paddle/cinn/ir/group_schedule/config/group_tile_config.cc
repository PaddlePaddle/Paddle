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

const int kMaxNumel = INT32_MAX;

BucketInfo::BucketInfo(int sp_lower_bound,
                       int sp_upper_bound,
                       int rb_lower_bound,
                       int rb_upper_bound,
                       bool sp_is_dynamic = false,
                       bool rb_is_dynamic = false) {
  BucketInfo::Dimension sp_dimension(
      sp_lower_bound, sp_upper_bound, "S", sp_is_dynamic);
  BucketInfo::Dimension rb_dimension(
      rb_lower_bound, rb_upper_bound, "R", rb_is_dynamic);
  this->space.push_back(sp_dimension);
  this->space.push_back(rb_dimension);
}

bool BucketInfo::operator==(const BucketInfo& other) const {
  if (this->space.size() != other.space.size()) {
    return false;
  }
  int length = this->space.size();
  for (int i = 0; i < length; i++) {
    if (this->space[i].is_dynamic != other.space[i].is_dynamic ||
        this->space[i].iter_type != other.space[i].iter_type ||
        this->space[i].lower_bound != other.space[i].lower_bound ||
        this->space[i].upper_bound != other.space[i].upper_bound) {
      return false;
    }
  }
  return true;
}

std::string BucketInfo::ToString() const {
  std::stringstream ss;
  ss << "BucketInfo: [";
  for (const auto& dim : space) {
    ss << dim.iter_type << "(" << dim.lower_bound << " - " << dim.upper_bound
       << "), ";
  }
  ss << "]";
  return ss.str();
}

int64_t Next2Power(int64_t n) {
  if (n == 1) {
    return 1;
  }
  return int64_t(std::pow(2.0, std::ceil(std::log2(n))));
}

std::shared_ptr<ScheduleConfig::BaseInfo> InitBasicInfo(
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      std::make_shared<ScheduleConfig::BaseInfo>();
  base_info->reduce_tensor_names = group_info->reduce_var_names;
  base_info->shared_var_names = group_info->shared_var_names;
  base_info->direct_output_var_names = group_info->direct_output_var_names;
  base_info->broadcast_info = group_info->broadcast_info;
  base_info->broadcast_to_elementwise = group_info->broadcast_to_elementwise;
  base_info->data_rank = group_info->data_space.size();

  std::set<int64_t> reduce_dim_loc;
  for (auto dim : group_info->reduce_axis) {
    if (dim < 0) {
      dim += base_info->data_rank;
    }
    base_info->reduce_axis.push_back(dim);
    reduce_dim_loc.insert(dim);
  }

  base_info->spatial_numel = 1;
  base_info->reduce_numel = 1;
  for (int64_t i = 0; i < base_info->data_rank; ++i) {
    if (reduce_dim_loc.count(i)) {
      if (group_info->data_space[i] == -1) base_info->has_dynamic_reduce = true;
      base_info->reduce_numel *= group_info->data_space[i];
    } else {
      if (group_info->data_space[i] == -1)
        base_info->has_dynamic_spatial = true;
      base_info->spatial_numel *= group_info->data_space[i];
    }
  }
  base_info->is_reduce_all =
      (base_info->reduce_axis.size() == base_info->data_rank);

  return base_info;
}

std::unordered_map<BucketInfo, ScheduleConfig::TileConfig, BucketInfoHash>
BuildPureStaticShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  if (base_info->spatial_numel == 1) {  // reduce all
    if (base_info->reduce_numel <= 256) {
      BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                             /* sp_upper_bound = */ 1,
                             /* rb_lower_bound = */ 1,
                             /* rb_upper_bound = */ 256};
      ScheduleConfig::TileConfig tile_config{
          /* warp_num = */ (base_info->reduce_numel + 31) / 32,
          /* tree_reduce_num = */ base_info->reduce_numel,
          /* spatial_inner_num = */ 1,
          /* reduce_method = */ BlockReduceMethod()};
      return {{bucket_info, tile_config}};
    } else if (base_info->reduce_numel <= 2048) {
      BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                             /* sp_upper_bound = */ 1,
                             /* rb_lower_bound = */ 257,
                             /* rb_upper_bound = */ kMaxNumel};
      ScheduleConfig::TileConfig tile_config{
          /* warp_num = */ 8,
          /* tree_reduce_num = */ 256,
          /* spatial_inner_num = */ 1,
          /* reduce_method = */ BlockReduceMethod()};
      return {{bucket_info, tile_config}};
    } else {
      BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                             /* sp_upper_bound = */ 1,
                             /* rb_lower_bound = */ 2049,
                             /* rb_upper_bound = */ kMaxNumel};
      ScheduleConfig::TileConfig tile_config{
          /* warp_num = */ 32,
          /* tree_reduce_num = */ 1024,
          /* spatial_inner_num = */ 1,
          /* reduce_method = */ BlockReduceMethod()};
      return {{bucket_info, tile_config}};
    }
  } else if (base_info->reduce_numel == 1) {  // no reduce
    int64_t spatial_block = Next2Power(base_info->spatial_numel);
    if (spatial_block > 1024) {
      spatial_block = 1024;
    }
    int64_t warp_num = spatial_block / 128;
    if (warp_num == 0) {
      warp_num = 1;
    }
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 1,
                           /* rb_upper_bound = */ 1};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ 1,
        /* spatial_inner_num = */ 4,
        /* reduce_method = */ NoneReduceMethod()};
    return {{bucket_info, tile_config}};
  } else if (base_info->reduce_numel <= 256) {
    // warp reduce
    int64_t reduce_block = Next2Power(base_info->reduce_numel);
    int64_t spatial_inner_num =
        std::min(256 / reduce_block, base_info->spatial_numel);
    int64_t tree_reduce_num = 32;
    int64_t warp_num = 8;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 1,
                           /* rb_upper_bound = */ 256};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ tree_reduce_num,
        /* spatial_inner_num = */ spatial_inner_num,
        /* reduce_method = */ WarpReduceMethod()};
    return {{bucket_info, tile_config}};
  } else if (base_info->reduce_numel <= 2048) {
    int64_t reduce_block =
        int64_t(std::ceil(base_info->reduce_numel * 1.0 / 256.0)) * 256;
    int64_t warp_num = reduce_block / 256;
    int64_t spatial_inner_num = 1;
    int64_t reduce_inner_num = 8;
    int64_t tree_reduce_num = reduce_block / reduce_inner_num;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 257,
                           /* rb_upper_bound = */ 2048};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ tree_reduce_num,
        /* spatial_inner_num = */ spatial_inner_num,
        /* reduce_method = */ BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  } else {
    int64_t warp_num = 32;
    int64_t spatial_inner_num = 1;
    int64_t tree_reduce_num = warp_num * 32;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 2049,
                           /* rb_upper_bound = */ kMaxNumel};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ tree_reduce_num,
        /* spatial_inner_num = */ spatial_inner_num,
        /* reduce_method = */ BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  }
}

std::unordered_map<BucketInfo, ScheduleConfig::TileConfig, BucketInfoHash>
BuildStaticSpatialConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  if (base_info->spatial_numel == 1) {  // reduce all
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ 1,
                           /* rb_lower_bound = */ 1,
                           /* rb_upper_bound = */ kMaxNumel,
                           /* sp_is_dynamic = */ false,
                           /* rb_is_dynamic = */ true};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 256,
        /* spatial_inner_num = */ 1,
        /* reduce_method = */ BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  } else {
    BucketInfo bucket_info_1_256{/* sp_lower_bound = */ 1,
                                 /* sp_upper_bound = */ kMaxNumel,
                                 /* rb_lower_bound = */ 1,
                                 /* rb_upper_bound = */ 256,
                                 /* sp_is_dynamic = */ false,
                                 /* rb_is_dynamic = */ true};
    ScheduleConfig::TileConfig tile_config_1_256{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 32,
        /* spatial_inner_num = */ 1,
        /* reduce_method = */ WarpReduceMethod()};

    BucketInfo bucket_info_257_2048{/* sp_lower_bound = */ 1,
                                    /* sp_upper_bound = */ kMaxNumel,
                                    /* rb_lower_bound = */ 257,
                                    /* rb_upper_bound = */ 2048,
                                    /* sp_is_dynamic = */ false,
                                    /* rb_is_dynamic = */ true};
    ScheduleConfig::TileConfig tile_config_257_2048{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 128,
        /* spatial_inner_num = */ 1,
        /* reduce_method = */ BlockReduceMethod()};

    BucketInfo bucket_info_2049_INF{/* sp_lower_bound = */ 1,
                                    /* sp_upper_bound = */ kMaxNumel,
                                    /* rb_lower_bound = */ 2049,
                                    /* rb_upper_bound = */ kMaxNumel,
                                    /* sp_is_dynamic = */ false,
                                    /* rb_is_dynamic = */ true};
    ScheduleConfig::TileConfig tile_config_2049_INF{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 256,
        /* spatial_inner_num = */ 1,
        /* reduce_method = */ BlockReduceMethod()};

    return {{bucket_info_1_256, tile_config_1_256},
            {bucket_info_257_2048, tile_config_257_2048},
            {bucket_info_2049_INF, tile_config_2049_INF}};
  }
}

std::unordered_map<BucketInfo, ScheduleConfig::TileConfig, BucketInfoHash>
BuildStaticReduceConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  if (base_info->reduce_numel == 1) {
    BucketInfo bucket_info__1_1023{/* sp_lower_bound = */ 1,
                                   /* sp_upper_bound = */ 1023,
                                   /* rb_lower_bound = */ 1,
                                   /* rb_upper_bound = */ 1,
                                   /* sp_is_dynamic = */ true,
                                   /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config__1_1023{
        /* warp_num = */ -1,
        /* tree_reduce_num = */ 1,
        /* spatial_inner_num = */ 1,
        /* reduce_method = */ NoneReduceMethod()};
    BucketInfo bucket_info__1024_1M{/* sp_lower_bound = */ 1024,
                                    /* sp_upper_bound = */ 1024 * 1024 - 1,
                                    /* rb_lower_bound = */ 1,
                                    /* rb_upper_bound = */ 1,
                                    /* sp_is_dynamic = */ true,
                                    /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config__1024_1M{
        /* warp_num = */ 32,
        /* tree_reduce_num = */ 1,
        /* spatial_inner_num = */ 4,
        /* reduce_method = */ NoneReduceMethod()};
    BucketInfo bucket_info__1M_INF{/* sp_lower_bound = */ 1024 * 1024,
                                   /* sp_upper_bound = */ kMaxNumel,
                                   /* rb_lower_bound = */ 1,
                                   /* rb_upper_bound = */ 1,
                                   /* sp_is_dynamic = */ true,
                                   /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config__1M_INF{
        /* warp_num = */ 32,
        /* tree_reduce_num = */ 1,
        /* spatial_inner_num = */ 4,
        /* reduce_method = */ NoneReduceMethod()};
    return {{bucket_info__1_1023, tile_config__1_1023},
            {bucket_info__1024_1M, tile_config__1024_1M},
            {bucket_info__1M_INF, tile_config__1M_INF}};
  } else if (base_info->reduce_numel <= 256) {
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 2,
                           /* rb_upper_bound = */ 256,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 32,
        /* spatial_inner_num = */ (256 / Next2Power(base_info->reduce_numel)),
        /* reduce_method = */ WarpReduceMethod()};
    return {{bucket_info, tile_config}};
  } else if (base_info->reduce_numel <= 2048) {
    int64_t reduce_block =
        int64_t(std::ceil(base_info->reduce_numel * 1.0 / 256.0)) * 256;
    int64_t warp_num = reduce_block / 256;
    int64_t spatial_inner_num = 1;
    int64_t reduce_inner_num = 8;
    int64_t tree_reduce_num = reduce_block / reduce_inner_num;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 257,
                           /* rb_upper_bound = */ 2048,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ tree_reduce_num,
        /* spatial_inner_num = */ spatial_inner_num,
        /* reduce_method = */ BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  } else {
    int64_t warp_num = 32;
    int64_t spatial_inner_num = 1;
    int64_t tree_reduce_num = warp_num * 32;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 2049,
                           /* rb_upper_bound = */ kMaxNumel,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    ScheduleConfig::TileConfig tile_config{
        /* warp_num = */ warp_num,
        /* tree_reduce_num = */ tree_reduce_num,
        /* spatial_inner_num = */ spatial_inner_num,
        /* reduce_method = */ BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  }
}

std::unordered_map<BucketInfo, ScheduleConfig::TileConfig, BucketInfoHash>
BuildDynamicShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  int64_t warp_num = 8;
  int64_t spatial_inner_num = 1;
  int64_t tree_reduce_num = warp_num * 32;
  BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                         /* sp_upper_bound = */ kMaxNumel,
                         /* rb_lower_bound = */ 1,
                         /* rb_upper_bound = */ kMaxNumel,
                         /* sp_is_dynamic = */ true,
                         /* rb_is_dynamic = */ true};
  ScheduleConfig::TileConfig tile_config{
      /* warp_num = */ warp_num,
      /* tree_reduce_num = */ tree_reduce_num,
      /* spatial_inner_num = */ spatial_inner_num,
      /* reduce_method = */ BlockReduceMethod()};
  return {{bucket_info, tile_config}};
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
CombineBaseInfoAndConfig(
    const std::unordered_map<BucketInfo,
                             ScheduleConfig::TileConfig,
                             BucketInfoHash>& config_map,
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info) {
  std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash> combined;
  for (const auto& bucket_config : config_map) {
    ScheduleConfig sch_config{base_info, std::move(bucket_config.second)};
    combined.insert({std::move(bucket_config.first), std::move(sch_config)});
  }
  return combined;
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
BuildScheduleConfig(
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info,
    const common::Target& target) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      InitBasicInfo(group_info);
  if (!base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static sptial and static reduce config.";
    return CombineBaseInfoAndConfig(
        BuildPureStaticShapeConfig(base_info, target), base_info);
  } else if (base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static sptial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticSpatialConfig(base_info, target),
                                    base_info);
  } else if (!base_info->has_dynamic_reduce && base_info->has_dynamic_spatial) {
    VLOG(6) << "Building dynamic sptial and static reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticReduceConfig(base_info, target),
                                    base_info);
  } else {  // (base_info->has_dynamic_reduce && base_info->has_dynamic_spatial)
    VLOG(6) << "Building dynamic spatial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildDynamicShapeConfig(base_info, target),
                                    base_info);
  }
}

}  // namespace ir
}  // namespace cinn
