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
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"

namespace cinn {

namespace hlir::framework::pir {
struct GroupInfo;
}  // namespace hlir::framework::pir

namespace ir {

struct ScheduleConfig {
  struct BaseInfo {
    std::vector<int64_t> reduce_axis;
    int64_t data_rank;
    int64_t reduce_numel;
    int64_t spatial_numel;
    bool has_dynamic_spatial{false};
    bool has_dynamic_reduce{false};
    bool is_reduce_all{false};

    std::set<std::string> reduce_tensor_names;
    std::set<std::string> temp_var_names;
    std::set<std::string> shared_var_names;
    std::set<std::string> direct_output_var_names;

    std::unordered_map<std::string, BroadcastInfo> broadcast_info;
    std::unordered_map<std::string, BroadcastInfo> broadcast_to_elementwise;
  };

  struct TileConfig {
    int64_t warp_num{1};
    int64_t tree_reduce_num{1};
    int64_t spatial_inner_num{1};
    ReduceMethod reduce_method{NoneReduceMethod()};
  };

  std::shared_ptr<BaseInfo> base_info;
  TileConfig tile_config;
};

struct BucketInfo {
  int64_t sp_lower_bound = 1;
  int64_t sp_upper_bound = INT64_MAX;
  int64_t rb_lower_bound = 1;
  int64_t rb_upper_bound = INT64_MAX;

  bool operator==(const BucketInfo& other) const {
    return this->sp_lower_bound == other.sp_lower_bound &&
           this->sp_upper_bound == other.sp_upper_bound &&
           this->rb_lower_bound == other.rb_lower_bound &&
           this->rb_upper_bound == other.rb_upper_bound;
  }
};

struct BucketInfoHash {
  std::size_t operator()(const BucketInfo& bucket_info) const noexcept {
    std::size_t hash_spl = std::hash<uint64_t>{}(bucket_info.sp_lower_bound);
    std::size_t hash_spu = std::hash<uint64_t>{}(bucket_info.sp_upper_bound);
    std::size_t hash_rbl = std::hash<uint64_t>{}(bucket_info.rb_lower_bound);
    std::size_t hash_rbu = std::hash<uint64_t>{}(bucket_info.rb_upper_bound);
    return adt::hash_combine(adt::hash_combine(hash_spl, hash_spu),
                             adt::hash_combine(hash_rbl, hash_rbu));
  }
};

std::shared_ptr<ScheduleConfig::BaseInfo> InitBasicInfo(
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info);

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
CombineBaseInfoAndConfig(
    const std::unordered_map<BucketInfo,
                             ScheduleConfig::TileConfig,
                             BucketInfoHash>& config_map,
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info);

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
BuildScheduleConfig(
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info,
    const common::Target& target);

}  // namespace ir
}  // namespace cinn
