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
  struct Dimension {
    int lower_bound;
    int upper_bound;
    std::string iter_type;
    bool is_dynamic;
    std::vector<double> weights;
    Dimension(int low, int upper, std::string iter_type, bool is_dynamic)
        : lower_bound(low),
          upper_bound(upper),
          iter_type(iter_type),
          is_dynamic(is_dynamic) {}
    Dimension(int low,
              int upper,
              std::string iter_type,
              bool is_dynamic,
              std::vector<double> weights)
        : lower_bound(low),
          upper_bound(upper),
          iter_type(iter_type),
          is_dynamic(is_dynamic),
          weights(weights) {}
  };
  std::vector<Dimension> space;

  std::string ToString() const;
  BucketInfo(int sp_lower_bound,
             int sp_upper_bound,
             int rb_lower_bound,
             int rb_upper_bound,
             bool sp_is_dynamic,
             bool rb_is_dynamic) BucketInfo(size_t size)
      : space(std::vector<Dimension>(size)) {}
  bool operator==(const BucketInfo& other) const;
};

struct BucketInfoHash {
  std::size_t operator()(const BucketInfo& bucket_info) const noexcept {
    PADDLE_ENFORCE_GT(
        bucket_info.space.size(),
        0,
        ::common::errors::InvalidArgument(
            "Bucketinfo 's dimension number should be more than 0"));

    std::size_t hash_past_dims = adt::hash_combine(
        std::hash<uint64_t>{}(bucket_info.space[0].lower_bound),
        std::hash<uint64_t>{}(bucket_info.space[0].upper_bound));
    int dims = bucket_info.space.size();
    if (dims == 1) {
      return hash_past_dims;
    } else {
      for (int i = 0; i < dims; i++) {
        std::size_t hash_temp_dim = adt::hash_combine(
            std::hash<uint64_t>{}(bucket_info.space[i].lower_bound),
            std::hash<uint64_t>{}(bucket_info.space[i].upper_bound));
        hash_past_dims = adt::hash_combine(hash_past_dims, hash_temp_dim);
      }
      return hash_past_dims;
    }
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
