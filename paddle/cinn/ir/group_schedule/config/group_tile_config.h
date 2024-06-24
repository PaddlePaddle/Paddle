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

using IterSpaceType = std::vector<std::pair<std::string, std::string>>;

struct ScheduleConfig {
  struct BaseInfo {
    std::vector<int64_t> reduce_axis;
    std::vector<int64_t> raw_reduce_axis;
    int64_t data_rank;
    int64_t raw_data_rank;
    int64_t reduce_numel;
    int64_t spatial_numel;
    bool has_dynamic_spatial{false};
    bool has_dynamic_reduce{false};
    bool is_reduce_all{false};
    IterSpaceType iter_space_type;

    std::set<std::string> reduce_tensor_names;
    std::set<std::string> temp_var_names;
    std::set<std::string> shared_var_names;
    std::set<std::string> direct_output_var_names;
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
    Dimension()
        : lower_bound(0),
          upper_bound(INT_MAX),
          iter_type("S"),
          is_dynamic(false) {}
    Dimension(int low, int upper, std::string iter_type, bool is_dynamic)
        : lower_bound(low),
          upper_bound(upper),
          iter_type(iter_type),
          is_dynamic(is_dynamic) {}
  };
  std::vector<Dimension> space;
  int bucket_priority = 100;

  std::string ToString() const;
  BucketInfo() = default;
  BucketInfo(int sp_lower_bound,
             int sp_upper_bound,
             int rb_lower_bound,
             int rb_upper_bound,
             bool sp_is_dynamic,
             bool rb_is_dynamic);
  explicit BucketInfo(size_t size) : space(std::vector<Dimension>(size)) {}
  explicit BucketInfo(const std::vector<Dimension>& dims);
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
      for (int i = 1; i < dims; i++) {
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
