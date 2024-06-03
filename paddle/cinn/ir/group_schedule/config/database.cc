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

#include "paddle/cinn/ir/group_schedule/config/database.h"

namespace cinn {
namespace ir {

void NaiveTileConfigDatabase::AddConfig(
    const common::Target& target,
    const BucketInfo& bucket_info,
    const ScheduleConfig::TileConfig& config,
    int priority) {
  IterSpaceType iter_space_type = [&] {
    std::vector<std::pair<std::string, std::string>> res;
    for (const auto& dim : bucket_info.space) {
      res.emplace_back(dim.iter_type, (dim.is_dynamic ? "dynamic" : "static"));
    }
    return res;
  }();
  config_map_[iter_space_type][bucket_info] = config;
}

TileConfigMap NaiveTileConfigDatabase::GetConfigs(
    const common::Target& target, const IterSpaceType& iter_space_type) const {
  if (config_map_.count(iter_space_type) == 0) {
    std::stringstream ss;
    ss << "[";
    for (const auto& item : iter_space_type) {
      ss << "[" << item.first << ", " << item.second << "], ";
    }
    ss << "]";
    return {};
  }
  return config_map_.at(iter_space_type);
}

ScheduleConfigManager& ScheduleConfigManager::Instance() {
  static ScheduleConfigManager schedule_config_manager;
  return schedule_config_manager;
}

void ScheduleConfigManager::AddConfigDatabase(
    const std::string& id,
    const std::shared_ptr<TileConfigDatabase>& database) {
  tile_config_data_[id] = database;
}

ScheduleConfigMap ScheduleConfigManager::ExtractConfigs(
    const common::Target& target,
    const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info) const {
  if (policy_ == "default" || tile_config_data_.count(policy_) == 0) {
    return BuildScheduleConfig(group_info, target);
  }
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      InitBasicInfo(group_info);
  IterSpaceType iter_space_type = [&] {
    std::string sp_state =
        base_info->has_dynamic_spatial ? "dynamic" : "static";
    std::string rd_state = base_info->has_dynamic_reduce ? "dynamic" : "static";
    return IterSpaceType{{"S", sp_state}, {"R", rd_state}};
  }();

  TileConfigMap tile_config_map =
      tile_config_data_.at(policy_)->GetConfigs(target, iter_space_type);
  return CombineBaseInfoAndConfig(tile_config_map, base_info);
}

void ScheduleConfigManager::SetPolicy(const std::string& policy) {
  policy_ = policy;
}

}  // namespace ir
}  // namespace cinn
