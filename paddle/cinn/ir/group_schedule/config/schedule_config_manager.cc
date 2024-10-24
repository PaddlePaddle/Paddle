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

#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/ir/group_schedule/config/file_database.h"

PD_DECLARE_string(tile_config_policy);

namespace cinn {
namespace ir {

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
    const std::shared_ptr<FusionGroupInfo>& group_info) const {
  auto ReadConfigs = [&](std::string policy) -> ScheduleConfigMap {
    std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
        InitBasicInfo(group_info);
    TileConfigMap tile_config_map = tile_config_data_.at(policy)->GetConfigs(
        target, base_info->iter_space_type);
    return CombineBaseInfoAndConfig(tile_config_map, base_info);
  };

  if (policy_ == "default" || tile_config_data_.count(policy_) == 0) {
    return BuildScheduleConfig(group_info, target);
  } else if (policy_ == "hybrid") {
    ScheduleConfigMap default_map = BuildScheduleConfig(group_info, target);
    ScheduleConfigMap stored_map = ReadConfigs(policy_);
    stored_map.insert(default_map.begin(), default_map.end());
    return stored_map;
  } else {
    VLOG(3) << "Enter policy branch: " << policy_;
    return ReadConfigs(policy_);
  }
}

void ScheduleConfigManager::SetPolicy(const std::string& policy) {
  policy_ = policy;
}

void InitScheduleConfig() {
  auto& schedule_config_manager = cinn::ir::ScheduleConfigManager::Instance();
  std::string policy;
  policy = FLAGS_tile_config_policy;
  schedule_config_manager.SetPolicy(policy);
  if (policy == "optimal" || policy == "hybrid") {
    std::shared_ptr<cinn::ir::TileConfigDatabase> tile_config_database =
        std::make_shared<cinn::ir::FileTileConfigDatabase>();
    schedule_config_manager.AddConfigDatabase(policy, tile_config_database);
  }
}

}  // namespace ir
}  // namespace cinn
