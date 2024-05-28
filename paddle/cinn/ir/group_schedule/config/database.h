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

#include <string>
#include <unordered_map>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"

namespace cinn {
namespace ir {

using ScheduleConfigMap =
    std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>;
using TileConfigMap =
    std::unordered_map<BucketInfo, ScheduleConfig::TileConfig, BucketInfoHash>;
using IterSpaceType = std::vector<std::pair<std::string, std::string>>;

class TileConfigDatabase {
 public:
  virtual void AddConfig(const common::Target& target,
                         const IterSpaceType& iter_space_type,
                         const BucketInfo& bucket_info,
                         const ScheduleConfig::TileConfig& config,
                         int priority) = 0;

  virtual TileConfigMap GetConfigs(
      const common::Target& target,
      const IterSpaceType& iter_space_type) const = 0;
};

class NaiveTileConfigDatabase final : public TileConfigDatabase {
 public:
  void AddConfig(const common::Target& target,
                 const IterSpaceType& iter_space_type,
                 const BucketInfo& bucket_info,
                 const ScheduleConfig::TileConfig& config,
                 int priority = 1) override;

  TileConfigMap GetConfigs(const common::Target& target,
                           const IterSpaceType& iter_space_type) const override;

 private:
  std::map<IterSpaceType, TileConfigMap> config_map_;
};

class ScheduleConfigManager {
 public:
  static ScheduleConfigManager& Instance();

  void AddConfigDatabase(const std::string& id,
                         const std::shared_ptr<TileConfigDatabase>& database);

  ScheduleConfigMap ExtractConfigs(
      const common::Target& target,
      const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info) const;

  void SetPolicy(const std::string& policy);

 private:
  ScheduleConfigManager() = default;
  ~ScheduleConfigManager() = default;
  ScheduleConfigManager(const ScheduleConfigManager&) = delete;
  void operator=(const ScheduleConfigManager&) = delete;

 private:
  std::unordered_map<std::string, std::shared_ptr<TileConfigDatabase>>
      tile_config_data_;
  std::string policy_ = "default";
};

}  // namespace ir
}  // namespace cinn
