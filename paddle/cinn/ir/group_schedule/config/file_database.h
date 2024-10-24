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

#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/config/tile_config_desc.pb.h"
namespace cinn {
namespace ir {

class FileTileConfigDatabase final : public TileConfigDatabase {
 public:
  void AddConfig(const common::Target& target,
                 const BucketInfo& bucket_info,
                 const ScheduleConfig::TileConfig& config,
                 int priority) override;
  TileConfigMap GetConfigs(const common::Target& target,
                           const IterSpaceType& iter_space_type) const override;

 private:
  TileConfigMap target_config_data_;
  bool ToFile(const common::Target& target, int priority);
};

}  // namespace ir
}  // namespace cinn
