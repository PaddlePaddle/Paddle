// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/config/filedatabase.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/group_schedule/search/config_searcher.h"
#include "paddle/cinn/ir/group_schedule/search/measurer.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

COMMON_DECLARE_bool(print_ir);

TEST(ConfigSearcher, TestReduceDemo) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Step 1: Construct iter space and tile config.
  cinn::ir::search::IterSpace iter_space;
  iter_space.space.push_back(cinn::ir::search::IterSpace::Dimension{
      33,
      128,
      "S",
      /* is_dynamic = */ true,
      std::vector<double>(128 - 32, 1.0)});
  iter_space.space.push_back(
      cinn::ir::search::IterSpace::Dimension{1024,
                                             1024,
                                             "R",
                                             /* is_dynamic = */ false,
                                             std::vector<double>(1, 1.0)});
  cinn::ir::BucketInfo bucket_info;
  bucket_info.sp_lower_bound = iter_space.space[0].lower_bound;
  bucket_info.sp_upper_bound = iter_space.space[0].upper_bound;
  bucket_info.rb_lower_bound = iter_space.space[1].lower_bound;
  bucket_info.rb_upper_bound = iter_space.space[1].upper_bound;
  cinn::ir::ScheduleConfig::TileConfig tile_config;
  tile_config.spatial_inner_num = 32;
  tile_config.warp_num = 32;
  tile_config.tree_reduce_num = 128;
  std::vector<std::pair<std::string, std::string>> iter_space_type = {
      std::make_pair("R", "dynamic"), std::make_pair("S", "dynamic")};
  // Step 2: Add to json/Read from json
  cinn::ir::FileTileConfigDatabase file_database;
  file_database.AddConfig(cinn::common::DefaultTarget(),
                          iter_space_type,
                          bucket_info,
                          tile_config,
                          2);
  cinn::ir::TileConfigMap tile_config_map =
      file_database.GetConfigs(cinn::common::DefaultTarget(), iter_space_type);
  for (auto& it : tile_config_map) {
    LOG(INFO) << "rb_lower_bound is " << it.first.rb_lower_bound;
    LOG(INFO) << "rb_upper_bound is " << it.first.rb_upper_bound;
    LOG(INFO) << "tile config is " << it.second.spatial_inner_num << " "
              << it.second.warp_num << " " << it.second.tree_reduce_num;
  }
}
