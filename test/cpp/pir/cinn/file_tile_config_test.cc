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
#include "paddle/cinn/ir/group_schedule/config/file_database.h"
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
PD_DECLARE_string(cinn_tile_config_filename_label);
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
bool PathExists(const std::string& path) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
  return false;
}

void RemoveDir(const cinn::common::Target target,
               const cinn::ir::IterSpaceType& iter_space_type) {
  std::string dirname = "";
  std::string filename = "";
  for (auto i : iter_space_type) {
    dirname += i.first;
    dirname += "_";
    filename += i.first + i.second;
    filename += "_";
  }
  const std::string kDirSuffix = "_EREBE";
  dirname = dirname.substr(0, dirname.size() - 1) + kDirSuffix;
  filename = filename.substr(0, filename.size() - 1);

  auto removedir = [](const std::string& test_path,
                      const std::string& file_name) {
    if (PathExists(test_path)) {
      std::string full_test_name = test_path + file_name;
      std::remove(full_test_name.c_str());
      LOG(INFO) << "File exist.";
    } else {
      LOG(INFO) << "File doesn't exist.";
    }
  };
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  const std::string kTestFileDir = "./tile_file_test/";
  if (root_path == "") {
    root_path = kTestFileDir;
  }
  std::string target_str = target.arch_str() + "_" + target.device_name_str();
  std::string file_name = "/" + filename + ".json";
  removedir(root_path + target_str + "/" + dirname, file_name);
  LOG(INFO) << "Dump_path "
            << root_path + target_str + "/" + dirname + " has been removed";
}

TEST(ConfigSearcher, TestReduceDemo) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Step 1: Construct iter space and tile config.
  cinn::ir::BucketInfo bucket_info;
  int s_dimension_lower = 13;
  int s_dimension_upper = 13;
  auto s_dimension_type = "S";
  auto s_dimension_is_dynamic = true;
  int r_dimension_lower = 4096;
  int r_dimension_upper = 4096;
  auto r_dimension_type = "R";
  auto r_dimension_is_dynamic = true;

  bucket_info.space.push_back(
      cinn::ir::BucketInfo::Dimension{s_dimension_lower,
                                      s_dimension_upper,
                                      s_dimension_type,
                                      s_dimension_is_dynamic});
  bucket_info.space.push_back(
      cinn::ir::BucketInfo::Dimension{r_dimension_lower,
                                      r_dimension_upper,
                                      r_dimension_type,
                                      r_dimension_is_dynamic});

  cinn::ir::ScheduleConfig::TileConfig tile_config;
  tile_config.spatial_inner_num = 9;
  tile_config.warp_num = 14;
  tile_config.tree_reduce_num = 512;
  // Use kTestFileDir in this test.
  const std::string prev_flag = FLAGS_cinn_tile_config_filename_label;
  const std::string kTestFileDir = "./tile_file_test/";
  FLAGS_cinn_tile_config_filename_label = kTestFileDir;
  std::vector<std::pair<std::string, std::string>> iter_space_type = {
      std::make_pair(s_dimension_type,
                     s_dimension_is_dynamic == true ? "dynamic" : "static"),
      std::make_pair(r_dimension_type,
                     r_dimension_is_dynamic == true ? "dynamic" : "static")};
  // If test file has been created, remove it.
  RemoveDir(cinn::common::DefaultTarget(), iter_space_type);
  // Step 2: Add to json / Read from json
  cinn::ir::FileTileConfigDatabase file_database;
  file_database.AddConfig(
      cinn::common::DefaultTarget(), bucket_info, tile_config, 2);
  cinn::ir::TileConfigMap tile_config_map =
      file_database.GetConfigs(cinn::common::DefaultTarget(), iter_space_type);
  // Delete the file
  RemoveDir(cinn::common::DefaultTarget(), iter_space_type);
  // Check the correctness
  for (auto& it : tile_config_map) {
    LOG(INFO) << "bucket info is: ";
    auto dims = it.first.space.size();
    for (int i = 0; i < dims; i++) {
      LOG(INFO) << "Dimension " << i
                << " 's lower_bound is: " << it.first.space[i].lower_bound;
      LOG(INFO) << "Dimension " << i
                << " 's upper_bound is: " << it.first.space[i].upper_bound;
      auto dimension_lower = i == 0 ? s_dimension_lower : r_dimension_lower;
      auto dimension_upper = i == 0 ? s_dimension_upper : r_dimension_upper;
      PADDLE_ENFORCE_EQ(it.first.space[i].lower_bound,
                        dimension_lower,
                        ::common::errors::InvalidArgument(
                            "GetConfigs function gets wrong dimension_lower"));
      PADDLE_ENFORCE_EQ(it.first.space[i].upper_bound,
                        dimension_upper,
                        ::common::errors::InvalidArgument(
                            "GetConfigs function gets wrong dimension_upper"));
    }
    LOG(INFO) << "tile config is " << it.second.spatial_inner_num << " "
              << it.second.warp_num << " " << it.second.tree_reduce_num;
    PADDLE_ENFORCE_EQ(it.second.spatial_inner_num,
                      tile_config.spatial_inner_num,
                      ::common::errors::InvalidArgument(
                          "GetConfigs function gets wrong spatial_inner_num"));
    PADDLE_ENFORCE_EQ(it.second.warp_num,
                      tile_config.warp_num,
                      ::common::errors::InvalidArgument(
                          "GetConfigs function gets wrong warp_num"));
    PADDLE_ENFORCE_EQ(it.second.tree_reduce_num,
                      tile_config.tree_reduce_num,
                      ::common::errors::InvalidArgument(
                          "GetConfigs function gets wrong tree_reduce_num"));
  }
  // Restore the previous flag
  FLAGS_cinn_tile_config_filename_label = prev_flag;
}
