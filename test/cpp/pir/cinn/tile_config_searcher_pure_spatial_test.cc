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
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
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
PD_DECLARE_bool(cinn_measure_kernel_time);
PHI_DECLARE_bool(enable_cinn_compile_cache);
PD_DECLARE_string(tile_config_policy);
constexpr int kThreadsPerWarp = 32;
constexpr int kMaxThreadsPerBlock = 1024;
// now each has the same weight
constexpr double s_w = 0.05;
constexpr double sampling_prob = 1.0;
constexpr int kMaxSamplingTimes = 300;
constexpr int kRepeats = 3;

// PureSpatial
std::shared_ptr<::pir::Program> BuildSpatialProgram(int spatial_size) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> shape = {spatial_size, 1};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto x_pow = builder.Build<paddle::dialect::PowOp>(x, 2.0).result(0);

  auto add_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1e-6, phi::DataType::FLOAT32)
          .out();
  auto x_add = builder.Build<paddle::dialect::AddOp>(add_num, x_pow).result(0);

  auto divide_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1024, phi::DataType::FLOAT32)
          .out();
  auto x_div =
      builder.Build<paddle::dialect::DivideOp>(x_add, divide_num).result(0);

  builder.Build<paddle::dialect::FetchOp>(x_div, "out", 0);
  return program;
}

// Get the tile size configuration for the given dimension lower bound
// dynamically.
int get_tile_size_config(int dimension_lower) {
  if (dimension_lower <= 2) {
    return 126;
  } else if (dimension_lower <= 128) {
    return 384;
  } else if (dimension_lower <= 512) {
    return 512;
  } else if (dimension_lower <= 1024) {
    return 1024;
  } else if (dimension_lower <= 2048) {
    return 2048;
  } else if (dimension_lower <= 4096) {
    return 4096;
  } else if (dimension_lower <= 8192) {
    return 8192;
  } else if (dimension_lower <= 16384) {
    return 16384;
  }
}

std::vector<cinn::ir::search::ConstraintFunc> GetSConstraints(
    int s_dimension_lower) {
  std::vector<cinn::ir::search::ConstraintFunc> constraints;
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp <= kMaxThreadsPerBlock;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[1] < 256 && candidate[1] % kThreadsPerWarp == 0 ||
               candidate[1] == 1 ||
               candidate[1] <= 512 && candidate[1] % 128 == 0 ||
               candidate[1] % 256 == 0;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp >= candidate[1];
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp % candidate[1] == 0;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[2] == 1 || candidate[2] == 2 || candidate[2] == 4 ||
               candidate[2] == 8 || candidate[2] == 16 || candidate[2] == 32;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] <= 4 ||
               candidate[0] <= 8 && candidate[0] % 2 == 0 ||
               candidate[0] % 4 == 0;
      });
  return constraints;
}

// Create Bucket Info
cinn::ir::BucketInfo CreateSBucket(int s_dimension_lower,
                                   int spatial_tile_width,
                                   bool is_spatial_dynamic) {
  cinn::ir::BucketInfo bucket_info;
  bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
      s_dimension_lower,
      s_dimension_lower + spatial_tile_width - 1,
      "S",
      /* is_dynamic = */ is_spatial_dynamic});
  return bucket_info;
}

// Search One Window for corresponding pattern
void SearchSWindow(bool is_spatial_dynamic,
                   int s_dimension_lower,
                   int spatial_tile_width,
                   int spatial_tile_config,
                   double s_weight) {
  std::vector<double> s_weights =
      std::vector<double>(spatial_tile_width, s_weight);
  // Step 1: Construct pir::Program.
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program_pure_spatial;
  if (!is_spatial_dynamic) {
    program_pure_spatial = BuildSpatialProgram(s_dimension_lower);
  } else {
    program_pure_spatial = BuildSpatialProgram(-1);
  }

  // Step 2: Switch schedule config manager mode.
  auto& schedule_config_manager = cinn::ir::ScheduleConfigManager::Instance();

  // Step 3: Construct iter space and objective function.
  cinn::ir::BucketInfo bucket_info;
  bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
      s_dimension_lower,
      s_dimension_lower + spatial_tile_width - 1,
      "S",
      /* is_dynamic = */ is_spatial_dynamic});
  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func_pure_spatial =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program_pure_spatial.get(),
          bucket_info,
          sampling_prob,
          kMaxSamplingTimes,
          kRepeats,
          std::vector<std::vector<double>>{s_weights});

  std::vector<std::unique_ptr<cinn::ir::search::BaseObjectiveFunc>>
      objective_funcs;
  objective_funcs.emplace_back(std::move(obj_func_pure_spatial));
  // Step 4: Construct config candidate range and constraints.
  std::vector<std::pair<int, int>> candidate_range{
      {1, 1}, {1, 1}, {1, 1}};  // {1, 32}, {1, 1}, {1, 32}
  std::vector<cinn::ir::search::ConstraintFunc> constraints =
      GetSConstraints(s_dimension_lower);
  // Step 5: Construct searcher and search.
  cinn::ir::search::ScheduleConfigSearcher searcher(
      std::move(objective_funcs), candidate_range, constraints);
  auto search_res = searcher.Search();

  // Step 6: Save the best candidate's config of each grid search to json
  cinn::ir::FileTileConfigDatabase file_database;
  cinn::ir::ScheduleConfig::TileConfig tile_bestconfig;
  tile_bestconfig.warp_num = search_res.second[0];
  tile_bestconfig.tree_reduce_num = search_res.second[1];
  tile_bestconfig.spatial_inner_num = search_res.second[2];
  // Extend bucketinfo 's static dim region
  if (bucket_info.space[0].is_dynamic == false &&
      bucket_info.space[0].lower_bound == bucket_info.space[0].upper_bound) {
    bucket_info.space[0].upper_bound =
        s_dimension_lower + spatial_tile_config - 1;
  }

  // Extend bucketinfo 's large value to infinite
  if (spatial_tile_config == 16384) {
    bucket_info.space[0].upper_bound = static_cast<int>(2e10);
  }
  file_database.AddConfig(
      cinn::common::DefaultTarget(), bucket_info, tile_bestconfig, 0);

  LOG(INFO) << "spatial tile dimension lower bound = " << s_dimension_lower
            << std::endl;
  LOG(INFO) << "min score = " << search_res.first;
  LOG(INFO) << "best candidate: "
            << cinn::utils::Join<int64_t>(search_res.second, ", ");
}
/**
 * @brief Test case for the ConfigSearcher.
 *
 */
void SearchForSTileConfig(int spatial_l_bound,
                          int spatial_r_bound,
                          bool is_s_dynamic) {
  FLAGS_cinn_measure_kernel_time = true;
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_tile_config_policy = "search";

  // Define the search space bounds and sampling probabilities.
  int spatial_left_bound = spatial_l_bound;
  int spatial_right_bound = spatial_r_bound;  // for easy test, set to 2. for
                                              // the whole test, set to 32767
  bool is_spatial_dynamic = is_s_dynamic;

  // Define the initial grid size for the spatial and reduction dimensions
  int spatial_tile_config = 0, reduce_tile_config = 0;
  int spatial_tile_width = 0, reduce_tile_width = 0;
  // Define weight for each dimension
  double s_weight = (is_spatial_dynamic ? s_w : 1.0);
  // Search for every window
  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower < spatial_right_bound ||
       s_dimension_lower == spatial_right_bound &&
           spatial_left_bound == spatial_right_bound;
       s_dimension_lower += spatial_tile_config) {
    // adjust the tile size for the spatial dimension dymaically
    spatial_tile_config = get_tile_size_config(s_dimension_lower);
    spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
    SearchSWindow(is_spatial_dynamic,
                  s_dimension_lower,
                  spatial_tile_width,
                  spatial_tile_config,
                  s_weight);
  }
}

TEST(ConfigSearcher, TestPureSpatialstatic) {
  int spatial_left_bound = 2;
  int spatial_right_bound = 2;  // To reproduce, set it to 32767
  bool is_spatial_dynamic = false;

  SearchForSTileConfig(
      spatial_left_bound, spatial_right_bound, is_spatial_dynamic);
}

TEST(ConfigSearcher, TestPureSpatialDynamic) {
  int spatial_left_bound = 2;
  int spatial_right_bound = 2;  // To reproduce, set it to 32767
  bool is_spatial_dynamic = true;

  SearchForSTileConfig(
      spatial_left_bound, spatial_right_bound, is_spatial_dynamic);
}
