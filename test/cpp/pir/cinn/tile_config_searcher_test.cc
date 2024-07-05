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

std::shared_ptr<::pir::Program> BuildReduceSumProgram(int spatial_size,
                                                      int reduce_size) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {spatial_size, reduce_size};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto out = builder
                 .Build<paddle::dialect::SumOp>(
                     x, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
                 .result(0);
  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// Get the tile size configuration for the given dimension lower bound
// dynamically.
int get_tile_size_config(int dimension_lower) {
  if (dimension_lower < 256) {
    return 64;
  } else if (dimension_lower < 1024) {
    return 256;
  } else if (dimension_lower < 2048) {
    return 512;
  } else {
    return 1024;
  }
}

/**
 * @brief Test case for the ConfigSearcher.
 *
 * This test case performs a search for the best configuration using the
 * ConfigSearcher. It iterates over different spatial and reduce tile sizes and
 * constructs a pir::Program. The search is performed using a
 * ScheduleConfigSearcher, which takes into account candidate ranges and
 * constraints. The objective function used for the search is a
 * WeightedSamplingTrailObjectiveFunc. The search results are logged, including
 * the minimum score and the best candidate configuration found.
 */
TEST(ConfigSearcher, TestSRReducePipeline) {
  FLAGS_cinn_measure_kernel_time = true;
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_tile_config_policy = "search";

  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Define the search space bounds and sampling probabilities.
  constexpr int spatial_left_bound = 32;
  constexpr int spatial_right_bound = 32;
  constexpr int reduce_left_bound = 32;
  constexpr int reduce_right_bound = 32;
  constexpr bool is_spatial_dynamic = true;
  constexpr bool is_reduce_dynamic = false;
  // now each has the same weight
  constexpr double s_w = 0.05;
  constexpr double r_w = 0.05;
  constexpr double sampling_prob = 1.0;
  constexpr int kMaxSamplingTimes = 1000;
  constexpr int kRepeats = 80;

  // Define the initial grid size for the spatial and reduction dimensions
  int s_bucket_increasing_width = 0, r_bucket_increasing_width = 0;
  int s_bucket_width = 0, r_bucket_width = 0;
  // Define weight for each dimension
  double s_weight = (is_spatial_dynamic ? s_w : 1.0);
  double r_weight = (is_reduce_dynamic ? r_w : 1.0);

  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower <= spatial_right_bound;
       s_dimension_lower += s_bucket_increasing_width) {
    // adjust the tile size for the spatial dimension dymaically
    s_bucket_increasing_width = get_tile_size_config(s_dimension_lower);
    s_bucket_width = (is_spatial_dynamic ? s_bucket_increasing_width : 1);
    for (int r_dimension_lower = reduce_left_bound;
         r_dimension_lower <= reduce_right_bound;
         r_dimension_lower += r_bucket_increasing_width) {
      // adjust the tile size for the reduce dimension dymaically
      r_bucket_increasing_width = get_tile_size_config(r_dimension_lower);
      r_bucket_width = (is_reduce_dynamic ? r_bucket_increasing_width : 1);

      std::vector<double> s_weights =
          std::vector<double>(s_bucket_width, s_weight);
      std::vector<double> r_weights =
          std::vector<double>(r_bucket_width, r_weight);

      // Step 1: Construct pir::Program.
      ::pir::IrContext* ctx = ::pir::IrContext::Instance();
      std::shared_ptr<::pir::Program> program;
      if (!is_spatial_dynamic && !is_reduce_dynamic) {
        program = BuildReduceSumProgram(s_dimension_lower, r_dimension_lower);
      } else if (is_spatial_dynamic && !is_reduce_dynamic) {
        program = BuildReduceSumProgram(-1, r_dimension_lower);
      } else if (!is_spatial_dynamic && is_reduce_dynamic) {
        program = BuildReduceSumProgram(s_dimension_lower, -1);
      } else {
        program = BuildReduceSumProgram(-1, -1);
      }

      // Step 2: Construct iter space and objective function.
      cinn::ir::BucketInfo bucket_info;
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          s_dimension_lower,
          s_dimension_lower + s_bucket_width - 1,
          "S",
          /* is_dynamic = */ is_spatial_dynamic});
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          r_dimension_lower,
          r_dimension_lower + r_bucket_width - 1,
          "R",
          /* is_dynamic = */ is_reduce_dynamic});
      std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func =
          std::make_unique<
              cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
              program.get(),
              bucket_info,
              sampling_prob,
              kMaxSamplingTimes,
              kRepeats,
              std::vector<std::vector<double>>{s_weights, r_weights});

      // Step 3: Construct config candidate range and constraints.
      std::vector<std::pair<int, int>> candidate_range{
          {1, 1}, {1, 1}, {1, 1}};  // {1, 32}, {1, 1024}, {1, 128}
      std::vector<cinn::ir::search::ConstraintFunc> constraints;
      constraints.emplace_back(
          [](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[0] * kThreadsPerWarp <= kMaxThreadsPerBlock;
          });
      constraints.emplace_back(
          [](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[1] % kThreadsPerWarp == 0 || candidate[1] == 1;
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
          [&](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[0] * kThreadsPerWarp / candidate[1] *
                       candidate[2] <=
                   s_dimension_lower;
          });
      constraints.emplace_back(
          [&](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[1] <= r_dimension_lower;
          });
      constraints.emplace_back(
          [](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[2] == 1 || candidate[2] == 2 ||
                   candidate[2] == 4 ||
                   candidate[2] >= 8 && candidate[2] < 32 &&
                       candidate[2] % 8 == 0 ||
                   candidate[2] >= 32 && candidate[2] % 32 == 0;
          });

      // Step 4: Construct searcher and search.
      cinn::ir::search::ScheduleConfigSearcher searcher(
          std::move(obj_func), candidate_range, constraints);
      auto search_res = searcher.Search();

      // Step 5: Save the best candidate's config of each grid search to json
      cinn::ir::FileTileConfigDatabase file_database;
      cinn::ir::ScheduleConfig::TileConfig tile_bestconfig;
      tile_bestconfig.warp_num = search_res.second[0];
      tile_bestconfig.tree_reduce_num = search_res.second[1];
      tile_bestconfig.spatial_inner_num = search_res.second[2];
      file_database.AddConfig(
          cinn::common::DefaultTarget(), bucket_info, tile_bestconfig, 0);

      LOG(INFO) << "spatial tile dimension lower bound = " << s_dimension_lower
                << ", reduce tile dimension lower bound = " << r_dimension_lower
                << std::endl;
      LOG(INFO) << "min score = " << search_res.first;
      LOG(INFO) << "best candidate: "
                << cinn::utils::Join<int64_t>(search_res.second, ", ");
    }
  }
}

TEST(ConfigSearcher, TestReduceDemo) {
  FLAGS_cinn_measure_kernel_time = true;
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_tile_config_policy = "search";

  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  std::vector<std::vector<int64_t>> shapes = {
      {1, 1, 4096},          // 0
      {1, 13, 4096},         // 1
      {128 * 12, 128, 128},  // 2
      {128, 128, 768},       // 3
      {2048, 32, 128},       // 4
      {2048, 8, 96},         // 5
      {13 * 2048, 32, 128}   // 6
  };
  auto shape = shapes[0];
  int shape0 = shape[0], shape1 = shape[1], shape2 = shape[2];
  // Step 1: Construct pir::Program.
  std::shared_ptr<::pir::Program> program =
      BuildReduceSumProgram(shape0 * shape1, shape2);
  // Step 2: Switch schedule config manager mode.
  auto& schedule_config_manager = cinn::ir::ScheduleConfigManager::Instance();
  // Step 3: Construct iter space and objective function.
  int s_dimension_lower = shape0 * shape1;
  int s_dimension_upper = shape0 * shape1;
  auto s_dimension_type = "S";
  auto s_dimension_is_dynamic = false;
  int r_dimension_lower = shape2;
  int r_dimension_upper = shape2;
  auto r_dimension_type = "R";
  auto r_dimension_is_dynamic = false;

  auto s_dimension = cinn::ir::BucketInfo::Dimension{s_dimension_lower,
                                                     s_dimension_upper,
                                                     s_dimension_type,
                                                     s_dimension_is_dynamic};
  auto r_dimension = cinn::ir::BucketInfo::Dimension{r_dimension_lower,
                                                     r_dimension_upper,
                                                     r_dimension_type,
                                                     r_dimension_is_dynamic};

  cinn::ir::BucketInfo bucket_info(
      std::vector<cinn::ir::BucketInfo::Dimension>{s_dimension, r_dimension});
  LOG(INFO) << "Bucket_info.space.size is: " << bucket_info.space.size();
  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program.get(), bucket_info);

  // Step 4: Construct config candidate range and constraints.
  std::vector<std::pair<int, int>> candidate_range{
      {1, 1}, {32, 32}, {1, 1}};  // {1, 8}, {1, 256}, {1, 256}
  std::vector<cinn::ir::search::ConstraintFunc> constraints;
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[1] % kThreadsPerWarp == 0 || candidate[1] == 1;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp <= kMaxThreadsPerBlock;
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
      [&](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * 32 / candidate[1] * candidate[2] <=
               s_dimension_lower;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[2] < 8 || candidate[2] % 8 == 0;
      });
  constraints.emplace_back(
      [&](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[1] <= r_dimension_upper;
      });

  // Step 5: Construct searcher and search.
  cinn::ir::search::ScheduleConfigSearcher searcher(
      std::move(obj_func), candidate_range, constraints);
  auto search_res = searcher.Search();
  LOG(INFO) << "min score = " << search_res.first;
  LOG(INFO) << "best candidate: "
            << cinn::utils::Join<int64_t>(search_res.second, ", ");

  // Step 6: Save the config to the file.
  cinn::ir::ScheduleConfig::TileConfig tile_config{
      search_res.second[0], search_res.second[1], search_res.second[2]};

  cinn::ir::FileTileConfigDatabase file_database;
  file_database.AddConfig(
      cinn::common::DefaultTarget(), bucket_info, tile_config, -1);
}
