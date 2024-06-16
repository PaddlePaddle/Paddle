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
  if (dimension_lower < 128) {
    return 32;
  } else if (dimension_lower < 512) {
    return 128;
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
TEST(ConfigSearcher, TestReduceDemo) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Define the search space bounds and sampling probabilities.
  constexpr int spatial_left_bound = 32;
  constexpr int spatial_right_bound = 32;
  constexpr int reduce_left_bound = 32;
  constexpr int reduce_right_bound = 32;
  constexpr bool is_spatial_dynamic = false;
  constexpr bool is_reduce_dynamic = true;
  constexpr sampling_prob_sp = 0.05;
  constexpr sampling_prob_rd = 0.05;

  // Define the initial grid size for the spatial and reduction dimensions of
  // the search space table.
  int spatial_tile_config =
      (is_spatial_dynamic ? get_tile_size_config(spatial_left_bound)
                          : get_tile_size_config(spatial_left_bound) - 1);
  int reduce_tile_config =
      (is_reduce_dynamic ? get_tile_size_config(reduce_left_bound)
                         : get_tile_size_config(reduce_left_bound) - 1);
  int spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
  int reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);
  double spatial_sampling_prob = (is_spatial_dynamic ? sampling_prob_sp : 1.0);
  double reduce_sampling_prob = (is_reduce_dynamic ? sampling_prob_rd : 1.0);

  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower <= spatial_right_bound;
       s_dimension_lower += spatial_tile_config) {
    // adjust the tile size for the spatial dimension dymaically
    spatial_tile_config = get_tile_size_config(s_dimension_lower);
    spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
    for (int r_dimension_lower = reduce_left_bound;
         r_dimension_lower <= reduce_right_bound;
         r_dimension_lower += reduce_tile_config) {
      // adjust the tile size for the reduce dimension dymaically
      reduce_tile_config = get_tile_size_config(r_dimension_lower);
      reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);

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

      // Step 2: Switch schedule config manager mode.
      auto& schedule_config_manager =
          cinn::ir::ScheduleConfigManager::Instance();
      schedule_config_manager.SetPolicy("custom");

      // Step 3: Construct iter space and objective function.
      cinn::ir::BucketInfo bucket_info;
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          s_dimension_lower,
          s_dimension_lower + spatial_tile_width - 1,
          "S",
          /* is_dynamic = */ is_spatial_dynamic,
          std::vector<double>(spatial_tile_width, spatial_sampling_prob)});
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          r_dimension_lower,
          r_dimension_lower + reduce_tile_width - 1,
          "R",
          /* is_dynamic = */ is_reduce_dynamic,
          std::vector<double>(reduce_tile_width, reduce_sampling_prob)});
      std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func =
          std::make_unique<
              cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
              program.get(), bucket_info);

      // Step 4: Construct config candidate range and constraints.
      std::vector<std::pair<int, int>> candidate_range{
          {1, 8}, {1, 256}, {1, 256}};
      std::vector<cinn::ir::search::ConstraintFunc> constraints;
      constraints.emplace_back(
          [](const cinn::ir::search::CandidateType& candidate) -> bool {
            return candidate[1] % kThreadsPerWarp == 0;
          });
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
            return candidate[2] % 8 == 0;
          });

      // Step 5: Construct searcher and search.
      cinn::ir::search::ScheduleConfigSearcher searcher(
          std::move(obj_func), candidate_range, constraints);
      auto search_res = searcher.Search();

      // Step 6: Write the best candidate's config of each grid search to json
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
                << cinn::utils::Join<int>(search_res.second, ", ");
    }
  }
}
