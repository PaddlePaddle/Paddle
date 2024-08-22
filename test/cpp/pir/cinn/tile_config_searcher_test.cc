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
PD_DECLARE_string(cinn_tile_config_filename_label);
constexpr int kThreadsPerWarp = 32;
constexpr int kMaxThreadsPerBlock = 1024;
// now each has the same weight
constexpr double s_w = 0.05;
constexpr double r_w = 0.05;
constexpr double sampling_prob = 1.0;
constexpr int kMaxSamplingTimes = 300;
constexpr int kRepeats = 3;

// layernorm
std::shared_ptr<::pir::Program> BuildLayerNormProgram(int spatial_size,
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
  auto sum_val =
      builder
          .Build<paddle::dialect::SumOp>(
              x, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);

  auto divide_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1024, phi::DataType::FLOAT32)
          .out();
  auto mean_val =
      builder.Build<paddle::dialect::DivideOp>(sum_val, divide_num).result(0);
  auto sub_num =
      builder.Build<paddle::dialect::SubtractOp>(x, mean_val).result(0);
  auto pow_val = builder.Build<paddle::dialect::PowOp>(x, 2.0).result(0);
  auto pow_sum =
      builder
          .Build<paddle::dialect::SumOp>(
              pow_val, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto variance_val =
      builder.Build<paddle::dialect::DivideOp>(pow_sum, divide_num).result(0);
  auto add_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1e-6, phi::DataType::FLOAT32)
          .out();
  auto add_val =
      builder.Build<paddle::dialect::AddOp>(variance_val, add_num).result(0);
  auto rsqrt_val = builder.Build<paddle::dialect::RsqrtOp>(add_val).result(0);
  auto out =
      builder.Build<paddle::dialect::MultiplyOp>(rsqrt_val, sub_num).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// rmsnorm
std::shared_ptr<::pir::Program> BuildRMSNormProgram(int spatial_size,
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
  auto pow_val = builder.Build<paddle::dialect::PowOp>(x, 2.0).result(0);
  auto sum_val =
      builder
          .Build<paddle::dialect::SumOp>(
              pow_val, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto divide_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), reduce_size, phi::DataType::FLOAT32)
          .out();
  auto div_val =
      builder.Build<paddle::dialect::DivideOp>(sum_val, divide_num).result(0);
  auto add_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1e-6, phi::DataType::FLOAT32)
          .out();
  auto add_val =
      builder.Build<paddle::dialect::AddOp>(div_val, add_num).result(0);
  auto rsqrt_val = builder.Build<paddle::dialect::RsqrtOp>(add_val).result(0);
  auto out = builder.Build<paddle::dialect::MultiplyOp>(rsqrt_val, x).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// Get the tile size configuration for the given dimension lower bound
// dynamically.
int get_tile_size_config_in_small_area(int dimension_lower) {
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
  }
}

int get_tile_size_config_in_large_area(int dimension_lower) {
  if (dimension_lower <= 2) {
    return 510;
  } else if (dimension_lower <= 512) {
    return 512;
  } else if (dimension_lower <= 4096) {
    return 4096;
  } else if (dimension_lower <= 8192) {
    return 8192;
  } else if (dimension_lower <= 16384) {
    return 16384;
  }
}

int get_spatial_range(int s_dimension_lower, int r_dimension_lower) {
  int compute_size = s_dimension_lower * r_dimension_lower;
  if (compute_size <= 1024 * 1024) {
    return 1;
  } else if (compute_size <= 1024 * 2048) {
    return 2;
  } else if (compute_size <= 2048 * 2048) {
    return 4;
  } else if ((s_dimension_lower > 4096) || (r_dimension_lower > 4096)) {
    return 8;
  }
  return 1;
}

void search_then_save_one_window(bool is_spatial_dynamic,
                                 bool is_reduce_dynamic,
                                 int s_dimension_lower,
                                 int r_dimension_lower,
                                 int spatial_tile_width,
                                 int reduce_tile_width,
                                 int spatial_tile_config,
                                 int reduce_tile_config,
                                 double s_weight,
                                 double r_weight) {
  std::vector<double> s_weights =
      std::vector<double>(spatial_tile_width, s_weight);
  std::vector<double> r_weights =
      std::vector<double>(reduce_tile_width, r_weight);
  // Step 1: Construct pir::Program.
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program_layer_norm;
  std::shared_ptr<::pir::Program> program_rms_norm;
  if (!is_spatial_dynamic && !is_reduce_dynamic) {
    program_layer_norm =
        BuildLayerNormProgram(s_dimension_lower, r_dimension_lower);
  } else if (is_spatial_dynamic && !is_reduce_dynamic) {
    program_layer_norm = BuildLayerNormProgram(-1, r_dimension_lower);
  } else if (!is_spatial_dynamic && is_reduce_dynamic) {
    program_layer_norm = BuildLayerNormProgram(s_dimension_lower, -1);
  } else {
    program_layer_norm = BuildLayerNormProgram(-1, -1);
  }

  if (!is_spatial_dynamic && !is_reduce_dynamic) {
    program_rms_norm =
        BuildRMSNormProgram(s_dimension_lower, r_dimension_lower);
  } else if (is_spatial_dynamic && !is_reduce_dynamic) {
    program_rms_norm = BuildRMSNormProgram(-1, r_dimension_lower);
  } else if (!is_spatial_dynamic && is_reduce_dynamic) {
    program_rms_norm = BuildRMSNormProgram(s_dimension_lower, -1);
  } else {
    program_rms_norm = BuildRMSNormProgram(-1, -1);
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
  bucket_info.space.push_back(
      cinn::ir::BucketInfo::Dimension{r_dimension_lower,
                                      r_dimension_lower + reduce_tile_width - 1,
                                      "R",
                                      /* is_dynamic = */ is_reduce_dynamic});
  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func_layernorm =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program_layer_norm.get(),
          bucket_info,
          sampling_prob,
          kMaxSamplingTimes,
          kRepeats,
          std::vector<std::vector<double>>{s_weights, r_weights});

  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func_rmsnorm =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program_rms_norm.get(),
          bucket_info,
          sampling_prob,
          kMaxSamplingTimes,
          kRepeats,
          std::vector<std::vector<double>>{s_weights, r_weights});

  std::vector<std::unique_ptr<cinn::ir::search::BaseObjectiveFunc>>
      objective_funcs;
  objective_funcs.emplace_back(std::move(obj_func_layernorm));
  objective_funcs.emplace_back(std::move(obj_func_rmsnorm));

  // Step 4: Construct config candidate range and constraints.
  std::vector<std::pair<int, int>> candidate_range{
      {1, 1}, {1, 32}, {1, 1}};  // {1, 32}, {1, 1024}, {1, 8}
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
      [&](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp / candidate[1] * candidate[2] <=
               s_dimension_lower;
      });
  constraints.emplace_back(
      [&](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[2] <=
               get_spatial_range(s_dimension_lower, r_dimension_lower);
      });
  constraints.emplace_back(
      [&](const cinn::ir::search::CandidateType& candidate) -> bool {
        return r_dimension_lower % candidate[1] == 0 || candidate[1] == 32 ||
               candidate[1] == 64;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[2] <= candidate[1];
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[2] == 1 || candidate[2] == 2 || candidate[2] == 4 ||
               candidate[2] == 8;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] <= 4 ||
               candidate[0] <= 8 && candidate[0] % 2 == 0 ||
               candidate[0] % 4 == 0;
      });

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
  if (bucket_info.space[1].is_dynamic == false &&
      bucket_info.space[1].lower_bound == bucket_info.space[1].upper_bound) {
    bucket_info.space[1].upper_bound =
        r_dimension_lower + reduce_tile_config - 1;
  }
  // Extend bucketinfo 's large value to infinite
  if (spatial_tile_config == 1000) {
    bucket_info.space[0].upper_bound = static_cast<int>(2e10);
  }
  if (reduce_tile_config == 1000) {
    bucket_info.space[1].upper_bound = static_cast<int>(2e10);
  }

  file_database.AddConfig(
      cinn::common::DefaultTarget(), bucket_info, tile_bestconfig, 0);

  LOG(INFO) << "spatial tile dimension lower bound = " << s_dimension_lower
            << ", reduce tile dimension lower bound = " << r_dimension_lower
            << std::endl;
  LOG(INFO) << "min score = " << search_res.first;
  LOG(INFO) << "best candidate: "
            << cinn::utils::Join<int64_t>(search_res.second, ", ");
  if (r_dimension_lower * s_dimension_lower >= (2048 * 1024)) {
    sleep(15);
  } else {
    sleep(2);
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
void TestSearchForTileConfig(int spatial_l_bound,
                             int spatial_r_bound,
                             int reduce_l_bound,
                             int reduce_r_bound,
                             bool is_s_dynamic,
                             bool is_r_dynamic,
                             bool search_single_large) {
  FLAGS_cinn_measure_kernel_time = true;
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_tile_config_policy = "search";
  // set tile_file path to test path when user use default setting
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  if (root_path == "") {
    const std::string kTestFileDir = "./tile_file_test/";
    FLAGS_cinn_tile_config_filename_label = kTestFileDir;
  }

  // Define the search space bounds and sampling probabilities.
  int spatial_left_bound = spatial_l_bound;
  int spatial_right_bound = spatial_r_bound;  // for easy test, set to 2. for
                                              // the whole test, set to 4096
  int reduce_left_bound = reduce_l_bound;
  int reduce_right_bound = reduce_r_bound;  // for easy test : set to 2. for the
                                            // whole test, set to 4096
  bool is_spatial_dynamic = is_s_dynamic;
  bool is_reduce_dynamic = is_r_dynamic;

  // Define the initial grid size for the spatial and reduction dimensions
  int spatial_tile_config = 0, reduce_tile_config = 0;
  int spatial_tile_width = 0, reduce_tile_width = 0;
  // Define weight for each dimension
  double s_weight = (is_spatial_dynamic ? s_w : 1.0);
  double r_weight = (is_reduce_dynamic ? r_w : 1.0);
  // (I) Search in the small area,
  // i.e, S:[2-4096]*R:[2-4096]
  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower < spatial_right_bound ||
       s_dimension_lower == spatial_right_bound &&
           spatial_left_bound == spatial_right_bound;
       s_dimension_lower += spatial_tile_config) {
    // adjust the tile size for the spatial dimension dymaically
    spatial_tile_config = get_tile_size_config_in_small_area(s_dimension_lower);
    spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
    for (int r_dimension_lower = reduce_left_bound;
         r_dimension_lower < reduce_right_bound ||
         r_dimension_lower == reduce_right_bound &&
             reduce_left_bound == reduce_right_bound;
         r_dimension_lower += reduce_tile_config) {
      // adjust the tile size for the reduce dimension dymaically
      reduce_tile_config =
          get_tile_size_config_in_small_area(r_dimension_lower);
      reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);
      search_then_save_one_window(is_spatial_dynamic,
                                  is_reduce_dynamic,
                                  s_dimension_lower,
                                  r_dimension_lower,
                                  spatial_tile_width,
                                  reduce_tile_width,
                                  spatial_tile_config,
                                  reduce_tile_config,
                                  s_weight,
                                  r_weight);
    }
  }

  if (search_single_large) {
    // (II) Search in the single large areas,
    // i.e., S:[4096-32768]*R:[2-1024], S:[2-1024]*R:[4096-32768]
    for (int s_dimension_lower = 2; s_dimension_lower < 1024;
         s_dimension_lower += spatial_tile_config) {
      // adjust the tile size for the spatial dimension dymaically
      spatial_tile_config =
          get_tile_size_config_in_large_area(s_dimension_lower);
      spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);

      for (int r_dimension_lower = 4096; r_dimension_lower < 32768;
           r_dimension_lower += reduce_tile_config) {
        // adjust the tile size for the reduce dimension dymaically
        reduce_tile_config =
            get_tile_size_config_in_large_area(r_dimension_lower);
        reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);

        search_then_save_one_window(is_spatial_dynamic,
                                    is_reduce_dynamic,
                                    s_dimension_lower,
                                    r_dimension_lower,
                                    spatial_tile_width,
                                    reduce_tile_width,
                                    spatial_tile_config,
                                    reduce_tile_config,
                                    s_weight,
                                    r_weight);
      }
    }

    for (int s_dimension_lower = 4096; s_dimension_lower < 32768;
         s_dimension_lower += spatial_tile_config) {
      // adjust the tile size for the spatial dimension dymaically
      spatial_tile_config =
          get_tile_size_config_in_large_area(s_dimension_lower);
      spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);

      for (int r_dimension_lower = 2; r_dimension_lower < 1024;
           r_dimension_lower += reduce_tile_config) {
        // adjust the tile size for the reduce dimension dymaically
        reduce_tile_config =
            get_tile_size_config_in_large_area(r_dimension_lower);
        reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);

        search_then_save_one_window(is_spatial_dynamic,
                                    is_reduce_dynamic,
                                    s_dimension_lower,
                                    r_dimension_lower,
                                    spatial_tile_width,
                                    reduce_tile_width,
                                    spatial_tile_config,
                                    reduce_tile_config,
                                    s_weight,
                                    r_weight);
      }
    }
  }
}

TEST(ConfigSearcher, TestDynamicDynamic) {
  int spatial_left_bound = 2;
  int spatial_right_bound = 2;  // To reproduce, set it to 4096
  int reduce_left_bound = 2;
  int reduce_right_bound = 2;  // To reproduce, set it to 4096
  bool is_spatial_dynamic = true;
  bool is_reduce_dynamic = true;
  bool search_single_large =
      false;  // To search rsingle large area, set it to true
  TestSearchForTileConfig(spatial_left_bound,
                          spatial_right_bound,
                          reduce_left_bound,
                          reduce_right_bound,
                          is_spatial_dynamic,
                          is_reduce_dynamic,
                          search_single_large);
}

TEST(ConfigSearcher, TestDynamicReduce) {
  int spatial_left_bound = 2;
  int spatial_right_bound = 2;  // To reproduce, set it to 4096
  int reduce_left_bound = 2;
  int reduce_right_bound = 2;  // To reproduce, set it to 4096
  bool is_spatial_dynamic = false;
  bool is_reduce_dynamic = true;
  bool search_single_large =
      false;  // To search rsingle large area, set it to true
  TestSearchForTileConfig(spatial_left_bound,
                          spatial_right_bound,
                          reduce_left_bound,
                          reduce_right_bound,
                          is_spatial_dynamic,
                          is_reduce_dynamic,
                          search_single_large);
}

TEST(ConfigSearcher, TestDynamicSpatial) {
  int spatial_left_bound = 2;
  int spatial_right_bound = 2;  // To reproduce, set it to 4096
  int reduce_left_bound = 2;
  int reduce_right_bound = 2;  // To reproduce, set it to 4096
  bool is_spatial_dynamic = true;
  bool is_reduce_dynamic = false;
  bool search_single_large =
      false;  // To search single large area, set it to true
  TestSearchForTileConfig(spatial_left_bound,
                          spatial_right_bound,
                          reduce_left_bound,
                          reduce_right_bound,
                          is_spatial_dynamic,
                          is_reduce_dynamic,
                          search_single_large);
}
