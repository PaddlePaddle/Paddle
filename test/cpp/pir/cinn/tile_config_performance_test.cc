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
#include <sys/stat.h>
#include <memory>
#include <sstream>

#include <fstream>
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

PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_tile_config_filename_label);
COMMON_DECLARE_bool(print_ir);
PD_DECLARE_bool(cinn_measure_kernel_time);
PHI_DECLARE_bool(enable_cinn_compile_cache);

#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)

std::string GenCSVFilePath(const cinn::common::Target target,
                           const cinn::ir::IterSpaceType& iter_space_type) {
  std::string dirname = "";
  std::string filename = "";
  for (auto i : iter_space_type) {
    dirname += i.first;
    dirname += "_";
    filename += i.first + i.second;
    filename += "_";
  }
  dirname = dirname.substr(0, dirname.size() - 1);
  filename = filename.substr(0, filename.size() - 1);

  auto checkexist = [](std::string test_path) {
    bool path_exists = false;
    struct stat statbuf;
    if (stat(test_path.c_str(), &statbuf) != -1) {
      if (S_ISDIR(statbuf.st_mode)) {
        path_exists = true;
      }
    }
    if (!path_exists) {
      PADDLE_ENFORCE_NE(MKDIR(test_path.c_str()),
                        -1,
                        ::common::errors::PreconditionNotMet(
                            "Can not create directory: %s, Make sure you "
                            "have permission to write",
                            test_path));
    }
  };
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  std::string target_str = target.arch_str() + "_" + target.device_name_str();
  checkexist(root_path);
  checkexist(root_path + target_str);
  checkexist(root_path + target_str + "/" + dirname);
  VLOG(3) << "Dump_path is " << root_path + dirname + "/" + filename + ".csv";
  return root_path + target_str + "/" + dirname + "/" + filename + ".csv";
}

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
 * This test case performs a perormance test for the best configuration derived
 * from the ConfigSearcher. It iterates over different spatial and reduce tile
 * sizes and constructs a pir::Program. The objective function used for the
 * search is a WeightedSamplingTrailObjectiveFunc. The performance results are
 * logged, including the default score and the best candidate score.
 */
TEST(ConfigSearcher, TestReducePipeline) {
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_cinn_measure_kernel_time = true;

  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Define the search space bounds and sampling probabilities.
  constexpr int spatial_left_bound = 32;
  constexpr int spatial_right_bound = 32;
  constexpr int reduce_left_bound = 32;
  constexpr int reduce_right_bound = 32;
  constexpr bool is_spatial_dynamic = false;
  constexpr bool is_reduce_dynamic = true;
  // now each has the same weight
  constexpr double s_w = 0.05;
  constexpr double r_w = 0.05;
  constexpr double sampling_prob = 1.0;
  constexpr int kMaxSamplingTimes = 560;
  constexpr int kRepeats = 32;

  // Define the initial grid size for the spatial and reduction dimensions
  int spatial_tile_config = 0, reduce_tile_config = 0;
  int spatial_tile_width = 0, reduce_tile_width = 0;
  // Define weight for each dimension
  double s_weight = (is_spatial_dynamic ? s_w : 1.0);
  double r_weight = (is_reduce_dynamic ? r_w : 1.0);

  auto s_dimension_type = "S";
  auto r_dimension_type = "R";

  // Get best configuration from json by file database.
  std::vector<std::pair<std::string, std::string>> iter_space_type = {
      std::make_pair(s_dimension_type,
                     is_spatial_dynamic == true ? "dynamic" : "static"),
      std::make_pair(r_dimension_type,
                     is_reduce_dynamic == true ? "dynamic" : "static")};
  cinn::ir::FileTileConfigDatabase file_database;
  cinn::ir::TileConfigMap best_tile_config_map =
      file_database.GetConfigs(cinn::common::DefaultTarget(), iter_space_type);

  // build csv file
  std::string dump_path =
      GenCSVFilePath(cinn::common::DefaultTarget(), iter_space_type);
  std::ofstream os(dump_path, std::ofstream::app);
  PADDLE_ENFORCE_EQ(os.good(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Cannot open the file to write:  %s", dump_path));

  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower <= spatial_right_bound;
       s_dimension_lower += spatial_tile_config) {
    // adjust the tile size for the spatial dimension dymaically
    spatial_tile_config = get_tile_size_config(s_dimension_lower);
    spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
    if (s_dimension_lower == 4096 && is_spatial_dynamic) {
      spatial_tile_width = 5820 + 1 - s_dimension_lower;  // 1048576 = 2^20
    }
    for (int r_dimension_lower = reduce_left_bound;
         r_dimension_lower <= reduce_right_bound;
         r_dimension_lower += reduce_tile_config) {
      // adjust the tile size for the reduce dimension dymaically
      reduce_tile_config = get_tile_size_config(r_dimension_lower);
      reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);
      if (r_dimension_lower == 4096 && is_reduce_dynamic) {
        reduce_tile_width = 5820 + 1 - r_dimension_lower;  // 1048576 = 2^20
      }

      std::vector<double> s_weights =
          std::vector<double>(spatial_tile_width, s_weight);
      std::vector<double> r_weights =
          std::vector<double>(reduce_tile_width, r_weight);

      LOG(INFO) << "spatial tile dimension lower bound = " << s_dimension_lower
                << ", reduce tile dimension lower bound = " << r_dimension_lower
                << std::endl;
      // Construct pir::Program.
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

      // Construct iter space and objective function.
      cinn::ir::BucketInfo bucket_info;
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          s_dimension_lower,
          s_dimension_lower + spatial_tile_width - 1,
          "S",
          /* is_dynamic = */ is_spatial_dynamic});
      bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
          r_dimension_lower,
          r_dimension_lower + reduce_tile_width - 1,
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

      cinn::ir::search::CandidateType best_candidate;
      for (auto& it : best_tile_config_map) {
        auto s_flag = false, r_flag = false;
        auto dims = it.first.space.size();
        // SR type support only
        if (dims == 2) {
          if (it.first.space[0].lower_bound <= s_dimension_lower &&
              it.first.space[0].upper_bound >=
                  s_dimension_lower + spatial_tile_width - 1) {
            s_flag = true;
          } else if (it.first.space[0].lower_bound == 4096 &&
                     s_dimension_lower == 4096) {
            s_flag = true;
          }
          if (it.first.space[1].lower_bound <= r_dimension_lower &&
              it.first.space[1].upper_bound >=
                  r_dimension_lower + reduce_tile_width - 1) {
            r_flag = true;
          } else if (it.first.space[1].lower_bound == 4096 &&
                     r_dimension_lower == 4096) {
            r_flag = true;
          }
        } else {
          PADDLE_THROW(
              ::common::errors::Unavailable("Now just support SR type."));
        }
        if (s_flag == true && r_flag == true) {
          best_candidate = {it.second.warp_num,
                            it.second.tree_reduce_num,
                            it.second.spatial_inner_num};
          break;
        }
      }
      if (best_candidate.empty()) {
        PADDLE_THROW(
            ::common::errors::Unavailable("Not found the best candidate."));
      }

      // Use the config in group_tile_config.cc
      cinn::ir::search::ScoreType best_score = 0;
      cinn::ir::search::CandidateType default_candidate;
      cinn::ir::search::ScoreType baseline_score = 0;
      for (int i = 0; i < 5; i++) {
        FLAGS_tile_config_policy = "default";
        cinn::ir::search::ScoreType temp_baseline_score =
            (*obj_func)(default_candidate);
        FLAGS_tile_config_policy = "search";
        cinn::ir::search::ScoreType temp_best_score =
            (*obj_func)(best_candidate);
        best_score = (best_score == 0) ? temp_best_score
                                       : std::min(best_score, temp_best_score);
        baseline_score = (baseline_score == 0)
                             ? temp_baseline_score
                             : std::min(baseline_score, temp_baseline_score);
      }
      LOG(INFO) << "Best score: " << best_score;
      LOG(INFO) << "Baseline score: " << baseline_score;
      LOG(INFO) << "Best candidate: " << best_candidate[0] << " "
                << best_candidate[1] << " " << best_candidate[2];
      // Write to csv file
      cinn::ir::search::ScoreType optim_percentage =
          (1 / best_score - 1 / baseline_score) * baseline_score;
      std::stringstream ss;
      ss << " { ";
      for (int i = 0; i < iter_space_type.size(); ++i) {
        ss << iter_space_type[i].first << "_" << iter_space_type[i].second
           << ": " << bucket_info.space[i].lower_bound << "-"
           << bucket_info.space[i].upper_bound << " ";
      }
      ss << "} ";
      os << ss.str() << " \t ";
      os << std::setprecision(3) << "  " << baseline_score << " \t "
         << best_score << " \t " << optim_percentage << "\n";
    }
  }
  os.close();
}
