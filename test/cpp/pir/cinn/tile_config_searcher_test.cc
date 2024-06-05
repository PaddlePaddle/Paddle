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

std::shared_ptr<::pir::Program> BuildReduceSumProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {1, 13, 4096};
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

TEST(ConfigSearcher, TestReduceDemo) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // Step 1: Construct pir::Program.
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildReduceSumProgram();

  // Step 2: Switch schedule config manager mode.
  auto& schedule_config_manager = cinn::ir::ScheduleConfigManager::Instance();
  schedule_config_manager.SetPolicy("custom");

  // Step 3: Construct iter space and objective function.
  cinn::ir::BucketInfo bucket_info;
  int s_dimension_lower = 13;
  int s_dimension_upper = 13;
  auto s_dimension_type = "S";
  auto s_dimension_is_dynamic = false;
  int r_dimension_lower = 4096;
  int r_dimension_upper = 4096;
  auto r_dimension_type = "R";
  auto r_dimension_is_dynamic = false;

  bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
      s_dimension_lower,
      s_dimension_upper,
      s_dimension_type,
      s_dimension_is_dynamic,
      std::vector<double>(s_dimension_upper - s_dimension_lower + 1, 1.0)});
  bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
      r_dimension_lower,
      r_dimension_upper,
      r_dimension_type,
      r_dimension_is_dynamic,
      std::vector<double>(r_dimension_upper - r_dimension_lower + 1, 1.0)});

  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program.get(), bucket_info);

  // Step 4: Construct config candidate range and constraints.
  std::vector<std::pair<int, int>> candidate_range{{1, 32}, {1, 1024}, {1, 1}};
  std::vector<cinn::ir::search::ConstraintFunc> constraints;
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[1] % kThreadsPerWarp == 0 || candidate[1] == 1;
      });
  constraints.emplace_back(
      [](const cinn::ir::search::CandidateType& candidate) -> bool {
        return candidate[0] * kThreadsPerWarp <= kMaxThreadsPerBlock;
      });

  // Step 5: Construct searcher and search.
  cinn::ir::search::ScheduleConfigSearcher searcher(
      std::move(obj_func), candidate_range, constraints);
  auto search_res = searcher.Search();
  LOG(INFO) << "min score = " << search_res.first;
  LOG(INFO) << "best candidate: "
            << cinn::utils::Join<int>(search_res.second, ", ");
}
