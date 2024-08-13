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
  if (dimension_lower < 32) {
    return 30;
  } else if (dimension_lower < 128) {
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
