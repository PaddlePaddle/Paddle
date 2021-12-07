// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/utils/table_printer.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace paddle {
namespace inference {}  // namespace inference
}  // namespace paddle

TEST(table_printer, output) {
  std::vector<std::string> header{"config", "value"};
  paddle::inference::TablePrinter table(header);

  // model_dir
  table.InsertRow({"model_dir", "./model_dir"});
  // model
  table.InsertRow({"model_file", "./model.pdmodel"});
  table.InsertRow({"params_file", "./model.pdiparams"});

  table.InsetDivider();
  // gpu
  table.InsertRow({"use_gpu", "true"});
  table.InsertRow({"gpu_device_id", "0"});
  table.InsertRow({"memory_pool_init_size", "100MB"});
  table.InsertRow({"thread_local_stream", "false"});
  table.InsetDivider();

  // trt precision
  table.InsertRow({"use_trt", "true"});
  table.InsertRow({"trt_precision", "fp32"});
  table.InsertRow({"enable_dynamic_shape", "true"});
  table.InsertRow({"DisableTensorRtOPs", "{}"});
  table.InsertRow({"EnableTensorRtOSS", "ON"});
  table.InsertRow({"tensorrt_dla_enabled", "ON"});
  table.InsetDivider();

  // lite
  table.InsertRow({"use_lite", "ON"});
  table.InsetDivider();

  // xpu
  table.InsertRow({"use_xpu", "true"});
  table.InsertRow({"xpu_device_id", "0"});
  table.InsetDivider();

  // ir
  table.InsertRow({"ir_optim", "true"});
  table.InsertRow({"ir_debug", "false"});
  table.InsertRow({"enable_memory_optim", "false"});
  table.InsertRow({"EnableProfile", "false"});
  table.InsertRow({"glog_info_disabled", "false"});
  table.InsetDivider();

  // cpu
  table.InsertRow({"CpuMathLibrary", "4"});
  // mkldnn
  table.InsertRow({"enable_mkldnn", "false"});
  table.InsertRow({"mkldnn_cache_capacity", "10"});

  // a long string
  table.InsertRow(
      {"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a long string "
       "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
       "------------------------------------------ a long value "
       "-----------------------------------------------------"});

  LOG(INFO) << table.PrintTable();
}
