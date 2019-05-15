/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/benchmark/op_test.h"
#include <fstream>
#include "gflags/gflags.h"
#include "gtest/gtest.h"

DEFINE_string(op_config_list, "", "Path of op config file.");
DEFINE_int32(specified_config_id, -1, "Test the specified op config.");

namespace paddle {
namespace operators {
namespace benchmark {

TEST(op_tester, base) {
  if (!FLAGS_op_config_list.empty()) {
    std::ifstream fin(FLAGS_op_config_list, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s",
                   FLAGS_op_config_list.c_str());
    std::vector<OpTesterConfig> op_configs;
    while (!fin.eof()) {
      VLOG(4) << "Reading config " << op_configs.size() << "...";
      OpTesterConfig config;
      bool result = config.Init(fin);
      if (result) {
        op_configs.push_back(config);
      }
    }
    if (FLAGS_specified_config_id >= 0 &&
        FLAGS_specified_config_id < static_cast<int>(op_configs.size())) {
      OpTest tester;
      tester.Init(op_configs[FLAGS_specified_config_id]);
      tester.Run();
    } else {
      for (size_t i = 0; i < op_configs.size(); ++i) {
        OpTest tester;
        tester.Init(op_configs[i]);
        tester.Run();
      }
    }
  } else {
    OpTest tester;
    OpTesterConfig config;
    config.op_type = "elementwise_add";
    config.inputs.resize(2);
    config.inputs[0].name = "X";
    config.inputs[0].dims = {64, 64};
    config.inputs[1].name = "Y";
    config.inputs[1].dims = {64, 1};
    tester.Init(config);
    tester.Run();
  }
}
}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
