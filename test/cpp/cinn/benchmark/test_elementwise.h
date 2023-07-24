// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "test/cpp/cinn/benchmark/test_utils.h"

namespace cinn {
namespace tests {

class ElementwiseAddTester : public OpBenchmarkTester {
 public:
  ElementwiseAddTester(
      const std::string &op_name,
      const std::vector<std::vector<int>> &input_shapes,
      const common::Target &target = common::DefaultHostTarget(),
      int repeat = 10,
      float diff = 1e-5)
      : OpBenchmarkTester(op_name, input_shapes, target, repeat, diff) {}

  template <typename T>
  void Compare() {
    auto all_args = GetAllArgs();
    std::vector<T *> all_datas;
    for (auto &arg : all_args) {
      auto *buffer = cinn_pod_value_to_buffer_p(&arg);
      all_datas.push_back(reinterpret_cast<T *>(buffer->memory));
    }

    int out_dims = GetOutDims();
    CHECK_EQ(all_datas.size(), 3U) << "elementwise_add should have 3 args.\n";
    for (int i = 0; i < out_dims; ++i) {
      EXPECT_EQ(all_datas[0][i] + all_datas[1][i], all_datas[2][i]);
    }
  }
};

}  // namespace tests
}  // namespace cinn
