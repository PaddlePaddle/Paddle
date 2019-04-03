// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/kernel.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace core {

int test_code{-1};
class SomeKernel : public OpKernel<TARGET(kHost), PRECISION(kFloat)> {
 public:
  void Run() override {
    LOG(INFO) << "SomeKernel executed";
    LOG(INFO) << param<operators::FcParam>().in_num_col_dims;
    test_code = param<operators::FcParam>().in_num_col_dims;
  }
};

TEST(Kernel, test) {
  SomeKernel kernel;
  operators::FcParam param;
  param.in_num_col_dims = 100;
  kernel.SetParam<operators::FcParam>(param);
  kernel.Run();
  ASSERT_EQ(test_code, 100);
}

}  // namespace core
}  // namespace lite
}  // namespace paddle
