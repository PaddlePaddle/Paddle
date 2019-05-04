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
    LOG(INFO) << Param<operators::FcParam>().in_num_col_dims;
    test_code = Param<operators::FcParam>().in_num_col_dims;
  }

  TargetType target() const override { return TARGET(kHost); }
  PrecisionType precision() const override { return PRECISION(kFloat); }
};

TEST(Kernel, test) {
  SomeKernel kernel;
  operators::FcParam param;
  param.in_num_col_dims = 100;
  kernel.SetParam<operators::FcParam>(param);
  kernel.Run();
  ASSERT_EQ(test_code, 100);
}

TEST(Kernel, kernel_type) {
  const std::string op_type = "fc";
  const std::string alias = "def";
  Place place(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
  auto kernel_type = KernelBase::SerializeKernelType(op_type, alias, place);
  LOG(INFO) << "kernel_type: " << kernel_type;
  ASSERT_EQ(kernel_type, "fc/def/1/1/1");

  std::string op_type1, alias1;
  Place place1;
  KernelBase::ParseKernelType(kernel_type, &op_type1, &alias1, &place1);
  ASSERT_EQ(op_type, op_type1);
  ASSERT_EQ(alias, alias1);
  ASSERT_EQ(place, place1);
}

}  // namespace core
}  // namespace lite
}  // namespace paddle
