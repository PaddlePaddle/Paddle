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

#include "paddle/fluid/lite/kernels/x86/pool_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(pool_x86, retrive_op) {
  auto pool2d =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "pool2d");
  ASSERT_FALSE(pool2d.empty());
  ASSERT_TRUE(pool2d.front());
}

TEST(pool2d_x86, init) {
  PoolCompute<float> pool2d;
  ASSERT_EQ(pool2d.precision(), PRECISION(kFloat));
  ASSERT_EQ(pool2d.target(), TARGET(kX86));
}

TEST(pool2d_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 4, 4};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }

  PoolCompute<float> pool2d;
  operators::PoolParam param;

  param.x = &x;
  param.output = &out;
  param.strides = {2, 2};
  param.paddings = {0, 0};
  param.ksize = {2, 2};
  param.pooling_type = "max";

  pool2d.SetParam(param);
  pool2d.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
