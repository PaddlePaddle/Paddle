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

#include "paddle/fluid/lite/kernels/x86/scale_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(scale_x86, retrive_op) {
  auto scale =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("scale");
  ASSERT_FALSE(scale.empty());
  ASSERT_TRUE(scale.front());
}

TEST(scale_x86, init) {
  ScaleCompute<float> scale;
  ASSERT_EQ(scale.precision(), PRECISION(kFloat));
  ASSERT_EQ(scale.target(), TARGET(kX86));
}

TEST(scale_x86, run_test) {
  lite::Tensor x, y, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  // ScaleCompute scale;
  ScaleCompute<float> scale;
  operators::ScaleParam param;

  param.x = &x;
  param.scale = 0.5;
  param.bias = 0;
  param.output = &out;

  scale.SetParam(param);
  scale.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
