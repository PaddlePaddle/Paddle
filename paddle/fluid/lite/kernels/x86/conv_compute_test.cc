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

#include "paddle/fluid/lite/kernels/x86/conv_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(conv_x86, retrive_op) {
  auto conv2d =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "conv2d");
  ASSERT_FALSE(conv2d.empty());
  ASSERT_TRUE(conv2d.front());
}

TEST(conv2d_x86, init) {
  Conv2dCompute<float> conv2d;
  ASSERT_EQ(conv2d.precision(), PRECISION(kFloat));
  ASSERT_EQ(conv2d.target(), TARGET(kX86));
}

TEST(conv2d_x86, run_test) {
  lite::Tensor x, filter, b, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 3, 3};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> filter_shape{1, 3, 3, 3};
  filter.Resize(lite::DDim(filter_shape));
  std::vector<int64_t> b_shape{1, 3, 1, 1};
  b.Resize(lite::DDim(b_shape));
  std::vector<int64_t> out_shape{batch_size, 1, 1, 1};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto filter_data = filter.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = 1;
  }
  for (int64_t i = 0; i < filter.dims().production(); i++) {
    filter_data[i] = 1;
  }
  for (int64_t i = 0; i < b.dims().production(); i++) {
    b_data[i] = 0;
  }

  Conv2dCompute<float> conv2d;
  operators::ConvParam param;

  param.x = &x;
  param.filter = &filter;
  param.bias = &b;
  param.output = &out;
  param.strides = {1, 1};
  param.paddings = {0, 0};
  param.groups = 1;
  param.dilations = {1, 1};

  conv2d.SetParam(param);
  conv2d.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i] << " ";
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, def);
