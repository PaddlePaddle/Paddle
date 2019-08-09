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

#include "paddle/fluid/lite/kernels/x86/elementwise_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(elementwise_add_x86, retrive_op) {
  auto elementwise_add =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "elementwise_add");
  ASSERT_FALSE(elementwise_add.empty());
  ASSERT_TRUE(elementwise_add.front());
}

TEST(elementwise_add_x86, init) {
  ElementwiseAddCompute<float> elementwise_add;
  ASSERT_EQ(elementwise_add.precision(), PRECISION(kFloat));
  ASSERT_EQ(elementwise_add.target(), TARGET(kX86));
}

TEST(elementwise_add_x86, run_test) {
  lite::Tensor x, y, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> y_shape{batch_size, 3, 2, 2};
  y.Resize(lite::DDim(y_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto y_data = y.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = 1;
  }
  for (int64_t i = 0; i < y.dims().production(); i++) {
    y_data[i] = 2;
  }

  // ElementwiseAddCompute elementwise_add;
  ElementwiseAddCompute<float> elementwise_add;
  operators::ElementwiseParam param;

  param.X = &x;
  param.Y = &y;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  elementwise_add.SetParam(param);
  elementwise_add.SetContext(std::move(ctx));
  elementwise_add.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);
