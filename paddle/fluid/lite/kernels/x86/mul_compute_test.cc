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

#include "paddle/fluid/lite/kernels/x86/mul_compute.h"
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

TEST(mul_x86, retrive_op) {
  auto mul =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("mul");
  ASSERT_FALSE(mul.empty());
  ASSERT_TRUE(mul.front());
}

TEST(mul_x86, init) {
  MulCompute<float> mul;
  ASSERT_EQ(mul.precision(), PRECISION(kFloat));
  ASSERT_EQ(mul.target(), TARGET(kX86));
}

TEST(mul_x86, run_test) {
  lite::Tensor x, y, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> y_shape{3, 4};
  y.Resize(lite::DDim(y_shape));
  std::vector<int64_t> out_shape{batch_size, 4};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto y_data = y.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < y.dims().production(); i++) {
    y_data[i] = static_cast<float>(i);
  }
  // MulCompute mul;
  MulCompute<float> mul;
  operators::MulParam param;

  param.x = &x;
  param.y = &y;
  param.output = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  mul.SetContext(std::move(ctx));
  mul.SetParam(param);
  mul.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
