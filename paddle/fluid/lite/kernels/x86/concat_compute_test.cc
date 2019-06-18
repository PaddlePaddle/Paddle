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

#include "paddle/fluid/lite/kernels/x86/concat_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(concat_x86, retrive_op) {
  auto concat =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "concat");
  ASSERT_FALSE(concat.empty());
  ASSERT_TRUE(concat.front());
}

TEST(concat_x86, init) {
  ConcatCompute<float> concat;
  ASSERT_EQ(concat.precision(), PRECISION(kFloat));
  ASSERT_EQ(concat.target(), TARGET(kX86));
}

TEST(concat_x86, run_test) {
  lite::Tensor x1, x2, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x1_shape{batch_size, 1, 3, 3};
  x1.Resize(lite::DDim(x1_shape));
  std::vector<int64_t> x2_shape{batch_size, 1, 3, 3};
  x2.Resize(lite::DDim(x2_shape));

  std::vector<lite::Tensor*> x = {&x1, &x2};

  std::vector<int64_t> out_shape{batch_size, 2, 3, 3};
  out.Resize(lite::DDim(out_shape));

  auto x1_data = x1.mutable_data<float>();
  auto x2_data = x2.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x1.dims().production(); i++) {
    x1_data[i] = 1;
    x2_data[i] = 2;
  }

  ConcatCompute<float> concat;
  operators::ConcatParam param;
  param.x = x;
  param.output = &out;
  param.axis = 1;

  concat.SetParam(param);
  concat.Run();

  std::cout << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, def);
