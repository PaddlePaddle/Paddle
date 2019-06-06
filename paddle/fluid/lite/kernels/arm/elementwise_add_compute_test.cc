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

#include "paddle/fluid/lite/kernels/arm/elementwise_add_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(elementwise_add_arm, retrive_op) {
  auto elementwise_add =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "elementwise_add");
  ASSERT_FALSE(elementwise_add.empty());
  ASSERT_TRUE(elementwise_add.front());
}

TEST(elementwise_add_arm, init) {
  ElementwiseAddCompute elementwise_add;
  ASSERT_EQ(elementwise_add.precision(), PRECISION(kFloat));
  ASSERT_EQ(elementwise_add.target(), TARGET(kARM));
}

template <typename dtype>
void elementwise_add_compute_ref(const operators::ElementwiseParam& param) {
  const dtype* x_data = param.X->data<const dtype>();
  const dtype* y_data = param.Y->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  DDim dim = param.X->dims();
  ASSERT_EQ(dim.data(), param.Out->dims().data());
  for (int i = 0; i < dim.production(); i++) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

TEST(elementwise_add, compute) {
  ElementwiseAddCompute elementwise_add;
  operators::ElementwiseParam param;

  lite::Tensor x, y, out, out_ref;
  x.Resize(DDim(std::vector<int64_t>({2, 3, 4, 5})));
  y.Resize(DDim(std::vector<int64_t>({2, 3, 4, 5})));
  out.Resize(DDim(std::vector<int64_t>({2, 3, 4, 5})));
  out_ref.Resize(DDim(std::vector<int64_t>({2, 3, 4, 5})));
  auto* x_data = x.mutable_data<float>();
  auto* y_data = y.mutable_data<float>();
  auto* out_data = out.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();
  for (int i = 0; i < x.dims().production(); i++) {
    x_data[i] = y_data[i] = i;
  }

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  elementwise_add.SetParam(param);
  elementwise_add.Run();

  param.Out = &out_ref;
  elementwise_add_compute_ref<float>(param);
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
