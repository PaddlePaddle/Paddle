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

#include "paddle/fluid/lite/kernels/arm/fc_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(fc_compute_naive, test) {
  lite::Tensor x, w, b, out, out1;
  const int batch_size = 2;
  x.Resize({batch_size, 3});
  w.Resize({4, 3});
  b.Resize({1, 4});
  out.Resize({batch_size, 4});
  out1.Resize({batch_size, 4});

  auto x_data = x.mutable_data<float>();
  auto w_data = w.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto out_data1 = out1.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().product(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().product(); i++) {
    b_data[i] = static_cast<float>(i);
  }

  fc_compute_naive(x_data, 3, batch_size,  //
                   w_data, 3, 4,           //
                   b_data, out_data);
  fc_compute_eigen(x_data, 3, batch_size,  //
                   w_data, 3, 4,           //
                   b_data, out_data1);

  for (int i = 0; i < out.dims().product(); i++) {
    EXPECT_NEAR(out_data[0], out_data1[0], 1e-6);
  }
}

TEST(fc_arm, init) {
  FcCompute fc;
  ASSERT_EQ(fc.precision(), PRECISION(kFloat));
  ASSERT_EQ(fc.target(), TARGET(kARM));
}

TEST(fc_arm, algorithm) {
  using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
  using matrix_map_t = Eigen::Map<matrix_t>;

  // dim 10, 20
  std::vector<float> input(10 * 20);
  std::vector<float> w(20 * 20);
  std::vector<float> output(10 * 20);

  Eigen::Map<const matrix_t> input_mat(input.data(), 10, 20);
  Eigen::Map<const matrix_t> weight_mat(w.data(), 20, 20);
  matrix_map_t output_mat(output.data(), 10, 20);

  output_mat = input_mat * weight_mat.transpose();
}

TEST(fc_arm, compute) {
  FcCompute fc;
  operators::FcParam param;

  lite::Tensor x;
  lite::Tensor w;
  lite::Tensor bias;
  lite::Tensor output;

  x.Resize({1, 10, 20});
  w.Resize({20, 20});
  bias.Resize({1, 10});
  output.Resize({10, 20});

  auto* x_data = x.mutable_data<float>();
  auto* w_data = w.mutable_data<float>();
  auto* bias_data = bias.mutable_data<float>();
  auto* output_data = output.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().product(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < bias.dims().product(); i++) {
    bias_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < output.dims().product(); i++) {
    output_data[i] = static_cast<float>(i);
  }

  param.in_num_col_dims = 2;
  param.input = &x;
  param.w = &w;
  param.bias = &bias;
  param.output = &output;
  param.in_mat_dims = x.dims();

  fc.SetParam(param);
  fc.Run();

  VLOG(3) << "x";
  for (int i = 0; i < 10 * 20; i++) {
    VLOG(3) << x_data[i];
  }

  VLOG(3) << "output:";
  for (int i = 0; i < 10 * 20; i++) {
    VLOG(3) << output.data<float>()[i];
  }
}

TEST(fc, retrive_op) {
  auto fc =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("fc");
  ASSERT_FALSE(fc.empty());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kARM, kFloat, kNCHW, def);
