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
#include "paddle/fluid/lite/kernels/x86/fc_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(fc_x86, retrive_op) {
  auto fc =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("fc");
  ASSERT_FALSE(fc.empty());
  ASSERT_TRUE(fc.front());
}

TEST(fc_x86, init) {
  FcCompute<float> fc;
  ASSERT_EQ(fc.precision(), PRECISION(kFloat));
  ASSERT_EQ(fc.target(), TARGET(kX86));
}

TEST(fc_x86, run_test) {
  lite::Tensor x, w, b, out;
  constexpr int batch_size = 2;
  std::vector<int64_t> x_shape{batch_size, 3};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> w_shape{3, 4};
  w.Resize(lite::DDim(w_shape));
  std::vector<int64_t> b_shape{1, 4};
  b.Resize(lite::DDim(b_shape));
  std::vector<int64_t> out_shape{1, 4};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto w_data = w.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().production(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().production(); i++) {
    b_data[i] = static_cast<float>(i);
  }

  /* lite::x86::math::fc_compute_eigen(x_data, batch_size, 3,  //
                                     w_data, 3, 4,           //
                                     b_data, ref_data); */

  // FcCompute fc;
  FcCompute<float> fc;
  operators::FcParam param;

  param.in_num_col_dims = 1;
  param.input = &x;
  param.w = &w;
  param.bias = &b;
  param.output = &out;
  param.in_mat_dims = x.dims();

  // std::unique_ptr<KernelContext> ctx(new KernelContext);
  // ctx->As<X86Context>();
  fc.SetParam(param);
  // fc.SetContext(std::move(ctx));
  fc.Run();

  VLOG(3) << "output vs ref";
  for (int i = 0; i < out.dims().production(); i++) {
    VLOG(3) << out_data[i];
  }

  /* for (int i = 0; i < out.dims().product(); ++i) {
     EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
   }*/
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
