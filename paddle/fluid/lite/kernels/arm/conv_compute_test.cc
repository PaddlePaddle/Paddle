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

#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/kernels/arm/fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(conv_arm, retrive_op) {
  auto conv =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("conv");
  ASSERT_FALSE(conv.empty());
  ASSERT_TRUE(conv.front());
}

TEST(conv_arm, init) {
  FcCompute conv;
  ASSERT_EQ(conv.precision(), PRECISION(kFloat));
  ASSERT_EQ(conv.target(), TARGET(kARM));
}

TEST(conv_arm, compare_test) {
  lite::Tensor x, w, b, out, ref;
  constexpr int batch_size = 2;
  x.Resize({batch_size, 3});
  w.Resize({3, 4});
  b.Resize({1, 4});
  out.Resize({batch_size, 4});
  ref.Resize({batch_size, 4});

  auto x_data = x.mutable_data<float>();
  auto w_data = w.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto ref_data = ref.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().product(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().product(); i++) {
    b_data[i] = static_cast<float>(i);
  }

  lite::arm::math::fc_compute_eigen(x_data, batch_size, 3,  //
                                    w_data, 3, 4,           //
                                    b_data, ref_data);

  // conv compute kernel
  FcCompute conv;
  operators::FcParam param;

  param.in_num_col_dims = 1;
  param.input = &x;
  param.w = &w;
  param.bias = &b;
  param.output = &out;
  param.in_mat_dims = x.dims();

  DeviceInfo::Init();
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  conv.SetParam(param);
  conv.SetContext(std::move(ctx));
  conv.Run();

  VLOG(3) << "output vs ref";
  for (int i = 0; i < out.dims().product(); i++) {
    VLOG(3) << out_data[i] << " vs " << ref_data[i];
  }

  for (int i = 0; i < out.dims().product(); ++i) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

TEST(conv_arm, num_col_dims) {
  FcCompute conv;
  operators::FcParam param;

  lite::Tensor x;
  lite::Tensor w;
  lite::Tensor bias;
  lite::Tensor output;

  x.Resize({1, 2, 3});
  w.Resize({3, 4});
  bias.Resize({1, 4});
  output.Resize({2, 4});

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

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  DeviceInfo::Init();

  conv.SetParam(param);
  conv.SetContext(std::move(ctx));
  conv.Run();
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv, kARM, kFloat, kNCHW, def);
