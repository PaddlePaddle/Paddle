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

#include "paddle/fluid/lite/kernels/arm/scale_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void scale_compute_ref(const operators::ScaleParam& param) {
  const dtype* x_data = param.x->mutable_data<const dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  DDim output_dims = param.output->dims();
  ASSERT_EQ(x_dims.data(), output_dims.data());
  bool bias_after_scale = param.bias_after_scale;
  float scale = param.scale;
  float bias = param.bias;
  if (!bias_after_scale) {
    bias *= scale;
  }
  for (int i = 0; i < output_dims.production(); i++) {
    output_data[i] = x_data[i] * scale + bias;
  }
}

TEST(scale_arm, init) {
  ScaleCompute scale;
  ASSERT_EQ(scale.precision(), PRECISION(kFloat));
  ASSERT_EQ(scale.target(), TARGET(kARM));
}

TEST(scale_arm, compute) {
  ScaleCompute scale;
  operators::ScaleParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_ref;

#if 1  // for ci speedup
  for (auto n : {1, 3}) {
    for (auto c : {1, 3}) {
      for (auto h : {3, 4}) {
        for (auto w : {4, 3}) {
          for (auto bias_after_scale : {true, false}) {
            for (auto s : {-1.0f, 0.13f}) {
              for (auto b : {-15.f, 0.11234f}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 11, 4}) {
      for (auto h : {3, 1, 11, 4}) {
        for (auto w : {1, 3, 4, 12}) {
          for (auto bias_after_scale : {true, false}) {
            for (auto s : {-100.25f, -1.0f, 0.13f, 3840.975f}) {
              for (auto b : {-3075.495f, -15.f, 0.11234f, 128.15f}) {
#endif

                x.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                auto* x_data = x.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_ref_data = output_ref.mutable_data<float>();
                for (int i = 0; i < x.dims().production(); i++) {
                  x_data[i] = i;
                }
                param.x = &x;
                param.output = &output;
                param.bias_after_scale = bias_after_scale;
                param.scale = s;
                param.bias = b;
                scale.SetParam(param);
                scale.Run();
                param.output = &output_ref;
                scale_compute_ref<float>(param);
                for (int i = 0; i < output.dims().production(); i++) {
                  EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(scale, retrive_op) {
  auto scale =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("scale");
  ASSERT_FALSE(scale.empty());
  ASSERT_TRUE(scale.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, def);
