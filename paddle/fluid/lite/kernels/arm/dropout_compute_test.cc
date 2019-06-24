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

#include "paddle/fluid/lite/kernels/arm/dropout_compute.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(dropout_arm, init) {
  DropoutCompute dropout;
  ASSERT_EQ(dropout.precision(), PRECISION(kFloat));
  ASSERT_EQ(dropout.target(), TARGET(kARM));
}

TEST(dropout, retrive_op) {
  auto dropout =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "dropout");
  ASSERT_FALSE(dropout.empty());
  ASSERT_TRUE(dropout.front());
}

template <typename dtype>
void dropout_compute_ref(const operators::DropoutParam& param) {
  const float* x_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>();
  int num = param.x->dims().production();
  const float prob_data = param.dropout_prob;
  if (param.dropout_implementation.compare(
          std::string({"downgrade_in_infer"})) == 0) {
    float scale = 1.0 - prob_data;
    for (int i = 0; i < num; i++) {
      output_data[i] = x_data[i] * scale;
    }
  } else {
    for (int i = 0; i < num; i++) {
      output_data[i] = x_data[i];
    }
  }
}

TEST(dropout_arm, compute) {
  DropoutCompute dropout;
  operators::DropoutParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_ref;

  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto prob : {0.2f, 0.8f})
            for (auto impl : {std::string({"downgrade_in_infer"}),
                              std::string({"upscale_in_train"})}) {
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
              param.dropout_prob = prob;
              param.dropout_implementation = impl;
              dropout.SetParam(param);
              dropout.Run();
              param.output = &output_ref;
              dropout_compute_ref<float>(param);
              for (int i = 0; i < output.dims().production(); i++) {
                EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
              }
            }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(dropout, kARM, kFloat, kNCHW, def);
