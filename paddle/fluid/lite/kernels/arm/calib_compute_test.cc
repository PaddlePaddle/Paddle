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

#include "paddle/fluid/lite/kernels/arm/calib_compute.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

static int get_rand(int start, int end) {
  int i = rand();  // NOLINT
  i = (i % (end - start)) + start;
  return i;
}

static void int8_to_fp32_basic(const int8_t* din, float* dout,
                               const float* scale, int axis_size,
                               int64_t outer_size, int64_t inner_size) {
  int loop_size = axis_size * outer_size;
  for (int i = 0; i < loop_size; ++i) {
    float scale_in = scale[i % axis_size];
    for (int j = 0; j < inner_size; ++j) {
      dout[j] = din[j] * scale_in;
    }
    dout += inner_size;
    din += inner_size;
  }
}

static void fp32_to_int8_basic(const float* din, int8_t* dout,
                               const float* scale, int axis_size,
                               int64_t outer_size, int64_t inner_size) {
  int loop_size = axis_size * outer_size;
  for (int i = 0; i < loop_size; ++i) {
    float inv_scale = 1.f / scale[i % axis_size];
    for (int j = 0; j < inner_size; ++j) {
      dout[j] = static_cast<int8_t>(roundf(din[j] * inv_scale));
    }
    dout += inner_size;
    din += inner_size;
  }
}

void calib_ref(const operators::CalibParam& param) {
  std::vector<float> scale = {param.in_scale};
  if (param.in_dtype == PRECISION(kFloat) &&
      param.out_dtype == PRECISION(kInt8)) {
    const auto* din = param.input->data<float>();
    auto* dout = param.output->mutable_data<signed char>();
    fp32_to_int8_basic(din, dout, scale.data(), 1, 1, param.input->numel());
    return;
  }
  if (param.in_dtype == PRECISION(kInt8) &&
      param.out_dtype == PRECISION(kFloat)) {
    const auto* din = param.input->data<signed char>();
    auto* dout = param.output->mutable_data<float>();
    int8_to_fp32_basic(din, dout, scale.data(), 1, 1, param.input->numel());
    return;
  }
  LOG(FATAL) << "Unsupport Dtype.";
}

TEST(calib_arm, retrive_op) {
  auto calib =
      KernelRegistry::Global()
          .Create<TARGET(kARM), PRECISION(kInt8), DATALAYOUT(kNCHW)>("calib");
  ASSERT_FALSE(calib.empty());
  ASSERT_TRUE(calib.front());
}

TEST(calib_arm, init) {
  CalibCompute calib;
  ASSERT_EQ(calib.precision(), PRECISION(kInt8));
  ASSERT_EQ(calib.target(), TARGET(kARM));
}

TEST(calib_arm, int8_to_fp32) {
  DeviceInfo::Init();
  for (auto n : {1, 2}) {
    for (auto c : {6, 32 /*, 128*/}) {
      for (auto h : {9, 18 /*, 56 , 112, 224, 512*/}) {
        for (auto w : {9, 18 /*, 56, 112, 224, 512*/}) {
          Tensor x;
          Tensor output;
          Tensor output_ref;
          // set the dims of input, output, ref output tensors
          x.Resize({n, c, h, w});
          output.Resize({n, c, h, w});
          output_ref.Resize({n, c, h, w});
          // initialize the data of input tensors
          auto* x_data = x.mutable_data<char>();
          auto* output_data = output.mutable_data<float>();
          for (int i = 0; i < x.dims().production(); i++) {
            float sign = i % 3 == 0 ? -1.0f : 1.0f;
            x_data[i] = sign * static_cast<float>(i % 128) * 0.013f;
          }
          // prepare kernel params and run
          CalibCompute calib;
          std::unique_ptr<KernelContext> ctx(new KernelContext);
          ctx->As<ARMContext>();
          calib.SetContext(std::move(ctx));
          operators::CalibParam param;
          param.in_scale = get_rand(0, 100) * 0.1f;
          param.in_dtype = PRECISION(kInt8);
          param.out_dtype = PRECISION(kFloat);
          param.input = &x;
          param.output = &output;
          calib.SetParam(param);
          calib.Launch();
          // invoking ref implementation and compare results
          param.output = &output_ref;
          calib_ref(param);
          auto* output_ref_data = output_ref.mutable_data<float>();
          for (int i = 0; i < output.dims().production(); i++) {
            EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
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

USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, fp32_to_int8);
