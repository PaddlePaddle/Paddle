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

#include "paddle/fluid/lite/kernels/arm/softmax_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void softmax_compute_ref(const operators::SoftmaxParam& param) {
  const dtype* x_data = param.x->mutable_data<const dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  ASSERT_EQ(x_dims.data(), param.output->dims().data());
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] = exp(x_data[offset] - max_data);
      sum_data += output_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

TEST(softmax_arm, init) {
  SoftmaxCompute softmax;
  ASSERT_EQ(softmax.precision(), PRECISION(kFloat));
  ASSERT_EQ(softmax.target(), TARGET(kARM));
}

TEST(softmax_arm, compute) {
  SoftmaxCompute softmax;
  operators::SoftmaxParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_ref;
#if 1
  for (auto n : {1, 3}) {
    for (auto c : {1, 4}) {
      for (auto h : {5, 1}) {
        for (auto w : {1, 6}) {
          for (auto axis : {-2, -1, 0, 1, 2}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 11, 4}) {
      for (auto h : {3, 1, 11, 4}) {
        for (auto w : {1, 3, 4, 12}) {
          for (auto axis : {-4, -3, -2, -1, 0, 1, 2, 3}) {
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
            param.axis = axis;
            param.output = &output;
            softmax.SetParam(param);
            softmax.Run();
            param.output = &output_ref;
            softmax_compute_ref<float>(param);
            for (int i = 0; i < output.dims().production(); i++) {
              EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
            }
          }
        }
      }
    }
  }
}

TEST(softmax, retrive_op) {
  auto softmax =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "softmax");
  ASSERT_FALSE(softmax.empty());
  ASSERT_TRUE(softmax.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, def);
