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
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int axis = param.axis;
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const dtype* din_ptr = x_data + offset;
      const dtype diny_data = y_data[j];
      dtype* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        dout_ptr[k] = din_ptr[k] + diny_data;
      }
    }
  }
}

TEST(elementwise_add, compute) {
  ElementwiseAddCompute elementwise_add;
  operators::ElementwiseParam param;
  lite::Tensor x, y, output, output_ref;

  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 4, 11}) {
      for (auto h : {1, 3, 4, 11}) {
        for (auto w : {1, 3, 4, 11}) {
          for (auto axis : {-1, 0, 1, 2, 3}) {
            for (auto yd{{n},
                         {c},
                         {h},
                         {w},
                         {n, c},
                         {c, h},
                         {h, w},
                         {n, c, h},
                         {c, h, w},
                         {n, c, h, w}}) {
              auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
              auto y_dim = DDim(std::vector<int64_t>(yd));
              int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

              if (axis_t + y_dim.size() > 4) continue;
              bool flag = false;
              for (int i = 0; i < y_dim.size(); i++) {
                if (x_dim[i + axis_t] != y_dim[i]) flag = true;
              }
              if (flag) continue;

              x.Resize(x_dim);
              y.Resize(y_dim);
              output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
              output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
              auto* x_data = x.mutable_data<float>();
              auto* output_data = output.mutable_data<float>();
              auto* output_ref_data = output_ref.mutable_data<float>();
              for (int i = 0; i < x.dims().production(); i++) {
                x_data[i] = i;
              }
              for (int i = 0; i < y.dims().production(); i++) {
                y_data[i] = i;
              }
              param.X = &x;
              param.Y = &y;
              param.axis = axis;
              param.Out = &output;
              softmax.SetParam(param);
              softmax.Run();
              param.Out = &output_ref;
              elementwise_add_compute_ref<float>(param);
              for (int i = 0; i < out.dims().production(); i++) {
                EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
              }
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

USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
