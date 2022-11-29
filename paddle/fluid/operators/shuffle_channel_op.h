/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ShuffleChannelOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    int group = ctx.Attr<int>("group");

    const auto& input_dims = input->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto weight = input_dims[3];

    auto feature_map_size = channel * height * weight;
    auto sp_sz = height * weight;
    int group_row = group;
    int group_column = channel / group_row;

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    for (int n = 0; n < num; ++n) {
      for (int i = 0; i < group_row; ++i) {
        for (int j = 0; j < group_column; ++j) {
          const T* p_i = input_data + n * feature_map_size +
                         (i * group_column + j) * sp_sz;
          T* p_o =
              output_data + n * feature_map_size + (j * group_row + i) * sp_sz;
          memcpy(p_o, p_i, sizeof(int) * sp_sz);
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ShuffleChannelGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    int group = ctx.Attr<int>("group");

    const auto& input_dims = input_grad->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto weight = input_dims[3];
    auto feature_map_size = channel * height * weight;
    auto sp_sz = height * weight;

    int group_row = group;
    int group_column = channel / group_row;

    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    const T* output_grad_data = output_grad->data<T>();
    for (int n = 0; n < num; ++n) {
      for (int i = 0; i < group_row; ++i) {
        for (int j = 0; j < group_column; ++j) {
          const T* p_i = output_grad_data + n * feature_map_size +
                         (i * group_column + j) * sp_sz;
          T* p_o = input_grad_data + n * feature_map_size +
                   (j * group_row + i) * sp_sz;
          memcpy(p_o, p_i, sizeof(int) * sp_sz);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
