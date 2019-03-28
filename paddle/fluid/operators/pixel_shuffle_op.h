/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PixelShuffleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");

    int upscale_factor = ctx.Attr<int>("upscale_factor");

    auto input_dims = input->dims();
    auto num = input_dims[0];
    auto channels = input_dims[1];
    auto height = input_dims[2];
    auto width = input_dims[3];
    auto output_dim = channels / (upscale_factor * upscale_factor);

    int count = num * channels * height * width;
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    for (int index = 0; index < count; index++) {
      int width_index = index % width;
      int height_index = ((index - width_index) % (width * height)) / width;
      int channel_index = ((index - width_index - height_index * width) %
                           (channels * height * width)) /
                          (height * width);
      int num_index = (index - width_index - height_index * width -
                       channel_index * height * width) /
                      (channels * height * width);

      int input_data_index = index;
      int output_width_index;
      int output_height_index;
      int output_channel_index;
      output_channel_index = channel_index / (upscale_factor * upscale_factor);
      output_width_index =
          width_index * upscale_factor + channel_index % upscale_factor;
      output_height_index =
          height_index * upscale_factor +
          (channel_index -
           output_channel_index * upscale_factor * upscale_factor) /
              upscale_factor;
      int output_data_index =
          num_index *
              (output_dim * height * upscale_factor * width * upscale_factor) +
          output_channel_index *
              (height * upscale_factor * width * upscale_factor) +
          output_height_index * width * upscale_factor + output_width_index;

      output_data[output_data_index] = input_data[input_data_index];
      //      output_data[index] = input_data[index];
    }
  }
};

template <typename DeviceContext, typename T>
class PixelShuffleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");

    auto input_dims = input->dims();
    auto num = input_dims[0];
    auto channels = input_dims[1];
    auto height = input_dims[2];
    auto width = input_dims[3];
    int upscale_factor = ctx.Attr<int>("upscale_factor");
    auto output_dim = channels / (upscale_factor * upscale_factor);

    int count = num * channels * height * width;
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    const T* output_grad_data = output_grad->data<T>();

    for (int index = 0; index < count; index++) {
      int output_width = width * upscale_factor;
      int output_height = height * upscale_factor;
      int output_channels = output_dim;

      int output_width_index = index % output_width;
      int output_height_index =
          ((index - output_width_index) % (output_width * output_height)) /
          (output_width);
      int output_channel_index =
          ((index - output_width_index - output_height_index * output_width) %
           (output_channels * output_height * output_width)) /
          (output_width * output_height);
      int num_index =
          (index - output_width_index - output_height_index * output_width -
           output_channel_index * (output_height * output_width)) /
          (output_channels * output_height * output_width);

      int input_channels = channels;
      int input_height = height;
      int input_width = width;

      int input_channel_index =
          output_channel_index * upscale_factor * upscale_factor +
          (output_width_index % upscale_factor) +
          (output_height_index % upscale_factor) * upscale_factor;
      int input_width_index = output_width_index / upscale_factor;
      int input_height_index = output_height_index / upscale_factor;
      int input_grad_index =
          num_index * (input_channels * input_height * input_width) +
          input_channel_index * (input_height * input_width) +
          input_height_index * input_width + input_width_index;

      input_grad_data[input_grad_index] =
          (index < num * output_channels * output_height * output_width)
              ? output_grad_data[index]
              : 0;
      //      input_grad_data[index] = output_grad_data[index];
    }
  }
};

}  // namespace operators
}  // namespace paddle
