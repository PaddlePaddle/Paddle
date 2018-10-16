/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class BilinearInterpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_t = ctx.Input<Tensor>("X");      // float tensor
    auto* output_t = ctx.Output<Tensor>("Out");  // float tensor
    auto out_dims = output_t->dims();
    auto* input = input_t->data<T>();
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    auto out_size_t = ctx.Input<Tensor>("OutSize");
    if (out_size_t != nullptr) {
      auto out_size_data = out_size_t->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
    auto* output = output_t->mutable_data<T>(
        {out_dims[0], out_dims[1], out_h, out_w}, ctx.GetPlace());
    int batch_size = input_t->dims()[0];
    int channels = input_t->dims()[1];
    int in_h = input_t->dims()[2];
    int in_w = input_t->dims()[3];

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = channels * in_hw;
    int out_chw = channels * out_hw;

    float ratio_h =
        (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
    float ratio_w =
        (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

    if (in_h == out_h && in_w == out_w) {
      memcpy(output, input, input_t->numel() * sizeof(T));
    } else {
      for (int k = 0; k < batch_size; ++k) {  // loop for batches
        for (int i = 0; i < out_h; ++i) {     // loop for images
          int h = ratio_h * i;
          int hid = (h < in_h - 1) ? 1 : 0;
          float h1lambda = ratio_h * i - h;
          float h2lambda = 1.f - h1lambda;

          for (int j = 0; j < out_w; ++j) {
            int w = ratio_w * j;
            int wid = (w < in_w - 1) ? 1 : 0;
            float w1lambda = ratio_w * j - w;
            float w2lambda = 1.f - w1lambda;
            // calculate four position for bilinear interpolation
            const T* in_pos = &input[k * in_chw + h * in_w + w];
            T* out_pos = &output[k * out_chw + i * out_w + j];

            for (int c = 0; c < channels; ++c) {  // loop for channels
              // bilinear interpolation
              out_pos[0] = static_cast<T>(
                  h2lambda * (w2lambda * in_pos[0] + w1lambda * in_pos[wid]) +
                  h1lambda * (w2lambda * in_pos[hid * in_w] +
                              w1lambda * in_pos[hid * in_w + wid]));
              in_pos += in_hw;
              out_pos += out_hw;
            }
          }
        }
      }
    }
  }
};

template <typename T>
class BilinearInterpGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_input_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_output_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_output = d_output_t->data<T>();
    auto* d_input = d_input_t->mutable_data<T>(ctx.GetPlace());
    auto& device_ctx =
        ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, d_input_t, static_cast<T>(0.0));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");

    auto out_size_t = ctx.Input<Tensor>("OutSize");
    if (out_size_t != nullptr) {
      auto out_size_data = out_size_t->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }

    int batch_size = d_input_t->dims()[0];
    int channels = d_input_t->dims()[1];
    int in_h = d_input_t->dims()[2];
    int in_w = d_input_t->dims()[3];

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = channels * in_hw;
    int out_chw = channels * out_hw;

    float ratio_h =
        (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
    float ratio_w =
        (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

    if (in_h == out_h && in_w == out_w) {
      memcpy(d_input, d_output, d_input_t->numel() * sizeof(T));
    } else {
      for (int k = 0; k < batch_size; ++k) {  // loop for batches
        for (int i = 0; i < out_h; ++i) {     // loop for images
          int h = ratio_h * i;
          int hid = (h < in_h - 1) ? 1 : 0;
          float h1lambda = ratio_h * i - h;
          float h2lambda = 1 - h1lambda;

          for (int j = 0; j < out_w; ++j) {
            int w = ratio_w * j;
            int wid = (w < in_w - 1) ? 1 : 0;
            float w1lambda = ratio_w * j - w;
            float w2lambda = 1 - w1lambda;
            T* in_pos = &d_input[k * in_chw + h * in_w + w];
            const T* out_pos = &d_output[k * out_chw + i * out_w + j];

            for (int c = 0; c < channels; ++c) {  // loop for channels
              in_pos[0] += static_cast<T>(h2lambda * w2lambda * out_pos[0]);
              in_pos[wid] += static_cast<T>(h2lambda * w1lambda * out_pos[0]);
              in_pos[hid * in_w] +=
                  static_cast<T>(h1lambda * w2lambda * out_pos[0]);
              in_pos[hid * in_w + wid] +=
                  static_cast<T>(h1lambda * w1lambda * out_pos[0]);
              in_pos += in_hw;
              out_pos += out_hw;
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
