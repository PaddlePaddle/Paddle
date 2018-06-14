/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/maxouting.h"

namespace paddle {
namespace operators {
namespace math {

// All tensors are in NCHW format, and the groups must be greater than 1
template <typename T>
class MaxOutFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* output,
                  int groups) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    int fea_size = input_height * input_width;
    // c_size means the output size of each sample
    int c_size = fea_size * output_channels;
    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; ++i) {
      int new_bindex = c_size * i;
      for (int c = 0; c < output_channels; ++c) {
        int new_cindex = fea_size * c;
        for (int f = 0; f < fea_size; ++f) {
          T ele = static_cast<T>(-FLT_MAX);
          for (int ph = 0; ph < groups; ++ph) {
            T x = input_data[(new_bindex + new_cindex) * groups +
                             ph * fea_size + f];
            ele = ele > x ? ele : x;
          }
          output_data[(new_bindex + new_cindex + f)] = ele;
        }
      }
    }
  }
};

template <class T>
class MaxOutGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, int groups) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    int fea_size = input_height * input_width;
    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; ++i) {
      int blen = fea_size * output_channels * i;
      for (int c = 0; c < output_channels; ++c) {
        int clen = fea_size * c;
        for (int f = 0; f < fea_size; ++f) {
          int input_idx0 = (blen + clen) * groups + f;
          bool continue_match = true;
          int output_idx = blen + clen + f;
          for (int g = 0; g < groups && continue_match; ++g) {
            int input_idx = input_idx0 + fea_size * g;
            if (input_data[input_idx] == output_data[output_idx]) {
              input_grad_data[input_idx] += output_grad_data[output_idx];
              continue_match = false;
            }
          }
        }
      }
    }
  }
};

template class MaxOutGradFunctor<platform::CPUDeviceContext, float>;
template class MaxOutGradFunctor<platform::CPUDeviceContext, double>;
template class MaxOutFunctor<platform::CPUDeviceContext, float>;
template class MaxOutFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
