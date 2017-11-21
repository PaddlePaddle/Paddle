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

#include "paddle/operators/math/maxouting.h"

namespace paddle {
namespace operators {
namespace math {

// All tensors are in NCHW format
template <typename T>
class Unpool2d_Max_Functor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  framework::Tensor * output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];

    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T* input_data = input.data<T>();
    const T* indices_data = indices.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index =  indices_data[i];
          if(index > output_feasize) {
            //抛一个异常！
          }
          output_data[index] = input_data[i];
        }
        input_data += input_feasize;
        indices_data += input_feasize;
        output_data += output_feasize;
      }
    }
  }
};



template <class T>
class Unpool2d_MaxGradFunctor<platform::CPUPlace, T> {
public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  framework::Tensor * input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];

    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T* input_data = input.data<T>();
    const T* indices_data = indices.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();

    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int f = 0; f < input_feasize; ++f) {
          int index = indices_data[i];
          if(index > output_feasize) {
            //抛一个异常！
          }
          input_grad_data[i] = output_grad_data[index];
        }
        input_grad_data += input_feasize;
        indices_data += input_feasize;
        output_grad_data += output_feasize;
      }
    }
  }
};

template class Unpool2d_MaxGradFunctor<platform::CPUPlace, float>;
template class Unpool2d_MaxGradFunctor<platform::CPUPlace, double>;
template class Unpool2d_MaxFunctor<platform::CPUPlace, float>;
template class Unpool2d_MaxFunctor<platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
