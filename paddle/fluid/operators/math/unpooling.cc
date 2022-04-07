/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/unpooling.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
class Unpool2dMaxFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];

          PADDLE_ENFORCE_LT(
              index, output_feasize,
              platform::errors::InvalidArgument(
                  "index should less than output tensor height * output tensor "
                  "width. Expected %ld < %ld, but got "
                  "%ld >= %ld. Please check input value.",
                  index, output_feasize, index, output_feasize));
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
class Unpool2dMaxGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const int* indices_data = indices.data<int>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];
          PADDLE_ENFORCE_LT(
              index, output_feasize,
              platform::errors::InvalidArgument(
                  "index should less than output tensor height * output tensor "
                  "width. Expected %ld < %ld, but got "
                  "%ld >= %ld. Please check input value.",
                  index, output_feasize, index, output_feasize));
          input_grad_data[i] = output_grad_data[index];
        }
        input_grad_data += input_feasize;
        indices_data += input_feasize;
        output_grad_data += output_feasize;
      }
    }
  }
};

template <typename T>
class Unpool3dMaxFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    int input_feasize = input_depth * input_height * input_width;
    int output_feasize = output_depth * output_height * output_width;
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];

          PADDLE_ENFORCE_LT(
              index, output_feasize,
              platform::errors::InvalidArgument(
                  "index should less than output tensor depth * output tensor "
                  "height "
                  "* output tensor width. Expected %ld < %ld, but got "
                  "%ld >= %ld. Please check input value.",
                  index, output_feasize, index, output_feasize));
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
class Unpool3dMaxGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    int input_feasize = input_depth * input_height * input_width;
    int output_feasize = output_depth * output_height * output_width;
    const int* indices_data = indices.data<int>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];
          PADDLE_ENFORCE_LT(
              index, output_feasize,
              platform::errors::InvalidArgument(
                  "index should less than output tensor depth * output tensor "
                  "height "
                  "* output tensor width. Expected %ld < %ld, but got "
                  "%ld >= %ld. Please check input value.",
                  index, output_feasize, index, output_feasize));
          input_grad_data[i] = output_grad_data[index];
        }
        input_grad_data += input_feasize;
        indices_data += input_feasize;
        output_grad_data += output_feasize;
      }
    }
  }
};

template class Unpool2dMaxGradFunctor<platform::CPUDeviceContext, float>;
template class Unpool2dMaxGradFunctor<platform::CPUDeviceContext, double>;
template class Unpool2dMaxFunctor<platform::CPUDeviceContext, float>;
template class Unpool2dMaxFunctor<platform::CPUDeviceContext, double>;
template class Unpool3dMaxGradFunctor<platform::CPUDeviceContext, float>;
template class Unpool3dMaxGradFunctor<platform::CPUDeviceContext, double>;
template class Unpool3dMaxFunctor<platform::CPUDeviceContext, float>;
template class Unpool3dMaxFunctor<platform::CPUDeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
