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

#include "paddle/operators/math/pooling.h"

namespace paddle {
namespace operators {
namespace math {

template <typename PoolProcess, typename T>
class Pool2dFunctor<platform::CPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    T* output_data = output.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          int hstart = ph * stride_height - padding_height;
          int hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            int wstart = pw * stride_width - padding_width;
            int wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);

            T ele = pool_process.initial();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_process.compute(ele, input_data[h * input_width + w]);
              }
            }
            int pool_size = (hend - hstart) * (wend - wstart);
            pool_process.finalize(ele, (static_cast<T>(pool_size)));
            output_data[ph * output_width + pw] = ele;
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

template <typename PoolProcess, class T>
class Pool2dGradFunctor<platform::CPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_grad_process) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          int hstart = ph * stride_height - padding_height;
          int hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            int wstart = pw * stride_width - padding_width;
            int wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);
            int pool_size = (hend - hstart) * (wend - wstart);
            float scale = 1.0 / pool_size;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_grad_process.compute(
                    input_data[h * input_width + w],
                    output_data[ph * output_width + pw],
                    output_grad_data[ph * output_width + pw],
                    input_grad_data[h * input_width + w],
                    static_cast<T>(scale));
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

template <class T>
class MaxPool2dGradFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          int hstart = ph * stride_height - padding_height;
          int hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            int wstart = pw * stride_width - padding_width;
            int wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);

            bool stop = false;
            for (int h = hstart; h < hend && !stop; ++h) {
              for (int w = wstart; w < wend && !stop; ++w) {
                int input_idx = h * input_width + w;
                int output_idx = ph * output_width + pw;
                if (input_data[input_idx] == output_data[output_idx]) {
                  input_grad_data[input_idx] += output_grad_data[output_idx];
                  stop = true;
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

template class MaxPool2dGradFunctor<platform::CPUPlace, float>;
// template class MaxPool2dGradFunctor<platform::CPUPlace, double>;

template class Pool2dFunctor<platform::CPUPlace,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool2dFunctor<platform::CPUPlace,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool2dGradFunctor<
    platform::CPUPlace, paddle::operators::math::MaxPoolGrad<float>, float>;
template class Pool2dGradFunctor<
    platform::CPUPlace, paddle::operators::math::AvgPoolGrad<float>, float>;
template class Pool2dFunctor<platform::CPUPlace,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool2dFunctor<platform::CPUPlace,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool2dGradFunctor<
    platform::CPUPlace, paddle::operators::math::MaxPoolGrad<double>, double>;
template class Pool2dGradFunctor<
    platform::CPUPlace, paddle::operators::math::AvgPoolGrad<double>, double>;

template <typename PoolProcess, class T>
class Pool3dFunctor<platform::CPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    T* output_data = output.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          int dstart = pd * stride_depth - padding_depth;
          int dend = std::min(dstart + ksize_depth, input_depth);
          dstart = std::max(dstart, 0);
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);
              int output_idx = (pd * output_height + ph) * output_width + pw;
              T ele = pool_process.initial();
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    pool_process.compute(
                        ele,
                        input_data[(d * input_height + h) * input_width + w]);
                  }
                }
              }
              int pool_size =
                  (dend - dstart) * (hend - hstart) * (wend - wstart);
              pool_process.finalize(ele, static_cast<T>(pool_size));
              output_data[output_idx] = ele;
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

template <typename PoolProcess, class T>
class Pool3dGradFunctor<platform::CPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_grad_process) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          int dstart = pd * stride_depth - padding_depth;
          int dend = std::min(dstart + ksize_depth, input_depth);
          dstart = std::max(dstart, 0);
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);

            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);

              int pool_size =
                  (dend - dstart) * (hend - hstart) * (wend - wstart);
              float scale = 1.0 / pool_size;
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    int output_idx =
                        (pd * output_height + ph) * output_width + pw;
                    pool_grad_process.compute(
                        input_data[input_idx], output_data[output_idx],
                        output_grad_data[output_idx],
                        input_grad_data[input_idx], static_cast<T>(scale));
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

template <class T>
class MaxPool3dGradFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          int dstart = pd * stride_depth - padding_depth;
          int dend = std::min(dstart + ksize_depth, input_depth);
          dstart = std::max(dstart, 0);
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);
              bool stop = false;
              for (int d = dstart; d < dend && !stop; ++d) {
                for (int h = hstart; h < hend && !stop; ++h) {
                  for (int w = wstart; w < wend && !stop; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    int output_idx =
                        (pd * output_height + ph) * output_width + pw;

                    if (input_data[input_idx] == output_data[output_idx]) {
                      input_grad_data[input_idx] +=
                          output_grad_data[output_idx];
                      stop = true;
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

template class MaxPool3dGradFunctor<platform::CPUPlace, float>;
// template class MaxPool3dGradFunctor<platform::CPUPlace, double>;

template class Pool3dFunctor<platform::CPUPlace,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool3dFunctor<platform::CPUPlace,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool3dGradFunctor<
    platform::CPUPlace, paddle::operators::math::MaxPoolGrad<float>, float>;
template class Pool3dGradFunctor<
    platform::CPUPlace, paddle::operators::math::AvgPoolGrad<float>, float>;
template class Pool3dFunctor<platform::CPUPlace,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool3dFunctor<platform::CPUPlace,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool3dGradFunctor<
    platform::CPUPlace, paddle::operators::math::MaxPoolGrad<double>, double>;
template class Pool3dGradFunctor<
    platform::CPUPlace, paddle::operators::math::AvgPoolGrad<double>, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
