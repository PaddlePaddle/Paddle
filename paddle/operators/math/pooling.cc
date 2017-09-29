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

template <typename T>
class MaxPool2dWithIndexFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
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
    T* output_data = output.mutable_data<T>(context.GetPlace());

    T* mask_data = mask.mutable_data<T>(context.GetPlace());

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

            T ele = static_cast<T>(-FLT_MAX);
            int index = -1;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                if (ele < input_data[h * input_width + w]) {
                  ele = input_data[h * input_width + w];
                  index = h * input_width + w;
                }
              }
            }
            output_data[ph * output_width + pw] = ele;
            mask_data[ph * output_width + pw] = index;
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template <typename T>
class MaxPool2dWithIndexGradFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input_grad.dims()[0];
    const int input_height = input_grad.dims()[2];
    const int input_width = input_grad.dims()[3];
    const int output_channels = output_grad.dims()[1];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* mask_data = mask.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < output_channels; ++c) {
        for (size_t ph = 0; ph < output_height; ++ph) {
          for (size_t pw = 0; pw < output_width; ++pw) {
            const size_t output_idx = ph * output_width + pw;
            const size_t input_idx = static_cast<size_t>(mask_data[output_idx]);

            input_grad_data[input_idx] += output_grad_data[output_idx];
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool2dWithIndexFunctor<platform::CPUPlace, float>;
template class MaxPool2dWithIndexGradFunctor<platform::CPUPlace, float>;
template class MaxPool2dWithIndexFunctor<platform::CPUPlace, double>;
template class MaxPool2dWithIndexGradFunctor<platform::CPUPlace, double>;

template <typename T>
class MaxPool3dWithIndexFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
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
    T* output_data = output.mutable_data<T>(context.GetPlace());
    T* mask_data = mask.mutable_data<T>(context.GetPlace());

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
              T ele = static_cast<T>(-FLT_MAX);
              int index = -1;
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    if (ele < input_data[input_idx]) {
                      index = input_idx;
                      ele = input_data[input_idx];
                    }
                  }
                }
              }
              output_data[output_idx] = ele;
              mask_data[output_idx] = index;
            }
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template <typename T>
class MaxPool3dWithIndexGradFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input_grad.dims()[0];
    const int input_depth = input_grad.dims()[2];
    const int input_height = input_grad.dims()[3];
    const int input_width = input_grad.dims()[4];
    const int output_channels = output_grad.dims()[1];
    const int output_depth = output_grad.dims()[2];
    const int output_height = output_grad.dims()[3];
    const int output_width = output_grad.dims()[4];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* mask_data = mask.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < output_channels; ++c) {
        for (size_t pd = 0; pd < output_depth; ++pd) {
          for (size_t ph = 0; ph < output_height; ++ph) {
            for (size_t pw = 0; pw < output_width; ++pw) {
              const size_t output_idx =
                  (pd * output_height + ph) * output_width + pw;
              const size_t input_idx =
                  static_cast<size_t>(mask_data[output_idx]);

              input_grad_data[input_idx] += output_grad_data[output_idx];
            }
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool3dWithIndexFunctor<platform::CPUPlace, float>;
template class MaxPool3dWithIndexGradFunctor<platform::CPUPlace, float>;
template class MaxPool3dWithIndexFunctor<platform::CPUPlace, double>;
template class MaxPool3dWithIndexGradFunctor<platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
