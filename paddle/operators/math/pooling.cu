/* Copyright (c) 2016 paddlepaddle Authors. All Rights Reserve.

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
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void KernelMaxPool2dWithIdx(
    const int nthreads, const T* input_data, T* output_data, T* mask_data,
    const int channels, const int input_height, const int input_width,
    const int output_height, const int output_width, const int ksize_height,
    const int ksize_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % channels;
    int batch_idx = index / output_width / output_height / channels;

    int hstart = ph * stride_height - padding_height;
    int hend = min(hstart + ksize_height, input_height);
    hstart = max(hstart, 0);

    int wstart = pw * stride_width - padding_width;
    int wend = min(wstart + ksize_width, input_width);
    wstart = max(wstart, 0);

    input_data += (batch_idx * channels + c) * input_height * input_width;
    T ele = -FLT_MAX;
    int max_index = -1;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * input_width + w;
        if (ele < input_data[input_index]) {
          max_index = input_index;
          ele = input_data[input_index];
        }
      }
    }
    output_data[index] = ele;
    mask_data[index] = max_index;
  }
}

template <typename T>
__global__ void KernelMaxPool2DWithIdxGrad(
    const int nthreads, T* input_grad, const T* output_grad, const T* mask_data,
    const int channels, const int input_height, const int input_width,
    const int output_height, const int output_width, const int ksize_height,
    const int ksize_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    int w_offset = index % input_width;
    int h_offset = (index / input_width) % input_height;
    int c_offset = (index / input_width / input_height) % channels;
    int batch_idx = index / input_width / input_height / channels;

    int ph_start =
        (h_offset + padding_height < ksize_height)
            ? 0
            : (h_offset + padding_height - ksize_height) / stride_height + 1;
    int pw_start =
        (w_offset + padding_width < ksize_width)
            ? 0
            : (w_offset + padding_width - ksize_width) / stride_width + 1;
    int ph_end =
        min((h_offset + padding_height) / stride_height + 1, output_height);
    int pw_end =
        min((w_offset + padding_width) / stride_width + 1, output_width);

    T gradient = 0;
    int input_current_featuremap_idx = h_offset * input_width + w_offset;
    int output_idx =
        (batch_idx * channels + c_offset) * output_height * output_width;

    mask_data += output_idx;
    output_grad += output_idx;
    for (int ph = ph_start; ph < ph_end; ++ph) {
      for (int pw = pw_start; pw < pw_end; ++pw) {
        if (mask_data[ph * output_width + pw] == input_current_featuremap_idx)
          gradient += output_grad[ph * output_width + pw];
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
class MaxPool2dWithIndexFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
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

    const T* input_data = input.data<T>();
    T* output_data = output.mutable_data<T>(context.GetPlace());
    T* mask_data = mask.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2dWithIdx<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(nthreads, input_data, output_data, mask_data,
                              input_channels, input_height, input_width,
                              output_height, output_width, ksize_height,
                              ksize_width, stride_height, stride_width,
                              padding_height, padding_width);
  }
};

template <typename T>
class MaxPool2dWithIndexGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input_grad.dims()[0];
    const int input_channels = input_grad.dims()[1];
    const int input_height = input_grad.dims()[2];
    const int input_width = input_grad.dims()[3];
    const int output_channels = output_grad.dims()[1];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* mask_data = mask.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2DWithIdxGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(nthreads, input_grad_data, output_grad_data,
                              mask_data, input_channels, input_height,
                              input_width, output_height, output_width,
                              ksize_height, ksize_width, stride_height,
                              stride_width, padding_height, padding_width);
  }
};

template class MaxPool2dWithIndexFunctor<platform::GPUPlace, float>;
template class MaxPool2dWithIndexGradFunctor<platform::GPUPlace, float>;
template class MaxPool2dWithIndexFunctor<platform::GPUPlace, double>;
template class MaxPool2dWithIndexGradFunctor<platform::GPUPlace, double>;

template <typename T>
__global__ void KernelMaxPool3DWithIdx(
    const int nthreads, const T* input_data, T* output_data, T* mask_data,
    const int channels, const int input_depth, const int input_height,
    const int input_width, const int output_depth, const int output_height,
    const int output_width, const int ksize_depth, const int ksize_height,
    const int ksize_width, const int stride_depth, const int stride_height,
    const int stride_width, const int padding_depth, const int padding_height,
    const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int pd = (index / output_width / output_height) % output_depth;
    int c = (index / output_width / output_height / output_depth) % channels;
    int batch_idx =
        index / output_width / output_height / output_depth / channels;

    int dstart = pd * stride_depth - padding_depth;
    int hstart = ph * stride_height - padding_height;
    int wstart = pw * stride_width - padding_width;
    int dend = min(dstart + ksize_depth, input_depth);
    int hend = min(hstart + ksize_height, input_height);
    int wend = min(wstart + ksize_width, input_width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    T ele = -FLT_MAX;
    int max_index = -1;
    input_data +=
        (batch_idx * channels + c) * input_depth * input_height * input_width;

    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (ele < input_data[(d * input_height + h) * input_width + w]) {
            max_index = (d * input_height + h) * input_width + w;
            ele = input_data[max_index];
          }
        }
      }
    }
    output_data[index] = ele;
    mask_data[index] = max_index;
  }
}

template <typename T>
__global__ void KernelMaxPool3DWithIdxGrad(
    const int nthreads, T* input_grad, const T* output_grad, const T* mask,
    const int channels, const int input_depth, const int input_height,
    const int input_width, const int output_depth, const int output_height,
    const int output_width, const int ksize_depth, const int ksize_height,
    const int ksize_width, const int stride_depth, const int stride_height,
    const int stride_width, const int padding_depth, const int padding_height,
    const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    int w_offset = index % input_width;
    int h_offset = (index / input_width) % input_height;
    int d_offset = (index / input_width / input_height) % input_depth;
    int c_offset =
        (index / input_width / input_height / input_depth) % channels;
    int batch_idx = index / input_width / input_height / input_depth / channels;

    int pd_start =
        (d_offset + padding_depth < ksize_depth)
            ? 0
            : (d_offset + padding_depth - ksize_depth) / stride_depth + 1;
    int ph_start =
        (h_offset + padding_height < ksize_height)
            ? 0
            : (h_offset + padding_height - ksize_height) / stride_height + 1;
    int pw_start =
        (w_offset + padding_width < ksize_width)
            ? 0
            : (w_offset + padding_width - ksize_width) / stride_width + 1;
    int pd_end =
        min((d_offset + padding_depth) / stride_depth + 1, output_depth);
    int ph_end =
        min((h_offset + padding_height) / stride_height + 1, output_height);
    int pw_end =
        min((w_offset + padding_width) / stride_width + 1, output_width);

    T gradient = 0;
    int input_current_feature_map_idx =
        (d_offset * input_height + h_offset) * input_width + w_offset;
    int output_idx = (batch_idx * channels + c_offset) * output_depth *
                     output_height * output_width;
    mask += output_idx;
    output_grad += output_idx;

    for (int pd = pd_start; pd < pd_end; ++pd) {
      for (int ph = ph_start; ph < ph_end; ++ph) {
        for (int pw = pw_start; pw < pw_end; ++pw) {
          if (mask[(pd * output_height + ph) * output_width + pw] ==
              input_current_feature_map_idx)
            gradient +=
                output_grad[(pd * output_height + ph) * output_width + pw];
        }
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
class MaxPool3dWithIndexFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
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

    const T* input_data = input.data<T>();
    T* output_data = output.mutable_data<T>(context.GetPlace());
    T* mask_data = mask.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DWithIdx<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, mask_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width);
  }
};

template <typename T>
class MaxPool3dWithIndexGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings) {
    const int batch_size = input_grad.dims()[0];
    const int input_channels = input_grad.dims()[1];
    const int input_depth = input_grad.dims()[2];
    const int input_height = input_grad.dims()[3];
    const int input_width = input_grad.dims()[4];
    const int output_channels = output_grad.dims()[1];
    const int output_depth = output_grad.dims()[2];
    const int output_height = output_grad.dims()[3];
    const int output_width = output_grad.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* output_grad_data = output_grad.data<T>();
    const T* mask_data = mask.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DWithIdxGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_grad_data, output_grad_data, mask_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width);
  }
};

template class MaxPool3dWithIndexFunctor<platform::GPUPlace, float>;
template class MaxPool3dWithIndexGradFunctor<platform::GPUPlace, float>;
template class MaxPool3dWithIndexFunctor<platform::GPUPlace, double>;
template class MaxPool3dWithIndexGradFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
