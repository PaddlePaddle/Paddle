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

// CUDA kernel to compute the depthwise convolution forward pass
template <typename T>
__global__ void KernelDepthwiseConv(
    const int nthreads, const T* const input_data, const T* const filter_data,
    const int batch_size, const int output_channels, const int output_height,
    const int output_width, const int input_channels, const int input_height,
    const int input_width, const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, T* const output_data) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

  if (index < nthreads) {
    const int batch = index / output_channels / output_height / output_width;
    const int c_out = (index / output_height / output_width) % output_channels;
    const int h_out = (index / output_width) % output_height;
    const int w_out = index % output_width;

    const int c_in = c_out / filter_multiplier;
    const T* weight = filter_data + c_out * filter_height * filter_width;
    T value = 0;
    const int h_in_start = -padding_height + h_out * stride_height;
    const int w_in_start = -padding_width + w_out * stride_width;
    const int h_in_end =
        -padding_height + h_out * stride_height + filter_height - 1;
    const int w_in_end =
        -padding_width + w_out * stride_width + filter_width - 1;
    if ((h_in_start >= 0) && (h_in_end < input_height) && (w_in_start >= 0) &&
        (w_in_end < input_width)) {
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          const int h_in = -padding_height + h_out * stride_height + kh;
          const int w_in = -padding_width + w_out * stride_width + kw;
          const int offset =
              ((batch * input_channels + c_in) * input_height + h_in) *
                  input_width +
              w_in;
          value += (*weight) * input_data[offset];
          ++weight;
        }
      }
    } else {
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          const int h_in = -padding_height + h_out * stride_height + kh;
          const int w_in = -padding_width + w_out * stride_width + kw;
          if ((h_in >= 0) && (h_in < input_height) && (w_in >= 0) &&
              (w_in < input_width)) {
            const int offset =
                ((batch * input_channels + c_in) * input_height + h_in) *
                    input_width +
                w_in;
            value += (*weight) * input_data[offset];
          }
          ++weight;
        }
      }
    }
    output_data[index] = value;
  }
}
/*
// CUDA kernel to compute the depthwise convolution backprop w.r.t input.
template <typename T>
__global__ void KernelDepthwiseConvInputGrad(const int nthreads,
                                      const T* const top_diff,
                                      const T* const weight_data,
                                      const int num,
                                      const int outputChannels,
                                      const int outputHeight,
                                      const int outputWidth,
                                      const int inputChannels,
                                      const int inputHeight,
                                      const int inputWidth,
                                      const int filterMultiplier,
                                      const int filterHeight,
                                      const int filterWidth,
                                      const int strideH,
                                      const int strideW,
                                      const int paddingH,
                                      const int paddingW,
                                      T* const bottom_diff) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int batch = index / inputChannels / inputHeight / inputWidth;
    const int c_in = (index / inputHeight / inputWidth) % inputChannels;
    const int h_in = (index / inputWidth) % inputHeight;
    const int w_in = index % inputWidth;

    const int c_out_start = c_in * filterMultiplier;

    int h_out_start = (h_in - filterHeight + paddingH + strideH) / strideH;
    h_out_start = 0 > h_out_start ? 0 : h_out_start;
    int h_out_end = (h_in + paddingH) / strideH;
    h_out_end = outputHeight - 1 < h_out_end ? outputHeight - 1 : h_out_end;
    int w_out_start = (w_in - filterWidth + paddingW + strideW) / strideW;
    w_out_start = 0 > w_out_start ? 0 : w_out_start;
    int w_out_end = (w_in + paddingW) / strideW;
    w_out_end = outputWidth - 1 < w_out_end ? outputWidth - 1 : w_out_end;

    T value = 0;

    for (int c_out = c_out_start; c_out < c_out_start + filterMultiplier;
         c_out++) {
      for (int h_out = h_out_start; h_out <= h_out_end; ++h_out) {
        const int filter_h = h_in + paddingH - h_out * strideH;
        for (int w_out = w_out_start; w_out <= w_out_end; ++w_out) {
          const int filter_w = w_in + paddingW - w_out * strideW;
          const int filter_offset = c_out * filterHeight * filterWidth +
                                    filter_h * filterWidth + filter_w;
          const int top_diff_offset =
              ((batch * outputChannels + c_out) * outputHeight + h_out) *
                  outputWidth +
              w_out;
          value += top_diff[top_diff_offset] * weight_data[filter_offset];
        }
      }
    }
    bottom_diff[index] += value;
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t filter.
template <typename T>
__global__ void KernelDepthwiseConvFilterGrad(const int num_i,
                                       const int nthreads,
                                       const T* const top_diff,
                                       const T* const inputData,
                                       const int num,
                                       const int outputChannels,
                                       const int outputHeight,
                                       const int outputWidth,
                                       const int inputChannels,
                                       const int inputHeight,
                                       const int inputWidth,
                                       const int filterMultiplier,
                                       const int filterHeight,
                                       const int filterWidth,
                                       const int strideH,
                                       const int strideW,
                                       const int paddingH,
                                       const int paddingW,
                                       T* const buffer_data) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int h_out = (index / outputWidth) % outputHeight;
    const int w_out = index % outputWidth;
    const int kh =
        (index / filterWidth / outputHeight / outputWidth) % filterHeight;
    const int kw = (index / outputHeight / outputWidth) % filterWidth;
    const int h_in = -paddingH + h_out * strideH + kh;
    const int w_in = -paddingW + w_out * strideW + kw;
    if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
        (w_in < inputWidth)) {
      const int c_out =
          index / (filterHeight * filterWidth * outputHeight * outputWidth);
      const int c_in = c_out / filterMultiplier;
      const int batch = num_i;
      const int top_offset =
          ((batch * outputChannels + c_out) * outputHeight + h_out) *
              outputWidth + w_out;
      const int bottom_offset =
          ((batch * inputChannels + c_in) * inputHeight + h_in) * inputWidth +
          w_in;
      buffer_data[index] = top_diff[top_offset] * inputData[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}
*/

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T>
class DepthwiseConvFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& filter, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* filter_data = filter.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelDepthwiseConv<T><<<grid, threads, 0, STREAM_DEFAULT>>>(
        nthreads, input_data, filter_data, batch_size, output_channels,
        output_height, output_width, input_channels, input_height, input_width,
        output_channels / input_channels, ksize_height, ksize_width,
        stride_height, stride_width, padding_height, padding_width,
        output_data);
  }
};

/*

template <typename T>
class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext, PoolProcess, T>
{
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_process, framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool2DGrad<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        pool_process, input_grad_data);
  }
};

template <typename T>
class DepthwiseConvdFilterGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  framework::Tensor* input_grad) {
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
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2DGrad<T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        input_grad_data);
  }
};
*/

template class DepthwiseConvFunctor<platform::CUDADeviceContext,
                                    paddle::operators::math::MaxPool<float>,
                                    float>;

/*
template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;
template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;

template class DepthwiseConvFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::MaxPool<double>, double>;
template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
*/

}  // namespace math
}  // namespace operators
}  // namespace paddle
