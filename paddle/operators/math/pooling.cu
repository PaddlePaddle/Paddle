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

template <typename PoolProcess, typename T>
__global__ void KernelPool2D(const int nthreads, const T* input_data,
                             T* output_data, const int channels,
                             const int input_height, const int input_width,
                             const int output_height, const int output_width,
                             const int ksize_height, const int ksize_width,
                             const int stride_height, const int stride_width,
                             const int padding_height, const int padding_width,
                             PoolProcess pool_process) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
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
    T ele = pool_process.initial();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        pool_process.compute(ele, input_data[h * input_width + w]);
      }
    }
    int pool_size = (hend - hstart) * (wend - wstart);
    pool_process.finalize(ele, (static_cast<T>(pool_size)));
    output_data[index] = ele;
  }
}

template <typename PoolProcess, typename T>
__global__ void KernelPool2DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, T* input_grad, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const int ksize_height, const int ksize_width,
    const int stride_height, const int stride_width, const int padding_height,
    const int padding_width, PoolProcess pool_process) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width + padding_width;
    int offsetH = (index / input_width) % input_height + padding_height;
    int offsetC = (index / input_width / input_height) % channels;
    int batch_idx = index / input_width / input_height / channels;

    int phstart = (offsetH < ksize_height)
                      ? 0
                      : (offsetH - ksize_height) / stride_height + 1;
    int pwstart = (offsetW < ksize_width)
                      ? 0
                      : (offsetW - ksize_width) / stride_width + 1;
    int phend = min(offsetH / stride_height + 1, output_height);
    int pwend = min(offsetW / stride_width + 1, output_width);
    T gradient = 0;
    T input = input_data[index];
    int output_idx =
        (batch_idx * channels + offsetC) * output_height * output_width;
    output_data += output_idx;
    output_grad += output_idx;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = min(hstart + ksize_height, input_height);
        int wend = min(wstart + ksize_width, input_width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
        int output_sub_idx = ph * output_width + pw;
        pool_process.compute(input, output_data[output_sub_idx],
                             output_grad[output_sub_idx], gradient,
                             static_cast<T>(1.0 / pool_size));
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
__global__ void KernelMaxPool2DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, T* input_grad, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const int ksize_height, const int ksize_width,
    const int stride_height, const int stride_width, const int padding_height,
    const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
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
    input_grad += (batch_idx * channels + c) * input_height * input_width;

    T ele = output_data[index];
    int maxIndex = -1;
    bool stop = false;
    for (int h = hstart; h < hend && !stop; ++h) {
      for (int w = wstart; w < wend && !stop; ++w) {
        if (ele == input_data[h * input_width + w]) {
          maxIndex = h * input_width + w;
          stop = true;
        }
      }
    }

    if (maxIndex != -1) {
      // atomic add
      atomicAdd(input_grad + maxIndex, output_grad[index]);
    }
  }
}

template <typename PoolProcess, typename T>
class Pool2dFunctor<platform::GPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process) {
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

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool2D<
        PoolProcess,
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(nthreads, input_data, output_data, input_channels,
                              input_height, input_width, output_height,
                              output_width, ksize_height, ksize_width,
                              stride_height, stride_width, padding_height,
                              padding_width, pool_process);
  }
};

template <typename PoolProcess, typename T>
class Pool2dGradFunctor<platform::GPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_process) {
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
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool2DGrad<
        PoolProcess,
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_grad_data,
        input_channels, input_height, input_width, output_height, output_width,
        ksize_height, ksize_width, stride_height, stride_width, padding_height,
        padding_width, pool_process);
  }
};

template <typename T>
class MaxPool2dGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
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
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2DGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_grad_data,
        input_channels, input_height, input_width, output_height, output_width,
        ksize_height, ksize_width, stride_height, stride_width, padding_height,
        padding_width);
  }
};

template class MaxPool2dGradFunctor<platform::GPUPlace, float>;
// template class MaxPool2dGradFunctor<platform::GPUPlace, double>; // The
// 64-bit floating-point version of atomicAdd() is only supported by devices of
// compute capability 6.x and higher.

template class Pool2dFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool2dFunctor<platform::GPUPlace,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool2dGradFunctor<
    platform::GPUPlace, paddle::operators::math::MaxPoolGrad<float>, float>;
template class Pool2dGradFunctor<
    platform::GPUPlace, paddle::operators::math::AvgPoolGrad<float>, float>;
template class Pool2dFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool2dFunctor<platform::GPUPlace,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool2dGradFunctor<
    platform::GPUPlace, paddle::operators::math::MaxPoolGrad<double>, double>;
template class Pool2dGradFunctor<
    platform::GPUPlace, paddle::operators::math::AvgPoolGrad<double>, double>;

template <typename PoolProcess, typename T>
__global__ void KernelPool3D(
    const int nthreads, const T* input_data, T* output_data, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    PoolProcess pool_process) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
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
    T ele = pool_process.initial();
    input_data +=
        (batch_idx * channels + c) * input_depth * input_height * input_width;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          pool_process.compute(
              ele, input_data[(d * input_height + h) * input_width + w]);
        }
      }
    }
    int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    pool_process.finalize(ele, static_cast<T>(pool_size));
    output_data[index] = ele;
  }
}

template <typename PoolProcess, typename T>
__global__ void KernelPool3DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, T* input_grad, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    PoolProcess pool_process) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width + padding_width;
    int offsetH = (index / input_width) % input_height + padding_height;
    int offsetD =
        (index / input_width / input_height) % input_depth + padding_depth;
    int offsetC = (index / input_width / input_height / input_depth) % channels;
    int batch_idx = index / input_width / input_height / input_depth / channels;

    int pdstart = (offsetD < ksize_depth)
                      ? 0
                      : (offsetD - ksize_depth) / stride_depth + 1;
    int phstart = (offsetH < ksize_height)
                      ? 0
                      : (offsetH - ksize_height) / stride_height + 1;
    int pwstart = (offsetW < ksize_width)
                      ? 0
                      : (offsetW - ksize_width) / stride_width + 1;
    int pdend = min((offsetD) / stride_depth + 1, output_depth);
    int phend = min((offsetH) / stride_height + 1, output_height);
    int pwend = min((offsetW) / stride_width + 1, output_width);

    T gradient = 0;
    T input = input_data[index];
    int output_idx = (batch_idx * channels + offsetC) * output_depth *
                     output_height * output_width;
    output_data += output_idx;
    output_grad += output_idx;

    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int dstart = pd * stride_depth - padding_depth;
          int hstart = ph * stride_height - padding_height;
          int wstart = pw * stride_width - padding_width;
          int dend = min(dstart + ksize_depth, input_depth);
          int hend = min(hstart + ksize_height, input_height);
          int wend = min(wstart + ksize_width, input_width);
          dstart = max(dstart, 0);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          int output_sub_idx = (pd * output_height + ph) * output_width + pw;
          pool_process.compute(input, output_data[output_sub_idx],
                               output_grad[output_sub_idx], gradient,
                               static_cast<T>(1.0 / pool_size));
        }
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
__global__ void KernelMaxPool3DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, T* input_grad, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height,
    const int padding_width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
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
    T ele = output_data[index];
    bool stop = false;
    int maxIdx = -1;
    input_data +=
        (batch_idx * channels + c) * input_depth * input_height * input_width;
    input_grad +=
        (batch_idx * channels + c) * input_depth * input_height * input_width;

    for (int d = dstart; d < dend && !stop; ++d) {
      for (int h = hstart; h < hend && !stop; ++h) {
        for (int w = wstart; w < wend && !stop; ++w) {
          if (ele == input_data[(d * input_height + h) * input_width + w]) {
            stop = true;
            maxIdx = (d * input_height + h) * input_width + w;
          }
        }
      }
    }
    if (maxIdx != -1) {
      // atomic add
      atomicAdd(input_grad + maxIdx, output_grad[index]);
    }
  }
}

template <typename PoolProcess, class T>
class Pool3dFunctor<platform::GPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process) {
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

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<
        PoolProcess,
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, input_channels, input_depth,
        input_height, input_width, output_depth, output_height, output_width,
        ksize_depth, ksize_height, ksize_width, stride_depth, stride_height,
        stride_width, padding_depth, padding_height, padding_width,
        pool_process);
  }
};

template <typename PoolProcess, class T>
class Pool3dGradFunctor<platform::GPUPlace, PoolProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_process) {
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
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3DGrad<
        PoolProcess,
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_grad_data,
        input_channels, input_depth, input_height, input_width, output_depth,
        output_height, output_width, ksize_depth, ksize_height, ksize_width,
        stride_depth, stride_height, stride_width, padding_depth,
        padding_height, padding_width, pool_process);
  }
};

template <class T>
class MaxPool3dGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
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
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_grad_data,
        input_channels, input_depth, input_height, input_width, output_depth,
        output_height, output_width, ksize_depth, ksize_height, ksize_width,
        stride_depth, stride_height, stride_width, padding_depth,
        padding_height, padding_width);
  }
};

template class MaxPool3dGradFunctor<platform::GPUPlace, float>;
// template class MaxPool3dGradFunctor<platform::GPUPlace, double>;  // The
// 64-bit floating-point version of atomicAdd() is only supported by devices of
// compute capability 6.x and higher.

template class Pool3dFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool3dFunctor<platform::GPUPlace,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool3dGradFunctor<
    platform::GPUPlace, paddle::operators::math::MaxPoolGrad<float>, float>;
template class Pool3dGradFunctor<
    platform::GPUPlace, paddle::operators::math::AvgPoolGrad<float>, float>;
template class Pool3dFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool3dFunctor<platform::GPUPlace,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool3dGradFunctor<
    platform::GPUPlace, paddle::operators::math::MaxPoolGrad<double>, double>;
template class Pool3dGradFunctor<
    platform::GPUPlace, paddle::operators::math::AvgPoolGrad<double>, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
