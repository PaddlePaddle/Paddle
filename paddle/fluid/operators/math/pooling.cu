/* Copyright (c) 2016 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/math/pooling.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
namespace math {

__device__ __forceinline__ int ADAPT_START_INDEX(int ph, int input_size,
                                                 int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

__device__ __forceinline__ int ADAPT_END_INDEX(int ph, int input_size,
                                               int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

template <typename PoolProcess, typename T>
__global__ void KernelPool2D(const int nthreads, const T* input_data,
                             const int channels, const int input_height,
                             const int input_width, const int output_height,
                             const int output_width, const int ksize_height,
                             const int ksize_width, const int stride_height,
                             const int stride_width, const int padding_height,
                             const int padding_width, PoolProcess pool_process,
                             bool exclusive, bool adaptive, T* output_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % channels;
    int batch_idx = index / output_width / output_height / channels;

    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      hstart = ADAPT_START_INDEX(ph, input_height, output_height);
      hend = ADAPT_END_INDEX(ph, input_height, output_height);

      wstart = ADAPT_START_INDEX(pw, input_width, output_width);
      wend = ADAPT_END_INDEX(pw, input_width, output_width);
    } else {
      hstart = ph * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);

      wstart = pw * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    input_data += (batch_idx * channels + c) * input_height * input_width;
    T ele = pool_process.initial();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        pool_process.compute(input_data[h * input_width + w], &ele);
      }
    }
    int pool_size = (exclusive || adaptive) ? (hend - hstart) * (wend - wstart)
                                            : ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}

template <typename PoolProcess, typename T>
__global__ void KernelPool2DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, const int channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const int ksize_height, const int ksize_width, const int stride_height,
    const int stride_width, const int padding_height, const int padding_width,
    PoolProcess pool_process, bool exclusive, bool adaptive, T* input_grad) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width + padding_width;
    int offsetH = (index / input_width) % input_height + padding_height;
    int offsetC = (index / input_width / input_height) % channels;
    int batch_idx = index / input_width / input_height / channels;

    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      phstart = offsetH * output_height / input_height;
      phend =
          min((offsetH + 1) * output_height / input_height + 1, output_height);
      pwstart = offsetW * output_width / input_width;
      pwend = min((offsetW + 1) * output_width / input_width + 1, output_width);
    } else {
      phstart = (offsetH < ksize_height)
                    ? 0
                    : (offsetH - ksize_height) / stride_height + 1;
      pwstart = (offsetW < ksize_width)
                    ? 0
                    : (offsetW - ksize_width) / stride_width + 1;
      phend = min(offsetH / stride_height + 1, output_height);
      pwend = min(offsetW / stride_width + 1, output_width);
    }
    T gradient = 0;
    T input = input_data[index];
    int output_idx =
        (batch_idx * channels + offsetC) * output_height * output_width;
    output_data += output_idx;
    output_grad += output_idx;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pool_size;
        if (adaptive) {
          pool_size = static_cast<int>(ceil(static_cast<double>(input_height) /
                                            ksize_height)) *
                      static_cast<int>(
                          ceil(static_cast<double>(input_width) / ksize_width));
        } else {
          int hstart = ph * stride_height - padding_height;
          int wstart = pw * stride_width - padding_width;
          int hend = min(hstart + ksize_height, input_height);
          int wend = min(wstart + ksize_width, input_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          pool_size = exclusive ? (hend - hstart) * (wend - wstart)
                                : ksize_height * ksize_width;
        }
        int output_sub_idx = ph * output_width + pw;
        pool_process.compute(input, output_data[output_sub_idx],
                             output_grad[output_sub_idx],
                             static_cast<T>(1.0 / pool_size), &gradient);
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
__global__ void KernelMaxPool2DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, const int channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const int ksize_height, const int ksize_width, const int stride_height,
    const int stride_width, const int padding_height, const int padding_width,
    T* input_grad) {
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
      platform::CudaAtomicAdd(input_grad + maxIndex, output_grad[index]);
    }
  }
}

template <typename PoolProcess, typename T>
void Pool2dDirectCUDAFunctor<PoolProcess, T>::operator()(
    const T* input, const std::vector<int>& input_shape,
    const std::vector<int>& output_shape, const std::vector<int>& ksize,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    PoolProcess pool_compute, bool exclusive, T* output, cudaStream_t stream) {
  const int batch_size = input_shape[0];
  const int input_channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_channels = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];
  const int ksize_height = ksize[0];
  const int ksize_width = ksize[1];
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  int nthreads = batch_size * output_channels * output_height * output_width;
  int blocks = (nthreads + 1024 - 1) / 1024;
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);

  KernelPool2D<PoolProcess, T><<<grid, threads, 0, stream>>>(
      nthreads, input, input_channels, input_height, input_width, output_height,
      output_width, ksize_height, ksize_width, stride_height, stride_width,
      padding_height, padding_width, pool_compute, exclusive, false, output);
}

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
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
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool2D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, pool_process, exclusive,
        adaptive, output_data);
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dGradFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive,
                  framework::Tensor* input_grad) {
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
        pool_process, exclusive, adaptive, input_grad_data);
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T>
class MaxPool2dGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
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

template class Pool2dDirectCUDAFunctor<paddle::operators::math::MaxPool<float>,
                                       float>;
template class Pool2dDirectCUDAFunctor<paddle::operators::math::AvgPool<float>,
                                       float>;

template class MaxPool2dGradFunctor<platform::CUDADeviceContext, float>;
template class MaxPool2dGradFunctor<platform::CUDADeviceContext, double>;

template class Pool2dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool2dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool2dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;
template class Pool2dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::AvgPoolGrad<float>,
                                 float>;
template class Pool2dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool2dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool2dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
template class Pool2dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::AvgPoolGrad<double>,
                                 double>;

template <typename PoolProcess, typename T>
__global__ void KernelPool3D(
    const int nthreads, const T* input_data, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    PoolProcess pool_process, bool exclusive, bool adaptive, T* output_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int pd = (index / output_width / output_height) % output_depth;
    int c = (index / output_width / output_height / output_depth) % channels;
    int batch_idx =
        index / output_width / output_height / output_depth / channels;

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      dstart = ADAPT_START_INDEX(pd, input_depth, output_depth);
      dend = ADAPT_END_INDEX(pd, input_depth, output_depth);

      hstart = ADAPT_START_INDEX(ph, input_height, output_height);
      hend = ADAPT_END_INDEX(ph, input_height, output_height);

      wstart = ADAPT_START_INDEX(pw, input_width, output_width);
      wend = ADAPT_END_INDEX(pw, input_width, output_width);
    } else {
      dstart = pd * stride_depth - padding_depth;
      hstart = ph * stride_height - padding_height;
      wstart = pw * stride_width - padding_width;
      dend = min(dstart + ksize_depth, input_depth);
      hend = min(hstart + ksize_height, input_height);
      wend = min(wstart + ksize_width, input_width);
      dstart = max(dstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
    }
    T ele = pool_process.initial();
    input_data +=
        (batch_idx * channels + c) * input_depth * input_height * input_width;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          pool_process.compute(
              input_data[(d * input_height + h) * input_width + w], &ele);
        }
      }
    }
    int pool_size = (exclusive || adaptive)
                        ? (dend - dstart) * (hend - hstart) * (wend - wstart)
                        : ksize_depth * ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}

template <typename PoolProcess, typename T>
__global__ void KernelPool3DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, const int channels, const int input_depth,
    const int input_height, const int input_width, const int output_depth,
    const int output_height, const int output_width, const int ksize_depth,
    const int ksize_height, const int ksize_width, const int stride_depth,
    const int stride_height, const int stride_width, const int padding_depth,
    const int padding_height, const int padding_width, PoolProcess pool_process,
    bool exclusive, bool adaptive, T* input_grad) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width + padding_width;
    int offsetH = (index / input_width) % input_height + padding_height;
    int offsetD =
        (index / input_width / input_height) % input_depth + padding_depth;
    int offsetC = (index / input_width / input_height / input_depth) % channels;
    int batch_idx = index / input_width / input_height / input_depth / channels;

    int pdstart, pdend;
    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      pdstart = offsetD * output_depth / input_depth;
      pdend = min((offsetD + 1) * output_depth / input_depth + 1, output_depth);
      phstart = offsetH * output_height / input_height;
      phend =
          min((offsetH + 1) * output_height / input_height + 1, output_height);
      pwstart = offsetW * output_width / input_width;
      pwend = min((offsetW + 1) * output_width / input_width + 1, output_width);
    } else {
      pdstart = (offsetD < ksize_depth)
                    ? 0
                    : (offsetD - ksize_depth) / stride_depth + 1;
      phstart = (offsetH < ksize_height)
                    ? 0
                    : (offsetH - ksize_height) / stride_height + 1;
      pwstart = (offsetW < ksize_width)
                    ? 0
                    : (offsetW - ksize_width) / stride_width + 1;
      pdend = min((offsetD) / stride_depth + 1, output_depth);
      phend = min((offsetH) / stride_height + 1, output_height);
      pwend = min((offsetW) / stride_width + 1, output_width);
    }

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
          int pool_size;
          if (adaptive) {
            pool_size =
                static_cast<int>(
                    ceil(static_cast<double>(input_depth) / ksize_depth)) *
                static_cast<int>(
                    ceil(static_cast<double>(input_height) / ksize_height)) *
                static_cast<int>(
                    ceil(static_cast<double>(input_width) / ksize_width));
          } else {
            int dstart = pd * stride_depth - padding_depth;
            int hstart = ph * stride_height - padding_height;
            int wstart = pw * stride_width - padding_width;
            int dend = min(dstart + ksize_depth, input_depth);
            int hend = min(hstart + ksize_height, input_height);
            int wend = min(wstart + ksize_width, input_width);
            dstart = max(dstart, 0);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            pool_size =
                exclusive ? (dend - dstart) * (hend - hstart) * (wend - wstart)
                          : ksize_depth * ksize_height * ksize_width;
          }
          int output_sub_idx = (pd * output_height + ph) * output_width + pw;
          pool_process.compute(input, output_data[output_sub_idx],
                               output_grad[output_sub_idx],
                               static_cast<T>(1.0 / pool_size), &gradient);
        }
      }
    }
    input_grad[index] = gradient;
  }
}

template <typename T>
__global__ void KernelMaxPool3DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, const int channels, const int input_depth,
    const int input_height, const int input_width, const int output_depth,
    const int output_height, const int output_width, const int ksize_depth,
    const int ksize_height, const int ksize_width, const int stride_depth,
    const int stride_height, const int stride_width, const int padding_depth,
    const int padding_height, const int padding_width, T* input_grad) {
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
      platform::CudaAtomicAdd(input_grad + maxIdx, output_grad[index]);
    }
  }
}

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
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
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_depth, input_height,
        input_width, output_depth, output_height, output_width, ksize_depth,
        ksize_height, ksize_width, stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width, pool_process, exclusive,
        adaptive, output_data);
  }
};

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dGradFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive,
                  framework::Tensor* input_grad) {
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
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3DGrad<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width, pool_process, exclusive, adaptive, input_grad_data);
  }
};

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <class T>
class MaxPool3dGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* input_grad) {
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
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DGrad<T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width, input_grad_data);
  }
};

template class MaxPool3dGradFunctor<platform::CUDADeviceContext, float>;
template class MaxPool3dGradFunctor<platform::CUDADeviceContext, double>;

template class Pool3dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool3dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool3dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;
template class Pool3dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::AvgPoolGrad<float>,
                                 float>;
template class Pool3dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool3dFunctor<platform::CUDADeviceContext,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool3dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
template class Pool3dGradFunctor<platform::CUDADeviceContext,
                                 paddle::operators::math::AvgPoolGrad<double>,
                                 double>;

template <typename T1, typename T2>
__global__ void KernelMaxPool2dWithIdx(
    const int nthreads, const T1* input_data, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const int ksize_height, const int ksize_width,
    const int stride_height, const int stride_width, const int padding_height,
    const int padding_width, bool adaptive, T1* output_data, T2* mask_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % channels;
    int batch_idx = index / output_width / output_height / channels;

    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      hstart = ADAPT_START_INDEX(ph, input_height, output_height);
      hend = ADAPT_END_INDEX(ph, input_height, output_height);

      wstart = ADAPT_START_INDEX(pw, input_width, output_width);
      wend = ADAPT_END_INDEX(pw, input_width, output_width);
    } else {
      hstart = ph * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);

      wstart = pw * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    input_data += (batch_idx * channels + c) * input_height * input_width;
    T1 ele = -FLT_MAX;
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

template <typename T1, typename T2>
__global__ void KernelMaxPool2DWithIdxGrad(
    const int nthreads, const T1* output_grad, const T2* mask_data,
    const int channels, const int input_height, const int input_width,
    const int output_height, const int output_width, const int ksize_height,
    const int ksize_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, bool adaptive,
    T1* input_grad) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width;
    int offsetH = (index / input_width) % input_height;
    int offsetC = (index / input_width / input_height) % channels;
    int batch_idx = index / input_width / input_height / channels;

    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      phstart = offsetH * output_height / input_height;
      phend =
          min((offsetH + 1) * output_height / input_height + 1, output_height);
      pwstart = offsetW * output_width / input_width;
      pwend = min((offsetW + 1) * output_width / input_width + 1, output_width);
    } else {
      phstart =
          (offsetH + padding_height < ksize_height)
              ? 0
              : (offsetH + padding_height - ksize_height) / stride_height + 1;
      pwstart =
          (offsetW + padding_width < ksize_width)
              ? 0
              : (offsetW + padding_width - ksize_width) / stride_width + 1;
      phend =
          min((offsetH + padding_height) / stride_height + 1, output_height);
      pwend = min((offsetW + padding_width) / stride_width + 1, output_width);
    }

    T1 gradient = 0;
    int input_current_featuremap_idx = offsetH * input_width + offsetW;
    int output_idx =
        (batch_idx * channels + offsetC) * output_height * output_width;

    mask_data += output_idx;
    output_grad += output_idx;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_data[ph * output_width + pw] == input_current_featuremap_idx)
          gradient += output_grad[ph * output_width + pw];
      }
    }
    input_grad[index] = gradient;
  }
}

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexFunctor<platform::CUDADeviceContext, T1, T2> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask) {
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

    const T1* input_data = input.data<T1>();
    T1* output_data = output->mutable_data<T1>(context.GetPlace());
    T2* mask_data = mask->mutable_data<T2>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2dWithIdx<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, adaptive, output_data,
        mask_data);
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor<platform::CUDADeviceContext, T1, T2> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_channels = input_grad->dims()[1];
    const int input_height = input_grad->dims()[2];
    const int input_width = input_grad->dims()[3];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = input_grad->mutable_data<T1>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool2DWithIdxGrad<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, output_grad_data, mask_data, input_channels, input_height,
        input_width, output_height, output_width, ksize_height, ksize_width,
        stride_height, stride_width, padding_height, padding_width, adaptive,
        input_grad_data);
  }
};

template class MaxPool2dWithIndexFunctor<platform::CUDADeviceContext, float,
                                         int>;
template class MaxPool2dWithIndexGradFunctor<platform::CUDADeviceContext, float,
                                             int>;
template class MaxPool2dWithIndexFunctor<platform::CUDADeviceContext, double,
                                         int>;
template class MaxPool2dWithIndexGradFunctor<platform::CUDADeviceContext,
                                             double, int>;

template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdx(
    const int nthreads, const T1* input_data, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    bool adaptive, T1* output_data, T2* mask_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int pd = (index / output_width / output_height) % output_depth;
    int c = (index / output_width / output_height / output_depth) % channels;
    int batch_idx =
        index / output_width / output_height / output_depth / channels;

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      dstart = ADAPT_START_INDEX(pd, input_depth, output_depth);
      dend = ADAPT_END_INDEX(pd, input_depth, output_depth);

      hstart = ADAPT_START_INDEX(ph, input_height, output_height);
      hend = ADAPT_END_INDEX(ph, input_height, output_height);

      wstart = ADAPT_START_INDEX(pw, input_width, output_width);
      wend = ADAPT_END_INDEX(pw, input_width, output_width);
    } else {
      dstart = pd * stride_depth - padding_depth;
      hstart = ph * stride_height - padding_height;
      wstart = pw * stride_width - padding_width;
      dend = min(dstart + ksize_depth, input_depth);
      hend = min(hstart + ksize_height, input_height);
      wend = min(wstart + ksize_width, input_width);
      dstart = max(dstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
    }

    T1 ele = -FLT_MAX;
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

template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdxGrad(
    const int nthreads, const T1* output_grad, const T2* mask,
    const int channels, const int input_depth, const int input_height,
    const int input_width, const int output_depth, const int output_height,
    const int output_width, const int ksize_depth, const int ksize_height,
    const int ksize_width, const int stride_depth, const int stride_height,
    const int stride_width, const int padding_depth, const int padding_height,
    const int padding_width, bool adaptive, T1* input_grad) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int offsetW = index % input_width;
    int offsetH = (index / input_width) % input_height;
    int offsetD = (index / input_width / input_height) % input_depth;
    int offsetC = (index / input_width / input_height / input_depth) % channels;
    int batch_idx = index / input_width / input_height / input_depth / channels;

    int pdstart, pdend;
    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      pdstart = offsetD * output_depth / input_depth;
      pdend = min((offsetD + 1) * output_depth / input_depth + 1, output_depth);
      phstart = offsetH * output_height / input_height;
      phend =
          min((offsetH + 1) * output_height / input_height + 1, output_height);
      pwstart = offsetW * output_width / input_width;
      pwend = min((offsetW + 1) * output_width / input_width + 1, output_width);
    } else {
      pdstart =
          (offsetD + padding_depth < ksize_depth)
              ? 0
              : (offsetD + padding_depth - ksize_depth) / stride_depth + 1;
      phstart =
          (offsetH + padding_height < ksize_height)
              ? 0
              : (offsetH + padding_height - ksize_height) / stride_height + 1;
      pwstart =
          (offsetW + padding_width < ksize_width)
              ? 0
              : (offsetW + padding_width - ksize_width) / stride_width + 1;
      pdend = min((offsetD + padding_depth) / stride_depth + 1, output_depth);
      phend =
          min((offsetH + padding_height) / stride_height + 1, output_height);
      pwend = min((offsetW + padding_width) / stride_width + 1, output_width);
    }

    T1 gradient = 0;
    int input_current_feature_map_idx =
        (offsetD * input_height + offsetH) * input_width + offsetW;
    int output_idx = (batch_idx * channels + offsetC) * output_depth *
                     output_height * output_width;
    mask += output_idx;
    output_grad += output_idx;

    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
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

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexFunctor<platform::CUDADeviceContext, T1, T2> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T1* input_data = input.data<T1>();
    T1* output_data = output->mutable_data<T1>(context.GetPlace());
    T2* mask_data = mask->mutable_data<T2>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DWithIdx<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_depth, input_height,
        input_width, output_depth, output_height, output_width, ksize_depth,
        ksize_height, ksize_width, stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width, adaptive, output_data,
        mask_data);
  }
};

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor<platform::CUDADeviceContext, T1, T2> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_channels = input_grad->dims()[1];
    const int input_depth = input_grad->dims()[2];
    const int input_height = input_grad->dims()[3];
    const int input_width = input_grad->dims()[4];
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

    const T1* output_grad_data = output_grad.data<T1>();
    const T2* mask_data = mask.data<T2>();
    T1* input_grad_data = input_grad->mutable_data<T1>(context.GetPlace());

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DWithIdxGrad<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, output_grad_data, mask_data, input_channels, input_depth,
        input_height, input_width, output_depth, output_height, output_width,
        ksize_depth, ksize_height, ksize_width, stride_depth, stride_height,
        stride_width, padding_depth, padding_height, padding_width, adaptive,
        input_grad_data);
  }
};

template class MaxPool3dWithIndexFunctor<platform::CUDADeviceContext, float,
                                         int>;
template class MaxPool3dWithIndexGradFunctor<platform::CUDADeviceContext, float,
                                             int>;
template class MaxPool3dWithIndexFunctor<platform::CUDADeviceContext, double,
                                         int>;
template class MaxPool3dWithIndexGradFunctor<platform::CUDADeviceContext,
                                             double, int>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
