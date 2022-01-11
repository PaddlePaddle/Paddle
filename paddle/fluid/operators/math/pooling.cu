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
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/fast_divmod.h"

#ifdef __HIPCC__
#define POOLING_BLOCK_SIZE 256
#else
#define POOLING_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {
namespace math {

struct FastDivModForPooling {
 public:
  platform::FastDivMod channel;
  platform::FastDivMod width;
  platform::FastDivMod height;

  explicit HOSTDEVICE FastDivModForPooling(const int channels,
                                           const int output_width,
                                           const int output_height) {
    channel = platform::FastDivMod(channels);
    width = platform::FastDivMod(output_width);
    height = platform::FastDivMod(output_height);
  }
};

struct FastDivModForPoolingWithMoreStaff {
 public:
  platform::FastDivMod channel;
  platform::FastDivMod width;
  platform::FastDivMod height;
  platform::FastDivMod ksize_w;
  platform::FastDivMod ksize_h;
  platform::FastDivMod stride_w;
  platform::FastDivMod stride_h;

  explicit HOSTDEVICE FastDivModForPoolingWithMoreStaff(
      const int channels, const int input_width, const int input_height,
      const int ksize_width, const int ksize_height, const int stride_width,
      const int stride_height) {
    channel = platform::FastDivMod(channels);
    width = platform::FastDivMod(input_width);
    height = platform::FastDivMod(input_height);
    ksize_w = platform::FastDivMod(ksize_width);
    ksize_h = platform::FastDivMod(ksize_height);
    stride_w = platform::FastDivMod(stride_width);
    stride_h = platform::FastDivMod(stride_height);
  }
};

template <typename FastDivModForPooling>
__device__ void OffsetPreparationFor4Dimension(
    int index, bool channel_last, FastDivModForPooling divmods,
    const int pad_width, const int pad_height, const int aux_width,
    const int aux_height, int* w_offset, int* h_offset, int* c_offset,
    int* stride) {
  if (!channel_last) { /* NCHW */
    auto input_width_divmod = divmods.width.Divmod(index);
    auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
    auto channel_divmod = divmods.channel.Divmod(input_height_divmod.val[0]);
    *w_offset = input_width_divmod.val[1] + pad_width;
    *h_offset = input_height_divmod.val[1] + pad_height;
    *c_offset = channel_divmod.val[1];
    *stride = (channel_divmod.val[0] * divmods.channel.divisor + *c_offset) *
              aux_height * aux_width;
  } else { /* NHWC */
    auto c_divmod = divmods.channel.Divmod(index);
    auto input_width_divmod = divmods.width.Divmod(c_divmod.val[0]);
    auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
    *c_offset = c_divmod.val[1];
    *w_offset = input_width_divmod.val[1] + pad_width;
    *h_offset = input_height_divmod.val[1] + pad_height;
    *stride = input_height_divmod.val[0] * aux_height * aux_width *
              divmods.channel.divisor;
  }
}

int GetThreadsPerBlock(const platform::CUDADeviceContext& ctx,
                       int threads_per_block, int64_t numel) {
  int sm_count = ctx.GetSMCount();
  if (numel / (sm_count << 1) < threads_per_block) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads_per_block = platform::RoundToPowerOfTwo(numel / (sm_count << 1));
  } else if (numel / (sm_count << 2) < threads_per_block) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads_per_block = platform::RoundToPowerOfTwo(numel / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  return std::max(64, threads_per_block);
}

template <typename PoolProcess, typename T>
__global__ void KernelPool2D(
    const int nthreads, const T* input_data, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const int ksize_height, const int ksize_width,
    const int stride_height, const int stride_width, const int padding_height,
    const int padding_width, FastDivModForPooling divmods,
    PoolProcess pool_process, bool exclusive, bool adaptive, T* output_data,
    bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int hstart, hend, wstart, wend;
    int w_offset, h_offset, c_offset, input_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(
        index, channel_last, divmods, 0, 0, input_width, input_height,
        &w_offset, &h_offset, &c_offset, &input_offset);
    input_data += input_offset;

    if (adaptive) {
      hstart = AdaptStartIndex(h_offset, input_height, output_height);
      hend = AdaptEndIndex(h_offset, input_height, output_height);
      wstart = AdaptStartIndex(w_offset, input_width, output_width);
      wend = AdaptEndIndex(w_offset, input_width, output_width);
    } else {
      hstart = h_offset * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);
      wstart = w_offset * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    T ele = pool_process.initial();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        auto input_idx = channel_last
                             ? (h * input_width + w) * channels + c_offset
                             : h * input_width + w;
        pool_process.compute(input_data[input_idx], &ele);
      }
    }
    int pool_size = (exclusive || adaptive) ? (hend - hstart) * (wend - wstart)
                                            : ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}

template <typename T, typename PoolProcess>
__global__ void KernelPool2DGrad(
    const int nthreads, const T* __restrict__ input_data,
    const T* __restrict__ output_data, const const T* __restrict__ output_grad,
    const int output_width, const int output_height, const int input_width,
    const int input_height, const int ksize_width, const int ksize_height,
    const int stride_width, const int stride_height, const int padding_width,
    const int padding_height, FastDivModForPoolingWithMoreStaff divmods,
    PoolProcess pool_process, bool exclusive, bool adaptive,
    T* __restrict__ input_grad, bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    T input = static_cast<T>(0);
    T input_grad_data = static_cast<T>(0);
    int phstart, phend, pwstart, pwend;
    int w_offset, h_offset, c_offset, output_offset;
    OffsetPreparationFor4Dimension<>(index, channel_last, divmods,
                                     padding_width, padding_height,
                                     output_width, output_height, &w_offset,
                                     &h_offset, &c_offset, &output_offset);
    if (pool_process.use_x) {
      input = input_data[index];
      output_data += output_offset;
    }
    output_grad += output_offset;

    if (adaptive) {
      auto tmp_phend = divmods.height.Divmod((h_offset + 1) * output_height);
      auto tmp_pwend = divmods.width.Divmod((w_offset + 1) * output_width);
      phstart = divmods.height.Div(h_offset * output_height);
      pwstart = divmods.width.Div(w_offset * output_width);
      phend = tmp_phend.val[1] > 0 ? tmp_phend.val[0] + 1 : tmp_phend.val[0];
      pwend = tmp_pwend.val[1] > 0 ? tmp_pwend.val[0] + 1 : tmp_pwend.val[0];

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          auto ksize_w_divmod = divmods.ksize_w.Divmod(input_width);
          auto ksize_h_divmod = divmods.ksize_h.Divmod(input_height);
          auto tmp_width = ksize_w_divmod.val[1] > 0 ? ksize_w_divmod.val[0] + 1
                                                     : ksize_w_divmod.val[0];
          auto tmp_height = ksize_h_divmod.val[1] > 0
                                ? ksize_h_divmod.val[0] + 1
                                : ksize_h_divmod.val[0];
          int pool_size = tmp_height * tmp_width;
          int tmp_idx = ph * output_width + pw;
          int output_sub_idx =
              channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                           : tmp_idx;
          T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                             : static_cast<T>(0);
          pool_process.compute(input, ouput_value, output_grad[output_sub_idx],
                               static_cast<T>(1.0 / pool_size),
                               &input_grad_data);
        }
      }
    } else {
      auto stride_height_div = divmods.stride_h.Div(h_offset - ksize_height);
      auto stride_width_div = divmods.stride_w.Div(w_offset - ksize_width);
      phstart = (h_offset < ksize_height) ? 0 : stride_height_div + 1;
      pwstart = (w_offset < ksize_width) ? 0 : stride_width_div + 1;
      phend = min(divmods.stride_h.Div(h_offset) + 1, output_height);
      pwend = min(divmods.stride_w.Div(w_offset) + 1, output_width);

      if (exclusive) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            int hstart = ph * stride_height - padding_height;
            int wstart = pw * stride_width - padding_width;
            int hend = min(hstart + ksize_height, input_height);
            int wend = min(wstart + ksize_width, input_width);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            int pool_size = (hend - hstart) * (wend - wstart);
            int tmp_idx = ph * output_width + pw;
            int output_sub_idx =
                channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                             : tmp_idx;
            T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                               : static_cast<T>(0);
            pool_process.compute(
                input, ouput_value, output_grad[output_sub_idx],
                static_cast<T>(1.0 / pool_size), &input_grad_data);
          }
        }
      } else {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            int pool_size = ksize_height * ksize_width;
            int tmp_idx = ph * output_width + pw;
            int output_sub_idx =
                channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                             : tmp_idx;
            T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                               : static_cast<T>(0);
            pool_process.compute(
                input, ouput_value, output_grad[output_sub_idx],
                static_cast<T>(1.0 / pool_size), &input_grad_data);
          }
        }
      }
    }
    input_grad[index] = input_grad_data;
  }
}

template <typename T>
__global__ void KernelMaxPool2DGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, const int channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const int ksize_height, const int ksize_width, const int stride_height,
    const int stride_width, const int padding_height, const int padding_width,
    T* input_grad, FastDivModForPooling divmods, bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int w_offset, h_offset, c_offset, input_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(
        index, channel_last, divmods, 0, 0, input_width, input_height,
        &w_offset, &h_offset, &c_offset, &input_offset);
    input_data += input_offset;
    input_grad += input_offset;

    int hstart = h_offset * stride_height - padding_height;
    int hend = min(hstart + ksize_height, input_height);
    hstart = max(hstart, 0);

    int wstart = w_offset * stride_width - padding_width;
    int wend = min(wstart + ksize_width, input_width);
    wstart = max(wstart, 0);

    T ele = output_data[index];
    int maxIndex = -1;
    bool stop = false;
    for (int h = hstart; h < hend && !stop; ++h) {
      for (int w = wstart; w < wend && !stop; ++w) {
        int input_data_idx = channel_last
                                 ? (h * input_width + w) * channels + c_offset
                                 : h * input_width + w;
        if (ele == input_data[input_data_idx]) {
          maxIndex = input_data_idx;
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
    bool exclusive, bool adaptive, T* output, gpuStream_t stream,
    PoolProcess pool_compute) {
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
  int thread_num = 1024;
#ifdef WITH_NV_JETSON
  // platform::ChangeThreadNum(context, &thread_num);
  thread_num = 512;
#endif
  int blocks = (nthreads + thread_num - 1) / thread_num;
  dim3 threads(thread_num, 1);
  dim3 grid(blocks, 1);

  auto pool_divmods =
      FastDivModForPooling(input_channels, output_width, output_height);
  KernelPool2D<PoolProcess, T><<<grid, threads, 0, stream>>>(
      nthreads, input, input_channels, input_height, input_width, output_height,
      output_width, ksize_height, ksize_width, stride_height, stride_width,
      padding_height, padding_width, pool_divmods, pool_compute, exclusive,
      adaptive, output);
}

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* output,
                  PoolProcess pool_process) {
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
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelPool2D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, pool_divmods, pool_process,
        exclusive, adaptive, output_data);
  }
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* output, PoolProcess pool_process) {
    bool channel_last = (data_format == "NHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output->dims()[3] : output->dims()[1];
    const int output_height =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_width =
        channel_last ? output->dims()[2] : output->dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelPool2D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, pool_divmods, pool_process,
        exclusive, adaptive, output_data, channel_last);
  }
};
/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
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
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* input_grad,
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
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = GetThreadsPerBlock(context, POOLING_BLOCK_SIZE, nthreads);
    int grids = (nthreads + blocks - 1) / blocks;

    auto pool_divmods = FastDivModForPoolingWithMoreStaff(
        input_channels, input_width, input_height, ksize_width, ksize_height,
        stride_width, stride_height);

    KernelPool2DGrad<T, PoolProcess><<<grids, blocks, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, output_width,
        output_height, input_width, input_height, ksize_width, ksize_height,
        stride_width, stride_height, padding_width, padding_height,
        pool_divmods, pool_process, exclusive, adaptive, input_grad_data);
  }
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* input_grad, PoolProcess pool_process) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

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
    int blocks = GetThreadsPerBlock(context, POOLING_BLOCK_SIZE, nthreads);
    int grids = (nthreads + blocks - 1) / blocks;

    auto pool_divmods = FastDivModForPoolingWithMoreStaff(
        input_channels, input_width, input_height, ksize_width, ksize_height,
        stride_width, stride_height);

    KernelPool2DGrad<T, PoolProcess><<<grids, blocks, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, output_width,
        output_height, input_width, input_height, ksize_width, ksize_height,
        stride_width, stride_height, padding_width, padding_height,
        pool_divmods, pool_process, exclusive, adaptive, input_grad_data,
        channel_last);
  }
};

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
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

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelMaxPool2DGrad<T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        input_grad_data, pool_divmods);
  }
  void operator()(
      const platform::CUDADeviceContext& context,
      const framework::Tensor& input, const framework::Tensor& output,
      const framework::Tensor& output_grad, const std::vector<int>& ksize,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::string data_format, framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

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

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);

    KernelMaxPool2DGrad<T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        input_grad_data, pool_divmods, channel_last);
  }
};

template class Pool2dDirectCUDAFunctor<paddle::operators::math::MaxPool<float>,
                                       float>;
template class Pool2dDirectCUDAFunctor<paddle::operators::math::AvgPool<float>,
                                       float>;

template class MaxPool2dGradFunctor<platform::CUDADeviceContext, float>;
template class MaxPool2dGradFunctor<platform::CUDADeviceContext, double>;
template class MaxPool2dGradFunctor<platform::CUDADeviceContext,
                                    paddle::platform::float16>;

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

template class Pool2dFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::MaxPool<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool2dFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::AvgPool<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool2dGradFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::MaxPoolGrad<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool2dGradFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::AvgPoolGrad<paddle::platform::float16>,
    paddle::platform::float16>;

template <typename PoolProcess, typename T>
__global__ void KernelPool3D(
    const int nthreads, const T* input_data, const int channels,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const int ksize_depth, const int ksize_height, const int ksize_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    PoolProcess pool_process, bool exclusive, bool adaptive, T* output_data,
    bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw, ph, pd, c, batch_idx;
    if (!channel_last) {
      pw = index % output_width;
      ph = (index / output_width) % output_height;
      pd = (index / output_width / output_height) % output_depth;
      c = (index / output_width / output_height / output_depth) % channels;
      batch_idx =
          index / output_width / output_height / output_depth / channels;
    } else {
      c = index % channels;
      pw = (index / channels) % output_width;
      ph = (index / channels / output_width) % output_height;
      pd = (index / channels / output_width / output_height) % output_depth;
      batch_idx =
          index / channels / output_width / output_height / output_depth;
    }

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      dstart = AdaptStartIndex(pd, input_depth, output_depth);
      dend = AdaptEndIndex(pd, input_depth, output_depth);

      hstart = AdaptStartIndex(ph, input_height, output_height);
      hend = AdaptEndIndex(ph, input_height, output_height);

      wstart = AdaptStartIndex(pw, input_width, output_width);
      wend = AdaptEndIndex(pw, input_width, output_width);
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

    int input_data_stride;
    if (!channel_last) { /* NCDHW */
      input_data_stride =
          (batch_idx * channels + c) * input_depth * input_height * input_width;
    } else { /* NDHWC */
      input_data_stride =
          batch_idx * input_depth * input_height * input_width * channels;
    }
    input_data += input_data_stride;

    T ele = pool_process.initial();
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          auto input_data_idx =
              channel_last
                  ? ((d * input_height + h) * input_width + w) * channels + c
                  : (d * input_height + h) * input_width + w;
          pool_process.compute(input_data[input_data_idx], &ele);
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

template <typename T, typename PoolProcess>
__global__ void KernelPool3DGrad(
    const int nthreads, const T* __restrict__ input_data,
    const T* __restrict__ output_data, const T* __restrict__ output_grad,
    const int channels, const int input_depth, const int input_height,
    const int input_width, const int output_depth, const int output_height,
    const int output_width, const int ksize_depth, const int ksize_height,
    const int ksize_width, const int stride_depth, const int stride_height,
    const int stride_width, const int padding_depth, const int padding_height,
    const int padding_width, PoolProcess pool_process, bool exclusive,
    bool adaptive, T* input_grad, bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int w_offset, h_offset, d_offset, c_offset, batch_idx, output_stride;
    T input = static_cast<T>(0);
    if (!channel_last) { /* "NCDHW" */
      w_offset = index % input_width + padding_width;
      h_offset = (index / input_width) % input_height + padding_height;
      d_offset =
          (index / input_width / input_height) % input_depth + padding_depth;
      c_offset = (index / input_width / input_height / input_depth) % channels;
      batch_idx = index / input_width / input_height / input_depth / channels;
      output_stride = (batch_idx * channels + c_offset) * output_depth *
                      output_height * output_width;
    } else { /* "NDHWC" */
      c_offset = index % channels;
      w_offset = (index / channels) % input_width + padding_width;
      h_offset =
          (index / channels / input_width) % input_height + padding_height;
      d_offset = (index / channels / input_width / input_height) % input_depth +
                 padding_depth;
      batch_idx = index / channels / input_width / input_height / input_depth;
      output_stride =
          batch_idx * output_depth * output_height * output_width * channels;
    }

    int pdstart, pdend;
    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      pdstart = AdaptStartIndex(d_offset, output_depth, input_depth);
      pdend = AdaptEndIndex(d_offset, output_depth, input_depth);

      phstart = AdaptStartIndex(h_offset, output_height, input_height);
      phend = AdaptEndIndex(h_offset, output_height, input_height);

      pwstart = AdaptStartIndex(w_offset, output_width, input_width);
      pwend = AdaptEndIndex(w_offset, output_width, input_width);
    } else {
      pdstart = (d_offset < ksize_depth)
                    ? 0
                    : (d_offset - ksize_depth) / stride_depth + 1;
      phstart = (h_offset < ksize_height)
                    ? 0
                    : (h_offset - ksize_height) / stride_height + 1;
      pwstart = (w_offset < ksize_width)
                    ? 0
                    : (w_offset - ksize_width) / stride_width + 1;
      pdend = min((d_offset) / stride_depth + 1, output_depth);
      phend = min((h_offset) / stride_height + 1, output_height);
      pwend = min((w_offset) / stride_width + 1, output_width);
    }
    if (pool_process.use_x) {
      input = input_data[index];
      output_data += output_stride;
    }
    output_grad += output_stride;
    T input_grad_data = static_cast<T>(0.0);

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

          int output_sub_idx =
              channel_last
                  ? ((pd * output_height + ph) * output_width + pw) * channels +
                        c_offset
                  : (pd * output_height + ph) * output_width + pw;
          T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                             : static_cast<T>(0);
          pool_process.compute(input, ouput_value, output_grad[output_sub_idx],
                               static_cast<T>(1.0 / pool_size),
                               &input_grad_data);
        }
      }
    }
    input_grad[index] = input_grad_data;
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
    const int padding_height, const int padding_width, T* input_grad,
    bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw, ph, pd, c, batch_idx;

    if (!channel_last) { /*NCDHW*/
      pw = index % output_width;
      ph = (index / output_width) % output_height;
      pd = (index / output_width / output_height) % output_depth;
      c = (index / output_width / output_height / output_depth) % channels;
      batch_idx =
          index / output_width / output_height / output_depth / channels;
    } else { /*NDHWC*/
      c = index % channels;
      pw = (index / channels) % output_width;
      ph = (index / channels / output_width) % output_height;
      pd = (index / channels / output_width / output_height) % output_depth;
      batch_idx =
          index / channels / output_width / output_height / output_depth;
    }

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

    int input_stride;
    if (!channel_last) {
      input_stride =
          (batch_idx * channels + c) * input_depth * input_height * input_width;
    } else {
      input_stride =
          batch_idx * input_depth * input_height * input_width * channels;
    }
    input_data += input_stride;
    input_grad += input_stride;
    for (int d = dstart; d < dend && !stop; ++d) {
      for (int h = hstart; h < hend && !stop; ++h) {
        for (int w = wstart; w < wend && !stop; ++w) {
          int input_data_idx =
              channel_last
                  ? ((d * input_height + h) * input_width + w) * channels + c
                  : (d * input_height + h) * input_width + w;
          if (ele == input_data[input_data_idx]) {
            stop = true;
            maxIdx = input_data_idx;
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

template <typename PoolProcess, typename T>
void Pool3dDirectCUDAFunctor<PoolProcess, T>::operator()(
    const T* input, const std::vector<int>& input_shape,
    const std::vector<int>& output_shape, const std::vector<int>& ksize,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    bool exclusive, bool adaptive, T* output, gpuStream_t stream,
    PoolProcess pool_compute) {
  const int batch_size = input_shape[0];
  const int input_channels = input_shape[1];
  const int input_depth = input_shape[2];
  const int input_height = input_shape[3];
  const int input_width = input_shape[4];
  const int output_channels = output_shape[1];
  const int output_depth = output_shape[2];
  const int output_height = output_shape[3];
  const int output_width = output_shape[4];
  const int ksize_depth = ksize[0];
  const int ksize_height = ksize[1];
  const int ksize_width = ksize[2];
  const int stride_depth = strides[0];
  const int stride_height = strides[1];
  const int stride_width = strides[2];
  const int padding_depth = paddings[0];
  const int padding_height = paddings[1];
  const int padding_width = paddings[2];

  int nthreads = batch_size * output_channels * output_depth * output_height *
                 output_width;
  int thread_num = 1024;
#ifdef WITH_NV_JETSON
  thread_num = 512;
#endif
  int blocks = (nthreads + thread_num - 1) / thread_num;
  dim3 threads(thread_num, 1);
  dim3 grid(blocks, 1);

  KernelPool3D<PoolProcess, T><<<grid, threads, 0, stream>>>(
      nthreads, input, input_channels, input_depth, input_height, input_width,
      output_depth, output_height, output_width, ksize_depth, ksize_height,
      ksize_width, stride_depth, stride_height, stride_width, padding_depth,
      padding_height, padding_width, pool_compute, exclusive, adaptive, output);
}

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dFunctor<platform::CUDADeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* output,
                  PoolProcess pool_process) {
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
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_depth, input_height,
        input_width, output_depth, output_height, output_width, ksize_depth,
        ksize_height, ksize_width, stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width, pool_process, exclusive,
        adaptive, output_data);
  }
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* output, PoolProcess pool_process) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output->dims()[4] : output->dims()[1];
    const int output_depth =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_height =
        channel_last ? output->dims()[2] : output->dims()[3];
    const int output_width =
        channel_last ? output->dims()[3] : output->dims()[4];

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
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<PoolProcess, T><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_depth, input_height,
        input_width, output_depth, output_height, output_width, ksize_depth,
        ksize_height, ksize_width, stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width, pool_process, exclusive,
        adaptive, output_data, channel_last);
  }
};

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
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
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* input_grad,
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
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3DGrad<T, PoolProcess><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width, pool_process, exclusive, adaptive, input_grad_data);
  }
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* input_grad, PoolProcess pool_process) {
    bool channel_last = (data_format == "NDHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

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

    KernelPool3DGrad<T, PoolProcess><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_depth, input_height, input_width, output_depth, output_height,
        output_width, ksize_depth, ksize_height, ksize_width, stride_depth,
        stride_height, stride_width, padding_depth, padding_height,
        padding_width, pool_process, exclusive, adaptive, input_grad_data,
        channel_last);  // add channel_last
  }
};

/*
 * tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
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
  void operator()(
      const platform::CUDADeviceContext& context,
      const framework::Tensor& input, const framework::Tensor& output,
      const framework::Tensor& output_grad, const std::vector<int>& ksize,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::string data_format, framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

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
        padding_width, input_grad_data, channel_last);  // add channel_last
  }
};

template class Pool3dDirectCUDAFunctor<paddle::operators::math::MaxPool<float>,
                                       float>;
template class Pool3dDirectCUDAFunctor<paddle::operators::math::AvgPool<float>,
                                       float>;

template class MaxPool3dGradFunctor<platform::CUDADeviceContext, float>;
template class MaxPool3dGradFunctor<platform::CUDADeviceContext, double>;
template class MaxPool3dGradFunctor<platform::CUDADeviceContext,
                                    paddle::platform::float16>;

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

template class Pool3dFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::MaxPool<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool3dFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::AvgPool<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool3dGradFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::MaxPoolGrad<paddle::platform::float16>,
    paddle::platform::float16>;
template class Pool3dGradFunctor<
    platform::CUDADeviceContext,
    paddle::operators::math::AvgPoolGrad<paddle::platform::float16>,
    paddle::platform::float16>;

template <typename T1, typename T2>
__global__ void KernelMaxPool2dWithIdx(
    const int nthreads, const T1* input_data, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const int ksize_height, const int ksize_width,
    const int stride_height, const int stride_width, const int padding_height,
    const int padding_width, bool adaptive, T1* output_data, T2* mask_data,
    FastDivModForPooling divmods) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int hstart, hend, wstart, wend;
    int w_offset, h_offset, c_offset, input_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(
        index, false, divmods, 0, 0, input_width, input_height, &w_offset,
        &h_offset, &c_offset, &input_offset);
    input_data += input_offset;

    if (adaptive) {
      hstart = AdaptStartIndex(h_offset, input_height, output_height);
      hend = AdaptEndIndex(h_offset, input_height, output_height);

      wstart = AdaptStartIndex(w_offset, input_width, output_width);
      wend = AdaptEndIndex(w_offset, input_width, output_width);
    } else {
      hstart = h_offset * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);

      wstart = w_offset * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

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
    T1* input_grad, FastDivModForPooling divmods) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int phstart, phend, pwstart, pwend;
    int w_offset, h_offset, c_offset, output_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(
        index, false, divmods, 0, 0, output_width, output_height, &w_offset,
        &h_offset, &c_offset, &output_offset);
    mask_data += output_offset;
    output_grad += output_offset;

    if (adaptive) {
      phstart = h_offset * output_height / input_height;
      phend =
          min((h_offset + 1) * output_height / input_height + 1, output_height);
      pwstart = w_offset * output_width / input_width;
      pwend =
          min((w_offset + 1) * output_width / input_width + 1, output_width);
    } else {
      phstart =
          (h_offset + padding_height < ksize_height)
              ? 0
              : (h_offset + padding_height - ksize_height) / stride_height + 1;
      pwstart =
          (w_offset + padding_width < ksize_width)
              ? 0
              : (w_offset + padding_width - ksize_width) / stride_width + 1;
      phend =
          min((h_offset + padding_height) / stride_height + 1, output_height);
      pwend = min((w_offset + padding_width) / stride_width + 1, output_width);
    }

    T1 input_grad_data = 0;
    int input_current_featuremap_idx = h_offset * input_width + w_offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_data[ph * output_width + pw] == input_current_featuremap_idx)
          input_grad_data += output_grad[ph * output_width + pw];
      }
    }
    input_grad[index] = input_grad_data;
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
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif

    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelMaxPool2dWithIdx<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, adaptive, output_data,
        mask_data, pool_divmods);
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

    auto pool_divmods =
        FastDivModForPooling(input_channels, input_width, input_height);
    KernelMaxPool2DWithIdxGrad<T1, T2><<<grid, threads, 0, context.stream()>>>(
        nthreads, output_grad_data, mask_data, input_channels, input_height,
        input_width, output_height, output_width, ksize_height, ksize_width,
        stride_height, stride_width, padding_height, padding_width, adaptive,
        input_grad_data, pool_divmods);
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
      dstart = AdaptStartIndex(pd, input_depth, output_depth);
      dend = AdaptEndIndex(pd, input_depth, output_depth);

      hstart = AdaptStartIndex(ph, input_height, output_height);
      hend = AdaptEndIndex(ph, input_height, output_height);

      wstart = AdaptStartIndex(pw, input_width, output_width);
      wend = AdaptEndIndex(pw, input_width, output_width);
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
    int w_offset = index % input_width;
    int h_offset = (index / input_width) % input_height;
    int d_offset = (index / input_width / input_height) % input_depth;
    int c_offset =
        (index / input_width / input_height / input_depth) % channels;
    int batch_idx = index / input_width / input_height / input_depth / channels;

    int pdstart, pdend;
    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      pdstart = d_offset * output_depth / input_depth;
      pdend =
          min((d_offset + 1) * output_depth / input_depth + 1, output_depth);
      phstart = h_offset * output_height / input_height;
      phend =
          min((h_offset + 1) * output_height / input_height + 1, output_height);
      pwstart = w_offset * output_width / input_width;
      pwend =
          min((w_offset + 1) * output_width / input_width + 1, output_width);
    } else {
      pdstart =
          (d_offset + padding_depth < ksize_depth)
              ? 0
              : (d_offset + padding_depth - ksize_depth) / stride_depth + 1;
      phstart =
          (h_offset + padding_height < ksize_height)
              ? 0
              : (h_offset + padding_height - ksize_height) / stride_height + 1;
      pwstart =
          (w_offset + padding_width < ksize_width)
              ? 0
              : (w_offset + padding_width - ksize_width) / stride_width + 1;
      pdend = min((d_offset + padding_depth) / stride_depth + 1, output_depth);
      phend =
          min((h_offset + padding_height) / stride_height + 1, output_height);
      pwend = min((w_offset + padding_width) / stride_width + 1, output_width);
    }

    T1 input_grad_data = 0;
    int input_current_feature_map_idx =
        (d_offset * input_height + h_offset) * input_width + w_offset;
    int output_idx = (batch_idx * channels + c_offset) * output_depth *
                     output_height * output_width;
    mask += output_idx;
    output_grad += output_idx;

    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[(pd * output_height + ph) * output_width + pw] ==
              input_current_feature_map_idx)
            input_grad_data +=
                output_grad[(pd * output_height + ph) * output_width + pw];
        }
      }
    }
    input_grad[index] = input_grad_data;
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
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &thread_num);
#endif

    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
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
