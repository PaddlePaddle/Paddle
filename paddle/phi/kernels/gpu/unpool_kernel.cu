// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/unpool_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename IndT>
__global__ void KernelUnpool2dMax(const int nthreads,
                                  const T* input_data,
                                  const IndT* indices_data,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_width / input_height) % channels;
    int n = linearIndex / input_width / input_height / channels;
    output_data += (n * channels + c) * output_height * output_width;
    IndT maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
  }
}

template <typename T, typename IndT>
__global__ void KernelUnpool3dMax(const int nthreads,
                                  const T* input_data,
                                  const IndT* indices_data,
                                  const int input_depth,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_depth,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_depth / input_width / input_height) % channels;
    int n = linearIndex / input_depth / input_width / input_height / channels;
    output_data +=
        (n * channels + c) * output_depth * output_height * output_width;
    IndT maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
  }
}

template <typename T, typename IndT, typename Context>
class Unpool2dMaxFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  DenseTensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const T* input_data = input.data<T>();
    const IndT* indices_data = indices.data<IndT>();
    T* output_data = dev_ctx.template Alloc<T>(output);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool2dMax<T, IndT>
        <<<grid, threads, 0, dev_ctx.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_height,
                                                 output_width);
  }
};

template <typename T, typename IndT, typename Context>
class Unpool3dMaxFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  DenseTensor* output) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const T* input_data = input.data<T>();
    const IndT* indices_data = indices.data<IndT>();
    T* output_data = dev_ctx.template Alloc<T>(output);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool3dMax<T, IndT>
        <<<grid, threads, 0, dev_ctx.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_depth,
                                                 output_height,
                                                 output_width);
  }
};

template <typename T, typename Context>
void UnpoolKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& indices,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const IntArray& output_size,
                  const std::string& data_format,
                  DenseTensor* out) {
  T* output_data = dev_ctx.template Alloc<T>(out);
  if (output_data) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out, static_cast<T>(0));
  }

  const auto& indices_type = indices.dtype();
  if (indices_type == phi::DataType::INT32) {
    Unpool2dMaxFunctor<T, int, Context> unpool2d_max_forward;
    unpool2d_max_forward(dev_ctx, x, indices, out);
  } else {
    Unpool2dMaxFunctor<T, int64_t, Context> unpool2d_max_forward;
    unpool2d_max_forward(dev_ctx, x, indices, out);
  }
}

template <typename T, typename Context>
void Unpool3dKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    const std::vector<int>& output_size,
                    const std::string& data_format,
                    DenseTensor* out) {
  T* output_data = dev_ctx.template Alloc<T>(out);
  if (output_data) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out, static_cast<T>(0));
  }

  const auto& indices_type = indices.dtype();
  if (indices_type == phi::DataType::INT32) {
    Unpool3dMaxFunctor<T, int, Context> unpool3d_max_forward;
    unpool3d_max_forward(dev_ctx, x, indices, out);
  } else {
    Unpool3dMaxFunctor<T, int64_t, Context> unpool3d_max_forward;
    unpool3d_max_forward(dev_ctx, x, indices, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unpool, GPU, ALL_LAYOUT, phi::UnpoolKernel, int, float, double, int64_t) {}

PD_REGISTER_KERNEL(unpool3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::Unpool3dKernel,
                   int,
                   float,
                   double,
                   int64_t) {}
