/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

using float16 = plat::float16;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <size_t D>
using EigenIndexArray = Eigen::array<Eigen::Index, D>;

template <size_t D>
using EigenPairArray = Eigen::array<std::pair<int64_t, int64_t>, D>;

template <typename T, size_t D>
__global__ void KePaddingEigen(const T *input_data,
                               const EigenIndexArray<D> in_stride_array,
                               const int64_t in_size,
                               const EigenPairArray<D> paddings, T *output_data,
                               const EigenIndexArray<D> out_stride_array) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = tid; index < in_size; index += blockDim.x * gridDim.x) {
    // compute each dimension index of input
    int dims[D]{0};
    int64_t k = index;
#pragma unroll
    for (int i = 0; i < D; i++) {
      dims[i] = k / in_stride_array[i];
      k -= dims[i] * in_stride_array[i];
    }

    // compute the total index of output
    int64_t out_id = 0;
#pragma unroll
    for (int i = 0; i < D; i++) {
      out_id += (dims[i] + paddings[i].first) * out_stride_array[i];
    }
    output_data[out_id] = input_data[index];
  }
}

template <typename T, size_t D>
void LaunchSlicePaddingKernel(const gpuStream_t &stream, const T *input_data,
                              const EigenIndexArray<D> &in_stride_array,
                              const int64_t in_size,
                              const EigenPairArray<D> &paddings, T *output_data,
                              const EigenIndexArray<D> &out_stride_array,
                              const int64_t out_size) {
  // Padding zero for output
  platform::GpuMemsetAsync(output_data, 0, out_size * sizeof(T), stream);

  int threads = 256;
  int block_num = threads * 10;
  int grids = (in_size + block_num - 1) / block_num;
  KePaddingEigen<T, D><<<grids, threads, 0, stream>>>(
      input_data, in_stride_array, in_size, paddings, output_data,
      out_stride_array);
}

template <typename T, size_t D>
void LaunchSlicePaddingFunction(
    const framework::ExecutionContext &context, framework::Tensor *d_input,
    const framework::DDim &in_dims, const framework::Tensor *d_out,
    const framework::DDim &out_dims,
    const Eigen::array<std::pair<int64_t, int64_t>, D> &paddings) {
  auto &dev_ctx = context.template device_context<CUDADeviceContext>();

  int64_t in_size = 1, out_size = 1;
#pragma unroll
  for (int i = 0; i < D; i++) {
    in_size *= in_dims[i];
    out_size *= out_dims[i];
  }

  Eigen::array<Eigen::Index, D> in_stride_array, out_stride_array;
  in_stride_array[D - 1] = out_stride_array[D - 1] = 1;
  for (int i = D - 2; i >= 0; i--) {
    in_stride_array[i] = in_stride_array[i + 1] * in_dims[i + 1];
    out_stride_array[i] = out_stride_array[i + 1] * out_dims[i + 1];
  }

  LaunchSlicePaddingKernel<T, D>(
      dev_ctx.stream(), d_out->data<T>(), out_stride_array, out_size, paddings,
      d_input->mutable_data<T>(context.GetPlace()), in_stride_array, in_size);
}

template <>
template <size_t D>
void SliceGradKernel<CUDADeviceContext, float16>::SliceComputeFunction(
    const framework::ExecutionContext &context, framework::Tensor *d_input,
    const framework::DDim &in_dims, const framework::Tensor *d_out,
    const framework::DDim &out_dims,
    const Eigen::array<std::pair<int64_t, int64_t>, D> &paddings) const {
  int64_t pad_size = 0;
#pragma unroll
  for (int i = 0; i < D; i++) {
    pad_size += paddings[i].first + paddings[i].second;
  }
  if (pad_size >= 100) {
    LaunchSlicePaddingFunction<float16, D>(context, d_input, in_dims, d_out,
                                           out_dims, paddings);
  } else {
    LaunchEigenFunction<D>(context, d_input, in_dims, d_out, out_dims,
                           paddings);
  }
}
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex128>);

REGISTER_OP_CUDA_KERNEL(
    slice_grad,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext,
                         plat::complex128>);
