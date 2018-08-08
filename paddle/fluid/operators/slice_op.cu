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

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T, int Rank>
__global__ void CUDASliceOpGradImpl(T *dx, framework::Dim<Rank> dx_strides,
                                    const T *dy, int64_t dy_size,
                                    framework::Dim<Rank> dy_strides,
                                    int64_t idx_offset) {
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= dy_size) return;
  int64_t dx_idx = idx_offset, dy_idx = i;
  for (auto j = 0; j < Rank; ++j) {
    dx_idx += (dy_idx / dy_strides[j]) * dx_strides[j];
    dy_idx %= dy_strides[j];
  }
  dx[dx_idx] = dy[i];
}

template <typename T>
struct CUDASliceOpGradFunctor {
  template <int Rank>
  void operator()(const platform::CUDADeviceContext &ctx, T *dx,
                  int64_t dx_size, const framework::Dim<Rank> &dx_strides,
                  const T *dy, int64_t dy_size,
                  const framework::Dim<Rank> &dy_strides, int64_t idx_offset) {
    auto stream = ctx.stream();
    cudaMemsetAsync(dx, 0, sizeof(T) * dx_size, stream);
    int threads = platform::PADDLE_CUDA_NUM_THREADS;
    int grids = (dy_size + threads - 1) / threads;
    CUDASliceOpGradImpl<<<grids, threads, 0, stream>>>(
        dx, dx_strides, dy, dy_size, dy_strides, idx_offset);
  }
};

template <typename T>
using CUDASliceGradKernel =
    SliceGradKernel<platform::CUDADeviceContext, T, CUDASliceOpGradFunctor<T>>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>);

REGISTER_OP_CUDA_KERNEL(slice_grad, ops::CUDASliceGradKernel<float>,
                        ops::CUDASliceGradKernel<double>);
