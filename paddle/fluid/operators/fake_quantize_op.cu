/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void FindAbsMaxKernel(const T* in, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ T shared_max[];
  if (gridDim.x > 1) {
    shared_max[tid] = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = fabs(in[i]);
      if (tmp > shared_max[tid]) {
        shared_max[tid] = tmp;
      }
    }
  } else {
    if (bid < n) {
      shared_max[tid] = fabs(in[bid]);
    } else {
      shared_max[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max[tid] < shared_max[tid + i])) {
      shared_max[tid] = shared_max[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max[0];
  }
}

template <typename T>
struct FindAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const CUDADeviceContext& ctx, const T* in, const int num,
                  T* out) {
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;

    Tensor max;
    T* max_data =
        max.mutable_data<T>(framework::make_ddim({grid}), ctx.GetPlace());
    FindAbsMaxKernel<T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
        in_data, num, max_data);
    FindAbsMaxKernel<T><<<1, block, block * sizeof(T), ctx.stream()>>>(
        max_data, grid, out);
  }
};

template <typename T>
__global__ void ClipAndQuantKernel(const T* in, const T* scale,
                                   const int bin_cnt, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  T s = scale[0];
  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    T x = in[bid];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt / s * v;
    out[bid] = round(v);
  }
}

template <typename T>
__global__ void FillScaleArray(T* scale_arr, T* out_scale, const int* it,
                               const int window_size, ) {
  int tid = threadIdx.x;
  int idx = it % window_size;
  // scale_arr[idx] = ;
}

template <typename T>
struct FindRangeAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale) {
    T* scale_arr = scales_arr->mutable_data<T>(cxt.GetPlace());
    auto& gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    int it;
    memory::Copy(platform::CPUPlace(), &it, gpu_place, iter.data<int>(),
                 sizeof(int), ctx.stream());
    int idx = current_iter % window_size;
    T removed;
    memory::Copy(platform::CPUPlace(), &removed, gpu_place, scale_arr + idx,
                 sizeof(T), ctx.stream());
    T cur;
    memory::Copy(gpu_place, &cur, gpu_place, cur_scale.data<T>(), sizeof(T),
                 ctx.stream());

    T max;
    memory::Copy(platform::CPUPlace(), &max, gpu_place, last_scale.data<T>(),
                 sizeof(T), ctx.stream());
    T* out_scale_data = out_scale->mutable_data<T>(gpu_place);
    if (max < cur) {
      max = cur;
      memory::Copy(gpu_place, out_scale_data, gpu_place, &max, sizeof(T),
                   ctx.stream());
    } else if (fabs(removed - max) < 1e-6) {
      int size = (it > window_size) ? window_size : it;
      FindAbsMaxFunctor<platform::CPUDeviceContext, T>()(ctx, scale_arr, size,
                                                         out_scale_data);
    }
  }
};

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const CPUDeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor* scale, const int bin_cnt,
                  framework::Tensor* out) {
    int num = in.numel();
    int block = 1024;
    int grid = (block - 1 + num) / block;

    T* in_data = in.data<T>();
    T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    ClipAndQuantKernel<T><<<grid, block, 0, ctx.stream()>>>(
        in_data, scale_data, bin_cnt, num, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(fake_quantize,
                        paddle::operators::FakeQuantizeCUDAKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        paddle::operators::FakeQuantizeCUDAKernel<
                            paddle::platform::CUDADeviceContext, double>);
