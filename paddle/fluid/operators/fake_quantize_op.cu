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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void FindAbsMaxKernel(const T* in, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ T shared_max_data[];
  if (gridDim.x > 1) {
    shared_max_data[tid] = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = fabs(in[i]);
      if (tmp > shared_max_data[tid]) {
        shared_max_data[tid] = tmp;
      }
    }
  } else {
    if (bid < n) {
      shared_max_data[tid] = fabs(in[bid]);
    } else {
      shared_max_data[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
struct FindAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const T* in,
                  const int num, T* out) {
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;

    framework::Tensor max;
    T* max_data =
        max.mutable_data<T>(framework::make_ddim({grid}), ctx.GetPlace());
    FindAbsMaxKernel<T><<<grid, block, 1024 * sizeof(T), ctx.stream()>>>(
        in, num, max_data);
    FindAbsMaxKernel<T><<<1, block, 1024 * sizeof(T), ctx.stream()>>>(
        max_data, grid, out);
  }
};

template struct FindAbsMaxFunctor<platform::CUDADeviceContext, float>;

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
__global__ void FindRangeAbsMaxAndFillArray(const T* cur_scale,
                                            const T* last_scale,
                                            const int64_t* iter,
                                            const int window_size, T* scale_arr,
                                            T* out_scale, int* need_find_max,
                                            int* out_size) {
  int it = iter[0];
  int idx = it % window_size;
  T removed = scale_arr[idx];
  T cur = cur_scale[0];
  scale_arr[idx] = cur;
  T max = last_scale[0];
  out_scale[0] = max < cur ? cur : max;
  if (fabs(removed - max) < 1e-6) {
    need_find_max[0] = 1;
    out_size[0] = it > window_size ? window_size : it;
  } else {
    need_find_max[0] = 0;
  }
}

template <typename T>
struct FindRangeAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale) {
    const auto gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());

    T* scale_arr = scales_arr->mutable_data<T>(gpu_place);
    T* out_scale_data = out_scale->mutable_data<T>(gpu_place);

    framework::Tensor need_find_max, out_size;
    int* find_max = need_find_max.mutable_data<int>(gpu_place);
    int* out_size_data = out_size.mutable_data<int>(gpu_place);

    FindRangeAbsMaxAndFillArray<T><<<1, 1, 0, ctx.stream()>>>(
        cur_scale.data<T>(), last_scale.data<T>(), iter.data<int64_t>(),
        window_size, scale_arr, out_scale_data, find_max, out_size_data);

    int g_find_max;
    memory::Copy(platform::CPUPlace(), &g_find_max, gpu_place, find_max,
                 sizeof(int), 0);
    if (g_find_max) {
      int len;
      memory::Copy(platform::CPUPlace(), &len, gpu_place, out_size_data,
                   sizeof(int), 0);
      FindAbsMaxFunctor<platform::CUDADeviceContext, T>()(ctx, scale_arr, len,
                                                          out_scale_data);
    }
  }
};

template struct FindRangeAbsMaxFunctor<platform::CUDADeviceContext, float>;

template <typename T>
struct FindMovingAverageAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in_accum,
                  const framework::Tensor& in_state, const T* cur_scale,
                  const float rate, framework::Tensor* out_state,
                  framework::Tensor* out_accum, framework::Tensor* out_scale) {
    const auto gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());

    T accum;
    memory::Copy(platform::CPUPlace(), &accum, gpu_place, in_accum.data<T>(),
                 sizeof(T), 0);
    T state;
    memory::Copy(platform::CPUPlace(), &state, gpu_place, in_state.data<T>(),
                 sizeof(T), 0);
    T scale;
    memory::Copy(platform::CPUPlace(), &scale, gpu_place, cur_scale, sizeof(T),
                 0);

    state = rate * state + 1;
    accum = rate * accum + scale;
    scale = accum / state;

    memory::Copy(gpu_place, out_accum->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &accum, sizeof(T), 0);
    memory::Copy(gpu_place, out_state->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &state, sizeof(T), 0);
    memory::Copy(gpu_place, out_scale->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &scale, sizeof(T), 0);
  }
};

template struct FindMovingAverageAbsMaxFunctor<platform::CUDADeviceContext,
                                               float>;

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    int num = in.numel();
    int block = 1024;
    int grid = (block - 1 + num) / block;

    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    ClipAndQuantKernel<T><<<grid, block, 0, ctx.stream()>>>(
        in_data, scale_data, bin_cnt, num, out_data);
  }
};

template struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(fake_quantize_abs_max,
                        ops::FakeQuantizeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(fake_channel_wise_quantize_abs_max,
                        ops::FakeChannelWiseQuantizeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(fake_quantize_range_abs_max,
                        ops::FakeQuantizeRangeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(
    fake_quantize_moving_average_abs_max,
    ops::FakeQuantizeMovingAverageAbsMaxKernel<CUDA, float>);
