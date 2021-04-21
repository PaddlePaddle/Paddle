// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/fluid/operators/elementwise/elementwise_op_tmp.cu.h"

#define DEBUG 0

namespace paddle {
namespace operators {

template <typename T>
int GetVectorizedSizeImpl(T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<CudaAlignedVector<T, 4>>::value;
  constexpr int vec2 = std::alignment_of<CudaAlignedVector<T, 2>>::value;
  if (sizeof(T) <= 16) {
    constexpr int vec8 = std::alignment_of<CudaAlignedVector<T, 8>>::value;
    if (address % vec8 == 0) return 8;
  }
  if (address % vec4 == 0) return 4;
  if (address % vec2 == 0) return 2;
  return 1;
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename ArgT, typename loader_t, int N, int nDims>
inline __device__ void ScalarizeKernelImpl(const loader_t &in_loaders, T *out,
                                           int tid, int remain) {
  T args[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    (in_loaders.s_loader[i])(in_loaders.data[i], &args[i], tid, remain);
  }
  // (*out) = args[0] + args[1];
}

template <typename T, typename ArgT, typename loader_t, int N, int nDims,
          int vec_size>
inline __device__ void VectorizeKernelImpl(const loader_t &in_loaders, T *out,
                                           int tid, int loop) {
  ArgT args[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    (in_loaders.v_loader[i])(in_loaders.data[i], &args[i], tid, loop);
  }
}

template <typename T, typename loader_t, int N, int vec_size, int nDims>
__global__ void CommonElementwiseKernel(const loader_t &in_loaders, T *out,
                                        int loop, int remain) {
  using ArgT = CudaAlignedVector<T, vec_size>;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  VectorizeKernelImpl<T, ArgT, loader_t, N, nDims, vec_size>(in_loaders, out,
                                                             tid, loop);
  if (remain) {
    ScalarizeKernelImpl<T, T, loader_t, N, nDims>(in_loaders, out, tid, remain);
  }
}
#endif  // (__NVCC__) || (__HIPCC__)

template <typename T, typename OffsetT, int vec_size, int N>
void CommonElementwiseCore(const framework::ExecutionContext &ctx,
                           std::vector<const framework::Tensor *> *ins,
                           framework::Tensor *out, const OffsetT &offset_pre) {
  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    int numel = out->numel();
    int loop = numel / vec_size;
    int remain = numel - loop * vec_size;
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    int dim_size = offset_pre.strides.size();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    T *out_data = out->mutable_data<T>(dev_ctx.GetPlace());

    switch (dim_size) {
      case 2: {
        const auto loader =
            TensorLoader<T, OffsetT, N, vec_size, 2>(ins, out, offset_pre);
        CommonElementwiseKernel<T, decltype(loader), N, vec_size,
                                2><<<blocks, threads, 0, stream>>>(
            loader, out_data, loop, remain);
        break;
      }
      case 3: {
        const auto loader =
            TensorLoader<T, OffsetT, N, vec_size, 3>(ins, out, offset_pre);
        CommonElementwiseKernel<T, decltype(loader), N, vec_size,
                                3><<<blocks, threads, 0, stream>>>(
            loader, out_data, loop, remain);
        break;
      }
      case 4: {
        const auto loader =
            TensorLoader<T, OffsetT, N, vec_size, 4>(ins, out, offset_pre);
        CommonElementwiseKernel<T, decltype(loader), N, vec_size,
                                4><<<blocks, threads, 0, stream>>>(
            loader, out_data, loop, remain);
        break;
      }
      case 5: {
        const auto loader =
            TensorLoader<T, OffsetT, N, vec_size, 5>(ins, out, offset_pre);
        CommonElementwiseKernel<T, decltype(loader), N, vec_size,
                                5><<<blocks, threads, 0, stream>>>(
            loader, out_data, loop, remain);
        break;
      }
      default: { ; }
    }
#endif  // (__NVCC__) || (__HIPCC__)
  }
}

template <typename T, int vec_size = 1>
void BroadcastDimsTransform(const framework::ExecutionContext &ctx,
                            std::vector<const framework::Tensor *> *ins,
                            framework::Tensor *out) {
  int input_num = ins->size();
  const auto merged_dims = DimensionTransform(ins, out->dims(), input_num);
  const auto offset_pre =
      OffsetPreCalculator<decltype(merged_dims)>(merged_dims);

  switch (input_num) {
    case 2: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 2>(ctx, ins, out,
                                                                  offset_pre);
      break;
    }
    case 3: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 3>(ctx, ins, out,
                                                                  offset_pre);
      break;
    }
    case 4: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 4>(ctx, ins, out,
                                                                  offset_pre);
      break;
    }
    default: { ; }
  }
}

template <typename DeviceContext, typename T>
void BroadcastElementwise(const framework::ExecutionContext &ctx,
                          std::vector<const framework::Tensor *> *ins,
                          framework::Tensor *out) {
  int in_vec_size = 8;
  T *out_data = out->mutable_data<T>(ctx.GetPlace());
  int out_vec_size = GetVectorizedSizeImpl<T>(out_data);

  for (auto *in : *ins) {
    auto temp_size = GetVectorizedSizeImpl(in->data<T>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 8: {
      BroadcastDimsTransform<T, 8>(ctx, ins, out);
      break;
    }
    case 4: {
      BroadcastDimsTransform<T, 4>(ctx, ins, out);
      break;
    }
    case 2: {
      BroadcastDimsTransform<T, 2>(ctx, ins, out);
      break;
    }
    default: {
      BroadcastDimsTransform<T>(ctx, ins, out);
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
