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

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_tmp.cu.h"

namespace paddle {
namespace operators {

template <typename T>
int GetVectorizedSizeImpl(T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<CudaAlignedVector<T, 4>>::value;
  constexpr int vec2 = std::alignment_of<CudaAlignedVector<T, 2>>::value;
  // if (sizeof(T) <= 16) {
  //   constexpr int vec8 =
  //       std::alignment_of<CudaAlignedVector<T, 8>>::value;
  //   if (address % vec8 == 0) return 8;
  // }
  if (address % vec4 == 0) return 4;
  if (address % vec2 == 0) return 2;
  return 1;
}

// template <typename T, typename ArgT, typename loader_t, int N, int nDims,
//           int vec_size>
// inline __device__ void VectorizeKernelImpl(loader_t *in_loaders,
//                           ArgT *out_data, int tid, int loop) {
//   ArgT arg[N];
// #pragma unroll
//   for (int i = 0; i < N; ++i) {
//     in_loaders->(v_loader[i])<T, ArgT, vec_size>(in_loaders->in_data[i], arg,
//                                                  tid, loop);
//   }
// #pragma unroll
//   for (int i = 0; i < N; ++i) {
//     out_data += arg[i]
//   }
// }

// template <typename T, typename loader_t, int N, int nDims>
// inline __device__ void ScalarizeKernelImpl(loader_t *in_loaders,
//                             T * out_data, int tid int remain) {
//   T arg[N];

// #pragma unroll
//   for (int i = 0; i < N; ++i) {
//     in_loaders->(s_loader[i])<T, T, 1>(in_loaders->tail_data[i], arg, tid,
//                                        remain);
//   }
// #pragma unroll
//   for (int i = 0; i < N; ++i) {
//     out_data += arg[i]
//   }
// }

// template <typename T, typename loader_t, int N, int vec_size, int nDims>
// __global__ void CommonElementwiseKernel(loader_t *in_loaders, T *out_data,
//                                         int loop, int remain) {
//   using ArgT = CudaAlignedVector<T, vec_size>;
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   VectorizeKernelImpl<T, ArgT, loader_t, N, nDims, vec_siz>(in_loaders, out,
//                                                             tid, loop);
//   if (remain) {
//     ScalarizeKernelImpl<T, T, loader_t, N, nDims>(in_loaders, out, tid,
//     remain);
//   }
// }

template <typename T, typename OffsetT, int vec_size, int N>
void CommonElementwiseCore(const framework::ExecutionContext &ctx,
                           std::vector<const framework::Tensor *> *ins,
                           framework::Tensor *out, const OffsetT *offset_pre) {
  auto numel = out->numel();
  constexpr int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  int loop = numel / vec_size;
  int remain = numel - loop * vec_size;

  auto dim_size = offset_pre->strides.size();
  auto loader = TensorLoader<T, OffsetT, N, vec_size, 1>(ins, out, offset_pre);
  // T *out_data = out->mutable_data<T>(ctx.GetPlace());
  using loader_t = decltype(loader);

  std::cout << numel << std::endl;
  std::cout << threads << std::endl;
  std::cout << blocks << std::endl;
  std::cout << loop << std::endl;
  std::cout << remain << std::endl;
  std::cout << dim_size << std::endl;

  // auto stream =
  //     ctx.template device_context<platform::CUDADeviceContext>().stream();

  // switch (dim_size) {
  //   case 2: {
  //     loader = TensorLoader<T, OffsetT, N, vec_size, 2>(ins, out,
  //     offset_pre);
  //     CommonElementwiseKernel<T, loader_t, N, vec_size,
  //                             2><<<blocks, threads, 0, stream>>>(
  //         &loader, out_data, loop, remain);
  //     break;
  //   }
  // }
  // case 3: {
  //   loader =
  //       TensorLoader<T, OffsetT, N, vec_size, 3>(ins, out, offset_pre);
  //   CommonElementwiseKernel<T, loader_t, N, vec_size,
  //                           3><<<blocks, threads, 0, stream>>>(
  //       &loader, out_data, loop, remain);
  //   break;
  // }
  // case 4: {
  //   loader =
  //       TensorLoader<T, OffsetT, N, vec_size, 4>(ins, out, offset_pre);
  //   CommonElementwiseKernel<T, loader_t, N, vec_size,
  //                           4><<<blocks, threads, 0, stream>>>(
  //       &loader, out_data, loop, remain);
  //   break;
  // }
  // case 5: {
  //   loader =
  //       TensorLoader<T, OffsetT, N, vec_size, 5>(ins, out, offset_pre);
  //   CommonElementwiseKernel<T, loader_t, N, vec_size,
  //                           5><<<blocks, threads, 0, stream>>>(
  //       &loader, out_data, loop, remain);
  //   break;
  // }
  // default: {
  //   CommonElementwiseKernel<T, loader_t, N,
  //                           1><<<blocks, threads, 0, stream>>>(
  //       &loader, out_data, loop, remain);
  //   break;
  // }
  //   }
}

template <typename T, int vec_size = 1>
void BroadcastDimsTransform(const framework::ExecutionContext &ctx,
                            std::vector<const framework::Tensor *> *ins,
                            framework::Tensor *out) {
  auto input_num = ins->size();
  auto merged_dims = DimensionTransform(ins, out->dims(), input_num);
  auto offset_pre = OffsetPreCalculator<decltype(merged_dims)>(&merged_dims);

  switch (input_num) {
    case 2: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 2>(ctx, ins, out,
                                                                  &offset_pre);
      break;
    }
    case 3: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 3>(ctx, ins, out,
                                                                  &offset_pre);
      break;
    }
    case 4: {
      CommonElementwiseCore<T, decltype(offset_pre), vec_size, 4>(ctx, ins, out,
                                                                  &offset_pre);
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
    // case 8: {
    //   BroadcastDimsTransform<T, 8>(ctx, ins, out);
    //   break;
    // }
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
