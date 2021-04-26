// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.1
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast_impl.cu.h"

namespace paddle {
namespace operators {

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename fetch_t, int N>
__device__ inline void BScalarizedKernelImpl(fetch_t data_fetch, int tid) {
  T args[N];
  data_fetch.load_scalar(args, tid);

#pragma unroll(N)
  for (int j = 1; j < N; ++j) {
    args[0] += args[j];
  }

  data_fetch.store_scalar(args, tid);
}

template <typename T, typename fetch_t, int N, int vec_size>
__device__ inline void BVectorizedKernelImpl(fetch_t data_fetch, int tid) {
  using ArgT = CudaAlignedVector<T, vec_size>;
  ArgT args[N];
  data_fetch.load_vector(args, tid);

#pragma unroll(N)
  for (int j = 1; j < N; ++j) {
#pragma unroll(vec_size)
    for (int i = 0; i < vec_size; ++i) {
      args[0].val[i] += args[j].val[i];
    }
  }
  data_fetch.store_vector(args, tid);
}

template <typename T, typename fetch_t, int N, int vec_size>
__global__ void CommonElementwiseKernel(fetch_t data_fetch, int main_tid,
                                        int tail_tid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < main_tid) {
    BVectorizedKernelImpl<T, fetch_t, N, vec_size>(data_fetch, tid);
  }
  if (tid < tail_tid) {
    BScalarizedKernelImpl<T, fetch_t, N>(data_fetch, tid);
  }
}
#endif

template <typename T, typename OffsetT, int vec_size, int N>
void CommonElementwiseCore(const platform::CUDADeviceContext &ctx,
                           const std::vector<const framework::Tensor *> &ins,
                           framework::Tensor *out,
                           const OffsetT &p_offset_pre) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + vec_size - 1) / vec_size + threads - 1) / threads;
  int main_tid = numel / vec_size;
  int tail_tid = numel % vec_size;
  int vec_len = main_tid * vec_size;

#if defined(__NVCC__) || defined(__HIPCC__)
  int dim_size = p_offset_pre.strides[0].size();
  auto stream = ctx.stream();
  T *out_data = out->data<T>();

#if DEBUG_OPT
  std::cout << "threads" << threads << threads;
  std::cout << "blocks " << blocks << threads;
  std::cout << "numel : " << numel << std::endl;
  std::cout << "vec_len : " << vec_len << std::endl;
  std::cout << "dim_size : " << dim_size << std::endl;
  std::cout << "main_tid : " << main_tid << std::endl;
  std::cout << "tail_tid : " << tail_tid << std::endl;
#endif

  switch (dim_size) {
    case 2: {
      auto data_fetch = DataFetch<T, OffsetT, N, vec_size, 2>(
          ins, p_offset_pre, out_data, vec_len);
      CommonElementwiseKernel<T, decltype(data_fetch), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_fetch, main_tid, tail_tid);
      break;
    }
    case 3: {
      auto data_fetch = DataFetch<T, OffsetT, N, vec_size, 3>(
          ins, p_offset_pre, out_data, vec_len);
      CommonElementwiseKernel<T, decltype(data_fetch), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_fetch, main_tid, tail_tid);
      break;
    }
    case 4: {
      auto data_fetch = DataFetch<T, OffsetT, N, vec_size, 4>(
          ins, p_offset_pre, out_data, vec_len);
      CommonElementwiseKernel<T, decltype(data_fetch), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_fetch, main_tid, tail_tid);
      break;
    }
    case 5: {
      auto data_fetch = DataFetch<T, OffsetT, N, vec_size, 5>(
          ins, p_offset_pre, out_data, vec_len);
      CommonElementwiseKernel<T, decltype(data_fetch), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_fetch, main_tid, tail_tid);
      break;
    }
    case 6: {
      auto data_fetch = DataFetch<T, OffsetT, N, vec_size, 6>(
          ins, p_offset_pre, out_data, vec_len);
      CommonElementwiseKernel<T, decltype(data_fetch), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_fetch, main_tid, tail_tid);
      break;
    }
    default: { ; }
  }
#endif
}

template <typename T, int vec_size = 1>
void BroadcastDimsTransform(const platform::CUDADeviceContext &ctx,
                            const std::vector<const framework::Tensor *> &ins,
                            framework::Tensor *out) {
  int input_num = ins.size();
  const auto merge_dims = DimensionTransform(ins, out->dims(), input_num);
  const auto offset_pre = OffsetPreCalculator<decltype(merge_dims)>(merge_dims);
#if defined(__NVCC__) || defined(__HIPCC__)
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
#endif
}

template <typename DeviceContext, typename T>
void LaunchBroadElementwiseCudaKernel(const framework::ExecutionContext &ctx,
                                      const framework::Tensor *x,
                                      const framework::Tensor *y,
                                      framework::Tensor *out) {
  int in_vec_size = 8;
  const std::vector<const framework::Tensor *> ins = {x, y};
  for (auto *in : ins) {
    auto temp_size = GetVectorizedSizeImpl<T>(in->data<T>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  T *out_data = out->data<T>();
  int out_vec_size = GetVectorizedSizeImpl<T>(out_data);
  int vec_size = std::min(out_vec_size, in_vec_size);
  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

  switch (vec_size) {
    case 4: {
      BroadcastDimsTransform<T, 4>(dev_ctx, ins, out);
      break;
    }
    case 2: {
      BroadcastDimsTransform<T, 2>(dev_ctx, ins, out);
      break;
    }
    default: {
      BroadcastDimsTransform<T>(dev_ctx, ins, out);
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
