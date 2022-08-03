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

#pragma once
#include <thrust/device_vector.h>

#include <algorithm>
#include <vector>

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/impl/graph_send_ue_recv_kernel_impl.h"

namespace phi {

#define CUDA_MAX_NUM_THREADS 1024

inline void CopyBCastOff(const BroadCastInfo& bcast_info,
                         thrust::device_vector<int64_t>& l_bcastoff,
                         thrust::device_vector<int64_t>& r_bcastoff) {
  l_bcastoff.resize(bcast_info.out_len);
  r_bcastoff.resize(bcast_info.out_len);
#ifdef PADDLE_WITH_HIP
  hipMemcpy(thrust::raw_pointer_cast(l_bcastoff.data()),
            bcast_info.l_offset.data(),
            sizeof(int64_t) * bcast_info.out_len,
            hipMemcpyHostToDevice);
  hipMemcpy(thrust::raw_pointer_cast(r_bcastoff.data()),
            bcast_info.r_offset.data(),
            sizeof(int64_t) * bcast_info.out_len,
            hipMemcpyHostToDevice);
#else
  cudaMemcpy(thrust::raw_pointer_cast(l_bcastoff.data()),
             bcast_info.l_offset.data(),
             sizeof(int64_t) * bcast_info.out_len,
             cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(r_bcastoff.data()),
             bcast_info.r_offset.data(),
             sizeof(int64_t) * bcast_info.out_len,
             cudaMemcpyHostToDevice);
#endif
}

inline int FindNumThreads(int dim, int max_num_threads = CUDA_MAX_NUM_THREADS) {
  PADDLE_ENFORCE_GE(dim,
                    0,
                    phi::errors::PreconditionNotMet(
                        "Required dim >= 0, but received dim = %d", dim));
  if (dim == 0) return 1;
  int res = max_num_threads;
  while (res > dim) {
    res = res >> 1;
  }
  return res;
}

template <typename T>
struct GraphSendUERecvSumCUDAFunctor {
  DEVICE inline void operator()(T* output, T val) {
    paddle::platform::CudaAtomicAdd(output, val);
  }
};

template <typename T>
struct GraphSendUERecvMaxCUDAFunctor {
  DEVICE inline void operator()(T* output, T val) {
    paddle::platform::CudaAtomicMax(output, val);
  }
};

template <typename T>
struct GraphSendUERecvMinCUDAFunctor {
  DEVICE inline void operator()(T* output, T val) {
    paddle::platform::CudaAtomicMin(output, val);
  }
};

template <typename T,
          typename IndexT,
          typename ReduceFunctor,
          typename ComputeFunctor>
__global__ void GraphSendUERecvCUDAKernel(const T* x_data,
                                          const T* e_data,
                                          const IndexT* src_indices,
                                          const IndexT* dst_indices,
                                          const int64_t* xbcast_off,
                                          const int64_t* ebcast_off,
                                          T* output,
                                          int64_t index_size,
                                          int64_t x_len,
                                          int64_t e_len,
                                          int64_t out_len,
                                          bool use_bcast,
                                          ComputeFunctor cfunctor,
                                          ReduceFunctor rfunctor) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + src * x_len;
    const T* e_off = e_data + ty * e_len;
    T* out_off = output + dst * out_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? xbcast_off[tx] : tx;
      int64_t e_add = use_bcast ? ebcast_off[tx] : tx;
      T val = cfunctor(x_off[x_add], e_off[e_add]);
      rfunctor(out_off + tx, val);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

}  // namespace phi
