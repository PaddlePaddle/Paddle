// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright 2022 The DGL team for some useful functions.

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
#include <thrust/fill.h>

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"

namespace phi {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF

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

inline int FindNumThreads(int dim, int max_num_threads) {
  PADDLE_ENFORCE_GE(dim,
                    0,
                    phi::errors::PreconditionNotMet(
                        "Required dim >= 0, but received dim = %d", dim));
  int res = max_num_threads;
  if (dim == 0) res = 1;
  while (res > dim) {
    res = res >> 1;
  }
  res = res <= 32 ? 32 : res;
  return res;
}

inline int FindNumBlocks(char axis, int nblocks, int max_num_blocks = -1) {
  int default_max_num_blocks = -1;
  switch (axis) {
    case 'x':
      default_max_num_blocks = CUDA_MAX_NUM_BLOCKS_X;
      break;
    case 'y':
      default_max_num_blocks = CUDA_MAX_NUM_BLOCKS_Y;
      break;
    case 'z':
      default_max_num_blocks = CUDA_MAX_NUM_BLOCKS_Z;
      break;
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("%c axis is not recognized", axis));
  }
  if (max_num_blocks == -1) {
    max_num_blocks = default_max_num_blocks;
  }
  PADDLE_ENFORCE_GT(
      max_num_blocks,
      0,
      phi::errors::InvalidArgument("max_num_blocks should be larger than 0, "
                                   "but received %d",
                                   max_num_blocks));
  if (nblocks < max_num_blocks) {
    return nblocks;
  }
  return max_num_blocks;
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

// x_grad: for backward mean with mul.
template <typename T, typename IndexT>
__global__ void ManipulateMeanGradCUDAKernelForMulX(const T* out_grad_data,
                                                    const T* e_data,
                                                    const IndexT* src_indices,
                                                    const IndexT* dst_indices,
                                                    const int* dst_count,
                                                    const int64_t* l_bcastoff,
                                                    const int64_t* r_bcastoff,
                                                    T* x_grad,
                                                    int64_t index_size,
                                                    int64_t l_len,
                                                    int64_t r_len,
                                                    int64_t out_len,
                                                    bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* out_grad_off = out_grad_data + src * l_len;
    const T* e_off = e_data + ty * r_len;
    T* x_grad_off = x_grad + dst * out_len;
    while (tx < out_len) {
      int64_t o_add = use_bcast ? l_bcastoff[tx] : tx;
      int64_t e_add = use_bcast ? r_bcastoff[tx] : tx;
      T val = out_grad_off[o_add] * e_off[e_add];
      paddle::platform::CudaAtomicAdd(x_grad_off + tx,
                                      val / static_cast<T>(dst_count[src]));
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// e_grad: backward sum for add.
template <typename T, typename IndexT>
__global__ void ManipulateSumGradCUDAKernelForAddE(const T* out_grad_data,
                                                   const IndexT* dst_indices,
                                                   const int64_t* r_bcastoff,
                                                   T* e_grad,
                                                   int64_t index_size,
                                                   int64_t r_len,
                                                   int64_t out_len,
                                                   bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    T* e_grad_off = e_grad + ty * r_len;
    const T* out_grad_off = out_grad_data + dst * out_len;
    while (tx < out_len) {
      int64_t e_add = use_bcast ? r_bcastoff[tx] : tx;
      paddle::platform::CudaAtomicAdd(e_grad_off + e_add, out_grad_off[tx]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// e_grad: backward sum for mul.
template <typename T, typename IndexT>
__global__ void ManipulateSumGradCUDAKernelForMulE(const T* x_data,
                                                   const T* out_grad_data,
                                                   const IndexT* src_indices,
                                                   const IndexT* dst_indices,
                                                   const int64_t* l_bcastoff,
                                                   const int64_t* r_bcastoff,
                                                   T* e_grad,
                                                   int64_t index_size,
                                                   int64_t l_len,
                                                   int64_t r_len,
                                                   int64_t out_len,
                                                   bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + src * l_len;
    T* e_grad_off = e_grad + ty * r_len;
    const T* out_grad_off = out_grad_data + dst * out_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? l_bcastoff[tx] : tx;
      int64_t e_add = use_bcast ? r_bcastoff[tx] : tx;
      paddle::platform::CudaAtomicAdd(e_grad_off + e_add,
                                      out_grad_off[tx] * x_off[x_add]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// e_grad: backward mean for add
template <typename T, typename IndexT>
__global__ void ManipulateMeanGradCUDAKernelForAddE(const T* out_grad_data,
                                                    const IndexT* dst_indices,
                                                    const int* dst_count,
                                                    const int64_t* r_bcastoff,
                                                    T* e_grad,
                                                    int64_t index_size,
                                                    int64_t r_len,
                                                    int64_t out_len,
                                                    bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    T* e_grad_off = e_grad + ty * r_len;
    const T* out_grad_off = out_grad_data + dst * out_len;
    while (tx < out_len) {
      int64_t e_add = use_bcast ? r_bcastoff[tx] : tx;
      paddle::platform::CudaAtomicAdd(
          e_grad_off + e_add,
          out_grad_off[tx] / static_cast<T>(dst_count[dst]));
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// e_grad: backward mean for mul.
template <typename T, typename IndexT>
__global__ void ManipulateMeanGradCUDAKernelForMulE(const T* x_data,
                                                    const T* out_grad_data,
                                                    const IndexT* src_indices,
                                                    const IndexT* dst_indices,
                                                    const int* dst_count,
                                                    const int64_t* l_bcastoff,
                                                    const int64_t* r_bcastoff,
                                                    T* e_grad,
                                                    int64_t index_size,
                                                    int64_t l_len,
                                                    int64_t r_len,
                                                    int64_t out_len,
                                                    bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + src * l_len;
    T* e_grad_off = e_grad + ty * r_len;
    const T* out_grad_off = out_grad_data + dst * out_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? l_bcastoff[tx] : tx;
      int64_t e_add = use_bcast ? r_bcastoff[tx] : tx;
      paddle::platform::CudaAtomicAdd(
          e_grad_off + e_add,
          out_grad_off[tx] * x_off[x_add] / static_cast<T>(dst_count[dst]));
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// x_grad, e_grad: backward min and max for add.
template <typename T, typename IndexT>
__global__ void ManipulateMinMaxGradCUDAKernelForAdd(const T* x_data,
                                                     const T* e_data,
                                                     const T* out,
                                                     const T* out_grad,
                                                     const IndexT* src_indices,
                                                     const IndexT* dst_indices,
                                                     const int64_t* xbcast_off,
                                                     const int64_t* ebcast_off,
                                                     T* x_grad,
                                                     T* e_grad,
                                                     int64_t index_size,
                                                     int64_t x_len,
                                                     int64_t e_len,
                                                     int64_t out_len,
                                                     bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + dst * x_len;
    const T* e_off = e_data + ty * e_len;
    const T* out_off = out + src * out_len;
    const T* out_grad_off = out_grad + src * out_len;
    T* x_grad_off = x_grad + dst * x_len;
    T* e_grad_off = e_grad + ty * e_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? xbcast_off[tx] : tx;
      int64_t e_add = use_bcast ? ebcast_off[tx] : tx;
      T val = x_off[x_add] + e_off[e_add];
      paddle::platform::CudaAtomicAdd(
          x_grad_off + x_add,
          out_grad_off[tx] * static_cast<T>(val == out_off[tx]));
      paddle::platform::CudaAtomicAdd(
          e_grad_off + e_add,
          out_grad_off[tx] * static_cast<T>(val == out_off[tx]));
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// x_grad, e_grad: backward min and max for mul.
template <typename T, typename IndexT>
__global__ void ManipulateMinMaxGradCUDAKernelForMul(const T* x_data,
                                                     const T* e_data,
                                                     const T* out,
                                                     const T* out_grad,
                                                     const IndexT* src_indices,
                                                     const IndexT* dst_indices,
                                                     const int64_t* xbcast_off,
                                                     const int64_t* ebcast_off,
                                                     T* x_grad,
                                                     T* e_grad,
                                                     int64_t index_size,
                                                     int64_t x_len,
                                                     int64_t e_len,
                                                     int64_t out_len,
                                                     bool use_bcast) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + dst * x_len;
    const T* e_off = e_data + ty * e_len;
    const T* out_off = out + src * out_len;
    const T* out_grad_off = out_grad + src * out_len;
    T* x_grad_off = x_grad + dst * x_len;
    T* e_grad_off = e_grad + ty * e_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? xbcast_off[tx] : tx;
      int64_t e_add = use_bcast ? ebcast_off[tx] : tx;
      T val = x_off[x_add] * e_off[e_add];
      paddle::platform::CudaAtomicAdd(
          x_grad_off + x_add,
          out_grad_off[tx] * static_cast<T>(val == out_off[tx]) * e_off[e_add]);
      paddle::platform::CudaAtomicAdd(
          e_grad_off + e_add,
          out_grad_off[tx] * static_cast<T>(val == out_off[tx]) * x_off[x_add]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

}  // namespace phi
