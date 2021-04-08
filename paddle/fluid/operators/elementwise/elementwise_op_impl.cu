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

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.h"

namespace paddle {
namespace operators {

template <int Vec_size, typename T, typename Functor>
__device__ void VectorizedKernelHelper(SameDimsData<T> data, int size,
                                       Functor func, int tid) {
  using VecType = AlignedVector<T, Vec_size>;
  const VecType *x = reinterpret_cast<const VecType *>(data.in0);
  const VecType *y = reinterpret_cast<const VecType *>(data.in1);
  VecType *z = reinterpret_cast<VecType *>(data.out);
  VecType x_vec, y_vec, z_vec;
  x_vec = x_vec[tid];
  y_vec = y_vec[tid];
  T *x_slr = reinterpret_cast<T *>(&x_vec);
  T *y_slr = reinterpret_cast<T *>(&y_vec);
  T *z_slr = reinterpret_cast<T *>(&z_vec);

#pragma unroll
  for (int i = 0; i < Vec_size; ++i) {
    z_slr[i] = x_slr[i] + y_slr[i];
  }

  z[tid] = z_vec;
}

template <int Vec_size, typename T, typename Functor>
__global__ void VectorizedSameDimsKernel(SameDimsData<T> data, int size,
                                         Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int remain = size - Vec_size * tid;
  if (remain >= Vec_size) {
    VectorizedKernelHelper<Vec_size>(data, size, func, tid);
  } else {
    ScalarKernelHelper(data, size, func, tid * Vec_size, remain);
  }
}

template <typename T, typename Functor>
__device__ void ScalarKernelHelper(SameDimsData<T> data, int size, Functor func,
                                   int start, int remain) {
  for (int i = 0; i < remain; ++i) {
    T x = (data.in0)[start + i];
    T y = (data.in1)[start + i];
    (data.out)[start + i] = x + y;
  }
}

template <typename T, typename Functor>
__global__ void ScalarSameDimsKernel(SameDimsData<T> data, int size,
                                     Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ScalarKernelHelper(data, size, func, tid, 0);
}

template <typename T, typename Functor>
void same_dims_launch_kernel(const framework::ExecutionContext &ctx,
                             SameDimsData<T> data, int64_t size, Functor func) {
  // calculate the max vec_size for all inputs and outputs
  int vec_size = data.GetVectorizedSize();
  int block_size = PADDLE_CUDA_THREAD_SIZE;
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  // cuda kernel
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  switch (vec_size) {
    case 8:
      VectorizedSameDimsKernel<8><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 4:
      VectorizedSameDimsKernel<4><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 2:
      VectorizedSameDimsKernel<2><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 1:
      ScalarSameDimsKernel<<<grid_size, block_size, 0, stream>>>(data, size,
                                                                 func);
      break;
    default:
      VLOG(3) << "Unsupported vectorized size!";
  }
}

}  // namespace operators
}  // namespace paddle
