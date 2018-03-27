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

#define EIGEN_USE_GPU
#include <stdio.h>
#include <algorithm>
#include "paddle/fluid/operators/sequence_expand_op.h"
#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void sequence_expand_kernel(const T* x_data, T* out_data,
                                       const size_t* lod,
                                       const size_t* out_offset,
                                       size_t lod_size, size_t element_len,
                                       size_t x_size) {
  int bid_x = blockIdx.x;
  if (bid_x > lod_size) return;
  int repeats = lod[bid_x];
  int offset = out_offset[bid_x];
  for (int tid_y = threadIdx.y; tid_y < repeats; tid_y += blockDim.y) {
    for (int tid_x = threadIdx.x; tid_x < element_len; tid_x += blockDim.x) {
      out_data[(offset + tid_y) * element_len + tid_x] =
          x_data[bid_x * element_len + tid_x];
    }
  }
}

template <typename T>
__global__ void sequence_expand_grad_kernel(const T* dout_data, T* dx_data,
                                            const size_t* lod,
                                            const size_t* out_offset,
                                            size_t lod_size, size_t element_len,
                                            size_t dout_size, size_t dx_size) {
  // reduce visit memory time.
  // dout_shm = [0 - dout_size-1], dx_shm = [dout_size-1, dout_size + dx_size-1]
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
      threadIdx.y == 0) {
    printf("lod_size=%ld, element_size=%ld, dout_size=%ld, dx_size=%ld\n",
           lod_size, element_len, dout_size, dx_size);
  }
  extern __shared__ T shm[];
  T* dout_shm = shm;
  T* dx_shm = &shm[dout_size];

  // int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = 0; idx < dout_size; ++idx) {
    if (idx < dx_size) {
      dx_shm[idx] = 0.0;
    }
    if (idx < dout_size) {
      dout_shm[idx] = dout_data[idx];
    }
  }

  int bid_x = blockIdx.x;
  if (bid_x > lod_size) return;
  int repeats = lod[bid_x];
  int offset = out_offset[bid_x];
  if (threadIdx.x == 0) {
    printf("repeats=%d, offset=%ld\n", repeats, offset);
  }
  for (int tid_y = threadIdx.y; tid_y < repeats; tid_y += blockDim.y) {
    for (int tid_x = threadIdx.x; tid_x < element_len; tid_x += blockDim.x) {
      T val = dout_shm[(offset + tid_y) * element_len + tid_x];
      platform::CudaAtomicAdd(&dx_shm[bid_x * element_len + tid_x], val);
      int dx_idx = bid_x * element_len + tid_x;
      int dout_idx = (offset + tid_y) * element_len + tid_x;
      printf("dx_idx=%d, dout_idx=%d, dx_data=%f, dout_data=%f, val=%f \n",
             dx_idx, dout_idx, dx_shm[dx_idx], dout_shm[dout_idx], val);
    }
  }
  __syncthreads();
  // copy shared memory back to dx
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < dx_size;
       idx += blockDim.x * gridDim.x) {
    dx_data[idx] = dx_shm[idx];
  }
}

template <typename T>
struct SequenceExpandFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const LoDTensor& x, LoDTensor* out) {
    auto x_dims = x.dims();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    auto lod = out->lod().back();
    framework::Vector<size_t> out_lod;
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      out_lod.push_back(lod[i + 1] - lod[i]);
    }

    int thread_x = std::max(static_cast<int>(element_len), 32);
    int block_x = static_cast<int>(out_lod.size());
    dim3 block_size(thread_x, 1024 / thread_x);
    dim3 grid_size(block_x, 1);
    sequence_expand_kernel<<<grid_size, block_size, 0, context.stream()>>>(
        x.data<T>(), out->mutable_data<T>(context.GetPlace()),
        out_lod.CUDAData(context.GetPlace()), lod.CUDAData(context.GetPlace()),
        out_lod.size(), element_len, framework::product(x_dims));
  }
};

template <typename T>
struct SequenceExpandGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const LoDTensor& x, const LoDTensor& out,
                  const LoDTensor& dout, LoDTensor* dx) {
    auto x_dims = x.dims();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    auto lod = out.lod().back();
    framework::Vector<size_t> out_lod;
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      out_lod.push_back(lod[i + 1] - lod[i]);
    }
    size_t dout_size = framework::product(dout.dims());
    size_t dx_size = framework::product(dx->dims());

    int thread_x = std::max(static_cast<int>(element_len), 32);
    dim3 block_size(thread_x, 1024 / thread_x);
    int block_x = static_cast<int>(out_lod.size());
    dim3 grid_size(block_x, 1);
    sequence_expand_grad_kernel<<<grid_size, block_size,
                                  (dout_size + dx_size) * sizeof(T),
                                  context.stream()>>>(
        dout.data<T>(), dx->mutable_data<T>(context.GetPlace()),
        out_lod.CUDAData(context.GetPlace()), lod.CUDAData(context.GetPlace()),
        out_lod.size(), element_len, dout_size, dx_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
