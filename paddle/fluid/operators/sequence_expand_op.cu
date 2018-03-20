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
#include "paddle/fluid/operators/sequence_expand_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void sequence_expand_kernel(const T* x_data, T* out_data,
                                       const size_t* lod, size_t lod_size,
                                       size_t element_len) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < static_cast<int>(lod_size - 1);
       tid_x += blockDim.x * gridDim.x) {
    int scale = lod[tid_x + 1] - lod[tid_x];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < scale; tid_y += blockDim.y * gridDim.y) {
      int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
      int item_start = tid_x / element_len;
      for (; tid_z < element_len; tid_z += blockDim.z * gridDim.z) {
        out_data[item_start * scale + tid_z] = x_data[item_start + tid_z];
      }
    }
  }
}

template <typename T>
__global__ void sequence_expand_grad_kernel(const T* dout_data, T* dx_data,
                                            const size_t* lod, size_t lod_size,
                                            size_t element_len,
                                            size_t dout_size) {
  extern __shared__ T shm[];
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < static_cast<int>(lod_size - 1);
       tid_x += blockDim.x * gridDim.x) {
    int scale = lod[tid_x + 1] - lod[tid_x];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < scale; tid_y += blockDim.y * gridDim.y) {
      int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
      int item_start = tid_x / element_len;
      for (; tid_z < element_len; tid_z += blockDim.z * gridDim.z) {
        shm[item_start + tid_z] += dout_data[item_start * scale + tid_z];
      }
    }
  }
  // synchronize before write to dx
  __syncthreads();
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < static_cast<int>(dout_size); idx += blockDim.x * gridDim.x) {
    dx_data[idx] = shm[idx];
  }
}

template <typename T>
struct SequenceExpandFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const LoDTensor& x, LoDTensor* out) {
    auto x_dims = x.dims();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto out_starts = out->lod().back();

    dim3 block_size(16, 32, element_len);
    dim3 grid_size(10, 10);
    sequence_expand_kernel<<<grid_size, block_size, 0, context.stream()>>>(
        x.data<T>(), out->mutable_data<T>(context.GetPlace()),
        out_starts.CUDAData(context.GetPlace()), out_starts.size(),
        element_len);
  }
};

template <typename T>
struct SequenceExpandGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const LoDTensor& x, const LoDTensor& out,
                  const LoDTensor& dout, LoDTensor* dx) {
    auto x_dims = x.dims();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    auto out_starts = out.lod().back();

    dim3 block_size(16, 32, element_len);
    dim3 grid_size(10, 10);
    size_t out_size = framework::product(dx->dims());
    sequence_expand_grad_kernel<<<grid_size, block_size, out_size * sizeof(T),
                                  context.stream()>>>(
        dout.data<T>(), dx->mutable_data<T>(context.GetPlace()),
        out_starts.CUDAData(context.GetPlace()), out_starts.size(), element_len,
        out_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext, float>);
