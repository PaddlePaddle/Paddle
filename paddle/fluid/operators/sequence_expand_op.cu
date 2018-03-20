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
__global__ sequence_expand_kernel(const T* x_data, T* out_data, size_t* lod,
                                  size_t element_len) {
  int BLOCK_SIZE = 1024;
  __shared__ T shm_lod[BLOCK_SIZE];
  for (int idx = threadIdx.x; idx < BLOCK_SIZE; ++idx) {
    shm_lod[idx] = lod[idx];
  }
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < lod.size();
       idx += blockDim.x * gridDim.x) {
    int scale = lod[i]
  }
}

template <typename T>
void SequenceExpandFunctor<platform::CPUDeviceContext, T>::operator()(
    const platform::CPUDeviceContext& context, const LoDTensor& x,
    LoDTensor* out) {
  x_dims = x.dims();
  size_t element_len = framework::product(x_dims) / x_dims[0];
  T* out_data = out->mutable_data<T>(context.GetPlace());
  auto out_starts = out->lod().back();

  const int kThreadsPerBlock = 1024;
  int block_cols = kThreadsPerBlock;
  if (out_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((out_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  dim3 block_size = dim3(block_cols, block_rows, 1);

  int max_threads = context.GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

  int grid_cols =
      std::min((out_cols + block_cols - 1) / block_cols, max_blocks);
  int grid_rows =
      std::min(max_blocks / grid_cols, std::max(out_rows / block_rows, 1));
  dim3 grid_size = dim3(grid_cols, grid_rows, 1);
  sequence_expand_kernel<<<grid_size, block_size, 0, context.stream()>>>(
      x.data<T>(), out->mutable_data<T>(context.GetPlace()),
      out_starts.CUDAData(context.GetPlace()), element_len);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CUDADeviceContext, float>);
