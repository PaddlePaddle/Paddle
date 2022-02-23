/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/sequence_ops/sequence_softmax_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T, int BlockDim>
using BlockReduce = cub::BlockReduce<T, BlockDim>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

template <typename T, int BlockDim>
__global__ void sequence_softmax_kernel(const T *in_data, const size_t *ref_lod,
                                        const size_t src_hight, T *out_data) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;
  __shared__ T shared_max_data;
  __shared__ T shared_sum_data;

  for (int i = blockIdx.x; i < src_hight; i += gridDim.x) {
    size_t start = ref_lod[i];
    size_t span = ref_lod[i + 1] - start;

    // Find the max ele
    T max_ele = -FLT_MAX;
    for (int tid = threadIdx.x; tid < span; tid += blockDim.x) {
      T ele = in_data[start + tid];
      max_ele = max_ele > ele ? max_ele : ele;
    }
    max_ele =
        BlockReduce<T, BlockDim>(temp_storage).Reduce(max_ele, cub::Max());
    if (threadIdx.x == 0) {
      shared_max_data = max_ele;
    }
    __syncthreads();

    // sum
    T sum_data = 0;
    for (int tid = threadIdx.x; tid < span; tid += blockDim.x) {
      T ele = in_data[start + tid];
      sum_data += real_exp(ele - shared_max_data);
    }
    sum_data =
        BlockReduce<T, BlockDim>(temp_storage).Reduce(sum_data, cub::Sum());
    if (threadIdx.x == 0) {
      shared_sum_data = sum_data;
    }
    __syncthreads();

    // get final resit
    for (int tid = threadIdx.x; tid < span; tid += blockDim.x) {
      T ele = in_data[start + tid];
      ele = real_exp(ele - shared_max_data) / shared_sum_data;
      out_data[start + tid] = ele;
    }
  }
}

template <typename T, int BlockDim>
__global__ void sequence_softmax_grad_kernel(const T *softmax_grad_data,
                                             const T *softmax_data,
                                             const size_t *ref_lod,
                                             const size_t src_hight,
                                             T *dx_data) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;
  __shared__ T shared_data;

  for (int i = blockIdx.x; i < src_hight; i += gridDim.x) {
    size_t start = ref_lod[i];
    size_t span = ref_lod[i + 1] - start;

    T result = 0;
    for (int tid = threadIdx.x; tid < span; tid += blockDim.x) {
      size_t idx = start + tid;
      T s_g_d = softmax_grad_data[idx];
      T s_d = softmax_data[idx];
      result += s_g_d * s_d;
    }
    result = BlockReduce<T, BlockDim>(temp_storage).Reduce(result, cub::Sum());
    if (threadIdx.x == 0) {
      shared_data = result;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < span; tid += blockDim.x) {
      size_t idx = start + tid;
      T s_g_d = softmax_grad_data[idx];
      T s_d = softmax_data[idx];
      dx_data[idx] = (s_g_d - shared_data) * s_d;
    }
  }
}

template <typename T>
struct SequenceSoftmaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext &context,
                  const LoDTensor &x,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  LoDTensor *out) {
    int height = ref_lod.size() - 1;

    const int kThreadsPerBlock = 32;
    int thread_x = kThreadsPerBlock;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    dim3 block_size(thread_x);
    dim3 grid_size(max_blocks);
    paddle::framework::MixVector<size_t> mixv_ref_lod(&ref_lod);
    sequence_softmax_kernel<
        T, kThreadsPerBlock><<<grid_size, block_size, 0, context.stream()>>>(
        x.data<T>(), mixv_ref_lod.CUDAData(context.GetPlace()), height,
        out->mutable_data<T>(context.GetPlace()));
  }
};

template <typename T>
struct SequenceSoftmaxGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext &context,
                  const LoDTensor &dout, const LoDTensor &out,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  LoDTensor *dx) {
    size_t height = ref_lod.size() - 1;

    const int kThreadsPerBlock = 32;
    int thread_x = kThreadsPerBlock;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    dim3 block_size(thread_x);
    dim3 grid_size(max_blocks);

    paddle::framework::MixVector<size_t> mixv_ref_lod(&ref_lod);
    sequence_softmax_grad_kernel<
        T, kThreadsPerBlock><<<grid_size, block_size, 0, context.stream()>>>(
        dout.data<T>(), out.data<T>(),
        mixv_ref_lod.CUDAData(context.GetPlace()), height,
        dx->mutable_data<T>(context.GetPlace()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax_grad,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext,
                                   double>);
