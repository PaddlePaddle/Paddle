// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/impl/sequence_softmax_kernel_impl.h"

namespace phi {

template <typename T, int BlockDim>
using BlockReduce = cub::BlockReduce<T, BlockDim>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

template <typename T, int BlockDim>
__global__ void sequence_softmax_kernel(const T *in_data,
                                        const size_t *ref_lod,
                                        const size_t src_height,
                                        T *out_data) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;
  __shared__ T shared_max_data;
  __shared__ T shared_sum_data;

  for (int i = blockIdx.x; i < src_height; i += gridDim.x) {
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
      sum_data += phi::funcs::real_exp(ele - shared_max_data);
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
      ele = phi::funcs::real_exp(ele - shared_max_data) / shared_sum_data;
      out_data[start + tid] = ele;
    }
  }
}

template <typename T>
struct SequenceSoftmaxFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext &context,
                  const DenseTensor &x,
                  const phi::Vector<size_t> &ref_lod, /*referenced lod*/
                  DenseTensor *out) {
    int height = ref_lod.size() - 1;

    const int kThreadsPerBlock = 32;
    int thread_x = kThreadsPerBlock;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    dim3 block_size(thread_x);
    dim3 grid_size(max_blocks);
    phi::MixVector<size_t> mixv_ref_lod(&ref_lod);
    sequence_softmax_kernel<T, kThreadsPerBlock>
        <<<grid_size, block_size, 0, context.stream()>>>(
            x.data<T>(),
            mixv_ref_lod.CUDAData(context.GetPlace()),
            height,
            context.Alloc<T>(out));
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(sequence_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::SequenceSoftmaxKernel,
                   float,
                   double) {}
