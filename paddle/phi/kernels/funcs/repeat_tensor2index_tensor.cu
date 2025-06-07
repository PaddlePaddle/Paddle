// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace funcs {

template <typename T>
__global__ void fill_array_kernel(T *output,
                                  const T *prefix,
                                  const T *repeats,
                                  int64_t n) {
  T idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T start = prefix[idx];
    T count = repeats[idx];

    for (T j = 0; j < count; j++) {
      output[start + j] = idx;
    }
  }
}

template <typename RepeatsT>
class RepeatsTensor2IndexTensorFunctor<phi::GPUContext, RepeatsT> {
 public:
  void operator()(const phi::GPUContext &ctx,
                  const DenseTensor &repeats,
                  DenseTensor *index) {
    const RepeatsT *repeats_ptr = repeats.data<RepeatsT>();
    int64_t num_reps = repeats.dims()[0];

    // compute prefix sum of repeats to get start index of each repeat
    RepeatsT *prefix_ptr;
    cudaMalloc(&prefix_ptr, num_reps * sizeof(RepeatsT));

    thrust::device_ptr<const RepeatsT> input_dev_ptr(repeats_ptr);
    thrust::device_ptr<RepeatsT> output_dev_ptr(prefix_ptr);

    cudaStream_t stream = ctx.stream();

    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           input_dev_ptr,
                           input_dev_ptr + num_reps,
                           output_dev_ptr);

    // get last prefix and repeat to compute total size of index tensor because
    // thrust::exclusive_scan does not return the last value
    RepeatsT last_prefix = 0;
    RepeatsT last_repeat = 0;
    cudaMemcpyAsync(&last_prefix,
                    prefix_ptr + num_reps - 1,
                    sizeof(RepeatsT),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(&last_repeat,
                    repeats_ptr + num_reps - 1,
                    sizeof(RepeatsT),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    int64_t total_size =
        static_cast<int64_t>(last_prefix) + static_cast<int64_t>(last_repeat);

    // resize & alloc index tensor
    index->Resize({total_size});
    ctx.template Alloc<RepeatsT>(index);

    if (total_size == 0) {
      cudaFree(prefix_ptr);
      return;
    }

    RepeatsT *index_ptr = index->data<RepeatsT>();
    int block_size = 256;
    int grid_size = (num_reps + block_size - 1) / block_size;
    fill_array_kernel<<<grid_size, block_size, 0, stream>>>(
        index_ptr, prefix_ptr, repeats_ptr, num_reps);

    cudaFree(prefix_ptr);
  }
};

using GPU = phi::GPUContext;
template class RepeatsTensor2IndexTensorFunctor<GPU, int>;
template class RepeatsTensor2IndexTensorFunctor<GPU, int64_t>;

}  // namespace funcs
}  // namespace phi
