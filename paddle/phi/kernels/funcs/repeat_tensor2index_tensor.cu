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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/exclusive_scan.h"
#include "paddle/phi/kernels/impl/repeat_interleave_kernel_impl.h"
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
void RepeatsTensor2IndexTensorFunctor<phi::GPUContext, RepeatsT>::operator()(
    const phi::GPUContext &dev_ctx,
    const DenseTensor &repeats,
    DenseTensor *index) {
#if defined(__NVCC__)
  const RepeatsT *repeats_ptr = repeats.data<RepeatsT>();
  int64_t num_reps = repeats.dims()[0];

  // compute prefix sum of repeats to get start index of each repeat
  DenseTensor prefix;
  prefix.Resize(common::make_ddim({num_reps}));
  dev_ctx.template Alloc<RepeatsT>(&prefix);
  auto *prefix_ptr = prefix.data<RepeatsT>();

  auto stream = dev_ctx.stream();
  phi::funcs::
      CubExclusiveScan<const RepeatsT *, RepeatsT *, cub::Sum, RepeatsT>(
          repeats_ptr,
          prefix_ptr,
          num_reps,
          static_cast<RepeatsT>(0),
          cub::Sum(),
          dev_ctx);

  // get last prefix and repeat to compute total size of index tensor
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
  dev_ctx.template Alloc<RepeatsT>(index);

  if (total_size == 0) {
    return;
  }

  RepeatsT *index_ptr = index->data<RepeatsT>();
  fill_array_kernel<<<(num_reps + PADDLE_CUDA_NUM_THREADS - 1) /
                          PADDLE_CUDA_NUM_THREADS,
                      PADDLE_CUDA_NUM_THREADS,
                      0,
                      stream>>>(index_ptr, prefix_ptr, repeats_ptr, num_reps);
#else
  DenseTensor repeats_cpu_copy;
  if (repeats.place().GetType() != phi::AllocationType::CPU) {
    phi::Copy(dev_ctx, repeats, phi::CPUPlace(), true, &repeats_cpu_copy);
  }
  const RepeatsT *repeats_data =
      repeats.place().GetType() == phi::AllocationType::CPU
          ? repeats.data<RepeatsT>()
          : repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    PADDLE_ENFORCE_GE(repeats_data[i],
                      0,
                      common::errors::InvalidArgument(
                          "repeats must grater or equal than 0, but got %d",
                          repeats_data[i]));
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<RepeatsT>(index_vec, dev_ctx, index);
#endif
}

template class RepeatsTensor2IndexTensorFunctor<phi::GPUContext, int>;
template class RepeatsTensor2IndexTensorFunctor<phi::GPUContext, int64_t>;

}  // namespace funcs
}  // namespace phi
