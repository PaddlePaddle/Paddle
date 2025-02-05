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

#include "paddle/phi/backends/gpu/gpu_context.h"

#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/partial_concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"

namespace phi {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

template <class T>
__global__ void ConcatPartialCUDAKernel(T **in,
                                        T *out,
                                        int64_t all_length,
                                        int64_t in_batch_len,
                                        int64_t start_index,
                                        int64_t out_batch_len,
                                        int64_t part_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < all_length) {
    int64_t bs_id = id / out_batch_len;
    int64_t bs_index = id % out_batch_len;
    int64_t var_id = bs_index / part_length;
    int64_t part_index = bs_index % part_length;
    int64_t in_id = start_index + part_index;
    const T *tmp = in[var_id];
    out[id] = tmp[bs_id * in_batch_len + in_id];
    id += blockDim.x * gridDim.x;
  }
}

template <typename T, typename Context>
void PartialConcatOpCUDAKernel(const Context &dev_ctx,
                               const std::vector<const DenseTensor *> &x,
                               int start_index,
                               int length,
                               DenseTensor *out) {
  auto in_vars = x;
  PADDLE_ENFORCE_EQ(in_vars[0] != nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "The input of partial concat should not be null."));

  auto input_dim = in_vars[0]->dims();
  PADDLE_ENFORCE_EQ(input_dim.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Only supports 2-D array with batch size in the 1st "
                        "dimension and data in the 2nd."));
  auto in_size = input_dim[1];
  // may be negative
  start_index = ComputeStartIndex(start_index, in_size);

  auto partial_len = length;
  if (partial_len < 0) {
    partial_len = in_size - start_index;
  }

  int in_num = in_vars.size();
  int batch_size = input_dim[0];
  int out_batch_len = partial_len * in_num;
  int all_length = batch_size * out_batch_len;

  constexpr size_t theory_sm_threads = 1024;
  auto stream = dev_ctx.stream();
  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  int grids;
  int blocks;
  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = CEIL_DIV(length, tile_size);
    blocks = tile_size;
  };

  T *out_data = dev_ctx.template Alloc<T>(out);

  std::vector<const T *> in_data;
  for (int i = 0; i < in_num; ++i) in_data.emplace_back(in_vars[i]->data<T>());

  auto tmp_in_array = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      in_data.size() * sizeof(T *),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          tmp_in_array->ptr(),
                          phi::CPUPlace(),
                          reinterpret_cast<void *>(in_data.data()),
                          in_data.size() * sizeof(T *),
                          dev_ctx.stream());

  T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
  ComputeKernelParameter(all_length);
  ConcatPartialCUDAKernel<T><<<grids, blocks, 0, stream>>>(in_array_data,
                                                           out->data<T>(),
                                                           all_length,
                                                           in_size,
                                                           start_index,
                                                           out_batch_len,
                                                           partial_len);
}
}  // namespace phi

PD_REGISTER_KERNEL(partial_concat,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialConcatOpCUDAKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
