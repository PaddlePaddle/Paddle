// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/gather_tree_kernel.h"

#include <algorithm>
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void GatherTree(const T *ids_data,
                           const T *parents_data,
                           T *out_data,
                           const int64_t max_length,
                           const int64_t batch_size,
                           const int64_t beam_size) {
  CUDA_KERNEL_LOOP(i, batch_size * beam_size) {
    int batch = i / beam_size;
    int beam = i % beam_size;
    auto idx =
        (max_length - 1) * batch_size * beam_size + batch * beam_size + beam;
    out_data[idx] = ids_data[idx];
    auto parent = parents_data[idx];
    for (int step = max_length - 2; step >= 0; step--) {
      idx = step * batch_size * beam_size + batch * beam_size;
      out_data[idx + beam] = ids_data[idx + parent];
      parent = parents_data[idx + parent];
    }
  }
}

template <typename T, typename Context>
void GatherTreeKernel(const Context &dev_ctx,
                      const DenseTensor &ids,
                      const DenseTensor &parents,
                      DenseTensor *out) {
  const auto *ids_data = ids.data<T>();
  const auto *parents_data = parents.data<T>();
  T *out_data = dev_ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_NOT_NULL(ids_data,
                          phi::errors::InvalidArgument(
                              "Input(Ids) of gather_tree should not be null."));

  PADDLE_ENFORCE_NOT_NULL(
      parents_data,
      phi::errors::InvalidArgument(
          "Input(Parents) of gather_tree should not be null."));

  auto &ids_dims = ids.dims();
  int64_t max_length = ids_dims[0];
  int64_t batch_size = ids_dims[1];
  int64_t beam_size = ids_dims[2];

  const int block = 512;
  int max_threads =
      std::min(static_cast<int64_t>(dev_ctx.GetMaxPhysicalThreadCount()),
               batch_size * beam_size);
  const int grid = std::max(max_threads / block, 1);
  GatherTree<<<grid, block>>>(
      ids_data, parents_data, out_data, max_length, batch_size, beam_size);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gather_tree, GPU, ALL_LAYOUT, phi::GatherTreeKernel, int, int64_t) {}
