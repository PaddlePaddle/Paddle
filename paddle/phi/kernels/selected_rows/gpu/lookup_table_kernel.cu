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

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {
namespace sr {
template <typename T,
          int BlockDimX,
          int BlockDimY,
          int GridDimX,
          bool PaddingFlag>
__global__ void LookupTable(T *output,
                            const T *table,
                            const int64_t *ids,
                            const int64_t N,
                            const int64_t K,
                            const int64_t D,
                            const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    PADDLE_ENFORCE(
        id >= 0,
        "Variable value (input) of OP(fluid.layers.embedding) "
        "expected >= 0 and < %ld, but got %ld. Please check input value.",
        N,
        id);
    PADDLE_ENFORCE(
        id < N,
        "Variable value (input) of OP(fluid.layers.embedding) "
        "expected >= 0 and < %ld, but got %ld. Please check input value.",
        N,
        id);
    T *out = output + idy * D;
    const T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<T>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T, typename Context>
void LookupTableCUDAKernel(const Context &dev_ctx,
                           const SelectedRows &w,
                           const DenseTensor &ids_in,
                           bool is_sparse,
                           bool is_distributed UNUSED,
                           int64_t padding_idx,
                           bool remote_prefetch UNUSED,
                           const std::string &entry_config UNUSED,
                           bool is_test,
                           const std::string &entry UNUSED,
                           const std::string &table_class UNUSED,
                           const std::vector<std::string> &table_names UNUSED,
                           int trainer_id UNUSED,
                           bool grad_inplace UNUSED,
                           const std::vector<std::string> &epmap UNUSED,
                           const std::vector<int64_t> &height_sections UNUSED,
                           SelectedRows *out) {
  auto *table_t = &w;
  auto *ids_t = &ids_in;
  auto *output_t = out;

  size_t N = table_t->dims()[0];
  size_t D = table_t->dims()[1];
  size_t K = ids_t->numel();

  auto *ids = ids_t->data<int64_t>();
  auto *table = table_t->value().data<T>();
  auto *output = dev_ctx.template Alloc<T>(output_t);

#ifdef PADDLE_WITH_HIP
  dim3 threads(64, 4);
#else
  dim3 threads(128, 8);
#endif  // PADDLE_WITH_HIP
  dim3 grids(8, 1);
#ifdef PADDLE_WITH_HIP
  if (padding_idx == -1)
    LookupTable<T, 64, 4, 8, false><<<grids, threads, 0, dev_ctx.stream()>>>(
        output, table, ids, N, K, D, padding_idx);
  else
    LookupTable<T, 64, 4, 8, true><<<grids, threads, 0, dev_ctx.stream()>>>(
        output, table, ids, N, K, D, padding_idx);
#else
  if (padding_idx == -1)
    LookupTable<T, 128, 8, 8, false><<<grids, threads, 0, dev_ctx.stream()>>>(
        output, table, ids, N, K, D, padding_idx);
  else
    LookupTable<T, 128, 8, 8, true><<<grids, threads, 0, dev_ctx.stream()>>>(
        output, table, ids, N, K, D, padding_idx);
#endif  // PADDLE_WITH_HIP
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(lookup_table_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::LookupTableCUDAKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int8_t,
                   int16_t) {}
