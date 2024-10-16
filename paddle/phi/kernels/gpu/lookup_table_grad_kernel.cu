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
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableGrad(T *table,
                                const T *output,
                                const int64_t *ids,
                                const int64_t N,
                                const int64_t K,
                                const int64_t D) {
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
    const T *out = output + idy * D;
    T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      phi::CudaAtomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T, typename Context>
void LookupTableGradCUDAKernel(
    const Context &dev_ctx,
    const DenseTensor &w,
    const DenseTensor &ids_in,
    const DenseTensor &out_grad,
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
    DenseTensor *w_grad) {
  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.

  auto ids_t = &ids_in;
  auto d_output_t = &out_grad;
  auto d_table_t = w_grad;

  int N = d_table_t->dims()[0];
  int D = d_table_t->dims()[1];
  int K = ids_t->numel();
  const int64_t *ids = ids_t->data<int64_t>();
  const T *d_output = d_output_t->data<T>();
  T *d_table = dev_ctx.template Alloc<T>(d_table_t);

  auto t = phi::EigenVector<T>::Flatten(*d_table_t);
  t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

#ifdef PADDLE_WITH_HIP
  dim3 threads(64, 4);
#else
  dim3 threads(128, 8);
#endif  // PADDLE_WITH_HIP
  dim3 grids(8, 1);

#ifdef PADDLE_WITH_HIP
  LookupTableGrad<T, 64, 4, 8><<<grids, threads, 0, dev_ctx.stream()>>>(
      d_table, d_output, ids, N, K, D);
#else
  LookupTableGrad<T, 128, 8, 8><<<grids, threads, 0, dev_ctx.stream()>>>(
      d_table, d_output, ids, N, K, D);
#endif  // PADDLE_WITH_HIP
}

template <typename T, typename Context>
void LookupTableSparseGradCUDAKernel(
    const Context &dev_ctx,
    const DenseTensor &w,
    const DenseTensor &ids_in,
    const DenseTensor &out_grad,
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
    SelectedRows *w_grad) {
  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.

  auto *ids = &ids_in;
  auto *table = &w;
  auto *d_output = &out_grad;
  auto *d_table = w_grad;

  auto *ids_data = ids->data<int64_t>();
  int64_t ids_num = ids->numel();

  auto stream = dev_ctx.stream();
  // copy GPU memory to CPU pinned memory
  phi::Vector<int64_t> new_rows;
  new_rows.resize(ids_num);
  auto gpu_place = dev_ctx.GetPlace();

  // TODO(yuyang18): Strange code here.
  phi::MixVector<int64_t> mixv_new_rows(&new_rows);
  phi::memory_utils::Copy(gpu_place,
                          mixv_new_rows.CUDAMutableData(dev_ctx.GetPlace()),
                          gpu_place,
                          ids_data,
                          ids_num * sizeof(int64_t),
                          stream);
  mixv_new_rows.CopyToCPU();
  d_table->set_rows(new_rows);

  auto *d_table_value = d_table->mutable_value();
  d_table_value->Resize({ids_num, table->dims()[1]});
  dev_ctx.template Alloc<T>(d_table_value);

  auto *d_table_data = d_table_value->data<T>();
  auto *d_output_data = d_output->data<T>();
  auto d_output_dims = d_output->dims();
  auto d_output_dims_2d =
      common::flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
  PADDLE_ENFORCE_EQ(d_table_value->dims(),
                    d_output_dims_2d,
                    common::errors::InvalidArgument(
                        "ShapeError: The shape of lookup_table@Grad and "
                        "output@Grad should be same. "
                        "But received lookup_table@Grad's shape = [%s], "
                        "output@Grad's shape = [%s].",
                        d_table_value->dims(),
                        d_output_dims_2d));
  phi::memory_utils::Copy(gpu_place,
                          d_table_data,
                          gpu_place,
                          d_output_data,
                          d_output->numel() * sizeof(T),
                          stream);
}
}  // namespace phi

PD_REGISTER_KERNEL(lookup_table_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LookupTableGradCUDAKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(lookup_table_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LookupTableSparseGradCUDAKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
