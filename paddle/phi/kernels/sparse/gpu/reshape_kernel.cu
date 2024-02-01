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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename IntT>
__global__ void ReshapeCooCudaKernel(const IntT* x_indices_data,
                                     const int num_x_sparse_part_dims,
                                     const int num_out_sparse_part_dims,
                                     const int64_t x_nnz,
                                     const int64_t* x_sparse_part_strides,
                                     const int64_t* out_sparse_part_strides,
                                     IntT* out_indices_data) {
  CUDA_KERNEL_LOOP_TYPE(j, x_nnz, int64_t) {
    IntT location = 0;
    for (int i = 0; i < num_x_sparse_part_dims; ++i) {
      location += x_indices_data[i * x_nnz + j] *
                  static_cast<IntT>(x_sparse_part_strides[i]);
    }
    for (int i = 0; i < num_out_sparse_part_dims; ++i) {
      out_indices_data[i * x_nnz + j] =
          location / static_cast<IntT>(out_sparse_part_strides[i]);
      location %= static_cast<IntT>(out_sparse_part_strides[i]);
    }
  }
}

template <typename T, typename IntT, typename Context>
void ReshapeCooGPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const phi::IntArray& shape,
                         SparseCooTensor* out) {
  int64_t x_nnz = x.nnz();
  std::vector<int> new_shape(shape.GetData().begin(), shape.GetData().end());
  phi::DDim out_dims = x.dims().reshape(new_shape);
  //  get sparse part dimensions of x and out
  std::vector<int64_t> x_sparse_part_dims;
  std::vector<int64_t> out_sparse_part_dims;
  for (int i = 0; i < x.sparse_dim(); ++i) {
    x_sparse_part_dims.push_back(x.dims()[i]);
  }
  for (int i = 0; i < out_dims.size() - x.dense_dim(); ++i) {
    out_sparse_part_dims.push_back(out_dims[i]);
  }

  DenseTensor out_indices = Empty<IntT, Context>(
      dev_ctx, {static_cast<int64_t>(out_sparse_part_dims.size()), x_nnz});
  DenseTensor out_values(x.values());
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of out indices
  const auto* x_indices_data = x.indices().data<IntT>();
  auto* out_indices_data = out_indices.data<IntT>();
  const phi::DDim& x_sparse_part_strides =
      common::stride(common::make_ddim(x_sparse_part_dims));
  const phi::DDim& out_sparse_part_strides =
      common::stride(common::make_ddim(out_sparse_part_dims));

  int64_t *destination_x_sparse_part_strides,
      *destination_out_sparse_part_strides;

  auto destination_x_sparse_part_strides_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * x_sparse_part_strides.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  destination_x_sparse_part_strides = reinterpret_cast<int64_t*>(
      destination_x_sparse_part_strides_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     reinterpret_cast<void*>(destination_x_sparse_part_strides),
                     phi::CPUPlace(),
                     x_sparse_part_strides.Get(),
                     sizeof(int64_t) * x_sparse_part_strides.size(),
                     dev_ctx.stream());

  auto destination_out_sparse_part_strides_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * out_sparse_part_strides.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  destination_out_sparse_part_strides = reinterpret_cast<int64_t*>(
      destination_out_sparse_part_strides_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     destination_out_sparse_part_strides,
                     phi::CPUPlace(),
                     out_sparse_part_strides.Get(),
                     sizeof(int64_t) * out_sparse_part_strides.size(),
                     dev_ctx.stream());

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_nnz, 1);
  ReshapeCooCudaKernel<<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         dev_ctx.stream()>>>(
      x_indices_data,
      x_sparse_part_dims.size(),
      out_sparse_part_dims.size(),
      x_nnz,
      destination_x_sparse_part_strides,
      destination_out_sparse_part_strides,
      out_indices_data);
}

template <typename T, typename Context>
void ReshapeCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const phi::IntArray& shape,
                      SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "ReshapeCooGPUKernel", ([&] {
        ReshapeCooGPUKernel<T, data_t, Context>(dev_ctx, x, shape, out);
      }));
}

// just copy from paddle\phi\kernels\sparse\cpu\reshape_kernel.cc
template <typename T, typename Context>
void ReshapeCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const phi::IntArray& shape,
                      SparseCsrTensor* out) {
  // transform csr format to coo format, and then use coo kernel
  const SparseCooTensor x_coo = CsrToCoo<T, Context>(dev_ctx, x);
  SparseCooTensor out_coo;
  ReshapeCooKernel<T, Context>(dev_ctx, x_coo, shape, &out_coo);
  CooToCsrKernel<T, Context>(dev_ctx, out_coo, out);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(reshape_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(reshape_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
